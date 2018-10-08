import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from training_configs import *

# XXX
def printgrad(x, gin, gout):
    print(x)
    print('tensor: ', gin)
    print('grad: ', gout)
    print('')


def spread_loss(activations, target_index, m):
    """
    Spread loss penalizes activations closer to the margin
    than the target activation.
    The margin 'm' is starts at a low value and increases slowly over time.
    """

    # Get the activation of the target class (a scalar)
    target_index = target_index.unsqueeze(dim=1)
    target_activation = activations.gather(-1, target_index)
    # Calculate loss per class activation
    loss_i = torch.pow(F.relu(m - (target_activation - activations)), 2)
    # Don't sum over target's activation
    loss_i.scatter_(-1, target_index, 0)

    return loss_i.sum(dim=-1).mean()


### Define model ###
class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(1, A, 5, stride=2, padding=2)
        self.primary_caps = PrimaryCaps(A, B)
        self.caps_conv1 = ConvCaps(B, C, stride=2)
        self.caps_conv2 = ConvCaps(C, D, stride=1)
        self.caps_class = ConvCaps(D, E, kernel_size=4)
        #self.caps_class.activations_conv.register_backward_hook(printgrad) XXX


    def forward(self, x):
        # First layer
        x = F.relu(self.conv1(x))
        # Primary caps layer
        pose, activations = self.primary_caps(x)
        # ConvCaps1 layer
        pose, activations = self.caps_conv1(pose, activations)
        # ConvCaps2 layer
        pose, activations = self.caps_conv2(pose, activations)
        # Caps Classes layer
        pose, activations = self.caps_class(pose, activations)

        return activations.squeeze()


    def get_params(self):
        """
        TODO
        """
        return list(self.conv1.parameters()) +\
            list(self.primary_caps.parameters()) +\
            list(self.caps_conv1.parameters()) +\
            list(self.caps_conv2.parameters()) +\
            list(self.caps_class.parameters())


class PrimaryCaps(nn.Module):
    def __init__(self, num_caps_in, num_caps_out, kernel_size=1, stride=1):
        """
        The primary caps layer.
        """
        super(PrimaryCaps, self).__init__()
        self.num_caps_in = num_caps_in
        self.num_caps_out = num_caps_out

        out_channels = num_caps_out * (POSE_DIM * POSE_DIM + 1)
        self.conv = nn.Conv2d(num_caps_in, out_channels, kernel_size, stride)


    def forward(self, x):
        """
        x: (BATCH_SIZE, channels, filter_size, filter_size)
        """

        # Convolution layer, followed by seperation of pose and activations
        x = self.conv(x)

        # Kernel size after convolution (check last dims)
        filter_size = x.size()[-1]
        
        # Seperate activations and pose
        activations = x[:, :self.num_caps_out, ...]
        pose        = x[:, self.num_caps_out:, ...]

        # The following views should work
        activations = activations.view(BATCH_SIZE,
            filter_size, filter_size, self.num_caps_out)
        pose = pose.view(BATCH_SIZE,
            filter_size, filter_size, self.num_caps_out, POSE_DIM, POSE_DIM)
        
        return pose, torch.sigmoid(activations)


class ConvCaps(nn.Module):
    def __init__(self, num_caps_in, num_caps_out, kernel_size=K, stride=1):
        """
        A capsule convolutional layer.
        """
        super(ConvCaps, self).__init__()
        self.num_caps_in = num_caps_in
        self.num_caps_out = num_caps_out

        # The weight matrices that transform the poses into votes
        self.transformation_conv = nn.Conv2d(
            num_caps_in * POSE_DIM * POSE_DIM,
            num_caps_out * POSE_DIM * POSE_DIM,
            kernel_size=kernel_size, stride=stride, groups=POSE_DIM * POSE_DIM)
        # ??? XXX
        self.activations_conv = nn.Conv2d(num_caps_in, num_caps_in,
            kernel_size=kernel_size, stride=stride)

        # Routing probability for capsule i to route to capsule j
        self.routing_prob = torch.zeros(1, 1, 1, num_caps_in, num_caps_out)

        # Increment me after each routing iteration (inverse temperature)
        self.inv_temp = torch.ones(1)

        # Learned parameters (you can call them "benefits", so to speak)
        self.beta_v = nn.Parameter(torch.randn(1, 1, 1, num_caps_out, POSE_DIM * POSE_DIM))
        self.beta_a = nn.Parameter(torch.randn(1, 1, 1, num_caps_out))


    def forward(self, pose_in, activations_in):
        """
        Takes the state of a capsule layer, outputs the state of the next
        capsule layer, where state means: pose + activations.

        First, we tranform the pose to votes by the transformation weight.
        Then, we get the votes and activations and last capsule layer, and
        output the pose and activations of the next capsule layer.
        """

        def caps_conv(x, conv):
            # View
            x = x.view(*x.size()[:3], -1)
            x = x.permute([0, 3, 1, 2])
            # Transformation
            x = conv(x)
            # View
            x = x.permute([0, 2, 3, 1])

            return x

        # Pose
        votes = caps_conv(pose_in, self.transformation_conv)
        votes = votes.view(*votes.size()[:-1],
            1, -1, POSE_DIM * POSE_DIM)

        # Activations
        activations_in = caps_conv(activations_in, self.activations_conv)
        activations_in = activations_in.view(
            *activations_in.size()[:-1], -1)

        # Get the result from EM-routing, skip the variances
        pose_means, _, activations_out = self.EM_routing(votes, activations_in)
        # Pack pose means back into the matrix form
        pose_means = pose_means.view(*pose_means.size()[:-1], POSE_DIM, POSE_DIM)

        return pose_means, activations_out


    def EM_routing(self, votes, activations_in):
        # Reset TODO: ?
        self.routing_prob = torch.zeros(1, 1, 1, self.num_caps_in, self.num_caps_out)
        self.routing_prob += 1 / self.num_caps_out

        # Perform EM-routing for ROUTING_ITER iterations
        for _ in range(ROUTING_ITER):
            output_params = self.M_step(votes, activations_in)
            self.E_step(votes, output_params)  # Update r_ij
            # Decrease temperature
            self.inv_temp = self.inv_temp + 1

        return output_params


    def M_step(self, votes, activations_in):
        """
        votes:        (input_caps.size, output_caps.size, POSE_DIM * POSE_DIM)

        activations:  (caps_num)
        routing_prob: (input_caps.size, output_caps.size)
        means:        (caps_num, POSE_DIM * POSE_DIM)
        variances:    (caps_num, POSE_DIM * POSE_DIM)
        """

        # Update routing probabilities according to the prev caps activations
        # (notice here -1 dim is the out dim)
        self.routing_prob = self.routing_prob * activations_in.unsqueeze(dim=-1)
        self.routing_prob = self.routing_prob.detach()
        # Add a pose dimension (-1 dim is h, the pose dim)
        routing_prob = self.routing_prob.unsqueeze(dim=-1)

        # Calculate the means of output caps pose (-3 dim is i, the in dim)
        pose_means = torch.sum(routing_prob * votes / routing_prob, dim=-3)

        # Calculate the variances of output caps pose (-3 dim is i, the in dim)
        votes_variances = torch.pow(votes - pose_means.unsqueeze(dim=-3), 2)
        pose_variances = torch.sum(routing_prob * votes_variances / routing_prob, dim=-3)

        # Calculate cost (notice here -2 dim is i, not j, and dim -1 is for h)
        routing_prob_sum = self.routing_prob.sum(dim=-2).unsqueeze(dim=-1)
        cost = (self.beta_v - 0.5 * torch.log(pose_variances)) * routing_prob_sum

        # Get the activations of the capsules in the next layer (-1 dim is h)
        activations_out = torch.sigmoid(self.inv_temp * (self.beta_a - cost.sum(dim=-1)))

        return pose_means, pose_variances, activations_out


    def E_step(self, votes, output_params):
        """
        votes:        (input_caps.size, output_caps.size, POSE_DIM * POSE_DIM)

        activations:  (caps_num)
        routing_prob: (input_caps.size, output_caps.size)
        means:        (caps_num, POSE_DIM * POSE_DIM)
        variances:    (caps_num, POSE_DIM * POSE_DIM)
        """
        
        pose_means, pose_variances, activations_out = output_params

        # Calculate probabilities of votes agreement with output caps
        # (dim -3 is unsqueezed to get i, then we sum over h = dim -1)

        # XXX something is blowing up the gradients here
        pose_means = pose_means.unsqueeze(dim=-3)
        pose_variances = pose_variances.unsqueeze(dim=-3)
        votes_variances = torch.pow(votes - pose_means, 2)
        exponent = -torch.sum(votes_variances / (EPSILON + 2 * pose_variances), dim=-1)
        coeff_inv = torch.sqrt(EPSILON + torch.prod(2 * PI * pose_variances, dim=-1))
        agreement_prob = torch.exp(exponent) / coeff_inv

        # Update routing probabilities
        activations_out = activations_out.unsqueeze(dim=-2)  # Add 'i' dim
        self.routing_prob = (EPSILON + activations_out * agreement_prob) / \
            torch.sum(EPSILON + activations_out * agreement_prob, dim=-2, keepdim=True)
        self.routing_prob = self.routing_prob.detach()













        