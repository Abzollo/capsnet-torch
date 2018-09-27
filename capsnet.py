import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from training_configs import *

# TODO: Add support for variable block sizes (including batch_size)

### Define model ###
class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        ...


class CapsLayer(nn.Module):
    def __init__(self, num_caps_in, num_caps_out):
        self.transformation = Variable(
            torch.zeros(caps_num_i, caps_num_j, POSE_DIM, POSE_DIM))
        self.router = CapsulesRouter(num_caps_in, num_caps_out)


    def forward(self, pose_in, activations_in):
        """
        Takes the state of a capsule layer, outputs the state of the next
        capsule layer, where state means: pose + activations.
        """

        # Apply transformation (-3 is j, the out dim)
        votes = pose_in.unsqueeze(dim=-3) @ self.transformation
        # Apply routing
        pose_out, activations_out = self.router(votes, activations_in)

        return pose_out, activations_out


class CapsulesRouter(nn.Module):
    def __init__(self, num_caps_in, num_caps_out):
        super(CapsuleRouter, self).__init__()
        # TODO: store num_caps?

        # Routing probability for capsule i to route to capsule j
        self.routing_prob = torch.zeros(num_caps_in, num_caps_out)
        self.routing_prob += 1 / num_caps_out  # TODO: do this for each forward()?

        # Increment me after each routing iteration (inverse temperature)
        self.inv_temp = torch.ones(1)

        # Learned parameters (you can call them "benefits", so to speak)
        self.beta_v = Variable(torch.zeros(num_caps_out, POSE_DIM * POSE_DIM))
        self.beta_a = Variable(torch.zeros(num_caps_out))


    def forward(self, votes, activations_in):
        """
        Get the votes (in matrix form) and activations and last capsule layer.
        Outputs the pose (in matrix form) and activations of the next capsule layer.
        """

        # Vectorize the votes_ij matrices (only the POSE_DIM x POSE_DIM part)
        votes = votes.view(self.num_caps_in, self.num_caps_out, POSE_DIM * POSE_DIM)
        # Get the result from EM-routing, skip the variances
        pose_means, _, activations_out = self.EM_routing(votes, activations_in)
        # Pack pose means back into the matrix form
        pose_means = pose_means.view(self.num_caps_out, POSE_DIM, POSE_DIM)

        return pose_means, activations_out


    def EM_routing(self, votes, activations_in):
        # Perform EM-routing for ROUTING_ITER iterations
        for _ in range(ROUTING_ITER):
            output_params = self.M_step(votes, activations_in)
            self.E_step(votes, output_params)  # Update r_ij
            # Decrease temperature
            self.inv_temp += 1

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
        self.routing_prob *= activations_in.unsqueeze(dim=-1)
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
        activations_out = F.sigmoid(self.inv_temp * (self.beta_a - cost.sum(dim=-1)))

        return (pose_means, pose_variances, activations_out)


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
        pose_means = pose_means.unsqueeze(dim=-3)
        pose_variances = pose_variances.unsqueeze(dim=-3)
        votes_variances = torch.pow(votes - pose_means, 2)
        exponent = -torch.sum(votes_variances / (2 * pose_variances), dim=-1)
        coeff_inv = torch.sqrt(torch.product(2 * PI * pose_variances, dim=-1))
        agreement_prob = torch.exp(exponent) / coeff_inv

        # Update routing probabilities
        activations_out = activations_out.unsqueeze(dim=-2)  # Add 'i' dim
        self.routing_prob = activations_out * agreement_prob / torch.sum(
            activations_out * agreement_prob, dim=-2, keepdim=True)













        