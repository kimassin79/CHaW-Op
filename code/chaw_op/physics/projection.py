from typing import Tuple





import torch


from torch import Tensor, nn








def graph_gradient(scalar: Tensor, position: Tensor, edge_index: Tensor) -> Tensor:


    source, target = edge_index


    displacement = position[target] - position[source]


    distance_squared = displacement.square().sum(-1, keepdim=True).clamp_min(1e-10)


    difference = scalar[target] - scalar[source]


    messages = difference * displacement / distance_squared


    result = scalar.new_zeros((scalar.shape[0], 3))


    result.index_add_(0, source, messages)


    counts = scalar.new_zeros((scalar.shape[0], 1))


    counts.index_add_(0, source, scalar.new_ones((source.shape[0], 1)))


    return result / counts.clamp_min(1.0)








def graph_divergence(vector: Tensor, position: Tensor, edge_index: Tensor) -> Tensor:


    source, target = edge_index


    displacement = position[target] - position[source]


    distance_squared = displacement.square().sum(-1, keepdim=True).clamp_min(1e-10)


    difference = vector[target] - vector[source]


    flux = (difference * displacement).sum(-1, keepdim=True) / distance_squared


    result = vector.new_zeros((vector.shape[0], 1))


    result.index_add_(0, source, flux)


    counts = vector.new_zeros((vector.shape[0], 1))


    counts.index_add_(0, source, vector.new_ones((source.shape[0], 1)))


    return result / counts.clamp_min(1.0)








def graph_laplacian(scalar: Tensor, position: Tensor, edge_index: Tensor) -> Tensor:


    return graph_divergence(graph_gradient(scalar, position, edge_index), position, edge_index)








class PoissonSolver(nn.Module):


    def __init__(self, iterations: int = 24, relaxation: float = 0.75) -> None:


        super().__init__()


        self.iterations = iterations


        self.relaxation = relaxation





    def forward(self, rhs: Tensor, position: Tensor, edge_index: Tensor) -> Tensor:


        source, target = edge_index


        displacement = position[target] - position[source]


        weights = displacement.square().sum(-1).clamp_min(1e-8).reciprocal()


        diagonal = rhs.new_zeros((rhs.shape[0], 1))


        diagonal.index_add_(0, source, weights.unsqueeze(-1))


        pressure = torch.zeros_like(rhs)


        for _ in range(self.iterations):


            neighbor = rhs.new_zeros(rhs.shape)


            neighbor.index_add_(0, source, pressure[target] * weights.unsqueeze(-1))


            proposal = (neighbor - rhs) / diagonal.clamp_min(1e-8)


            pressure = pressure * (1.0 - self.relaxation) + proposal * self.relaxation


            pressure = pressure - pressure.mean()


        return pressure








class LerayProjection(nn.Module):


    def __init__(self, iterations: int = 24) -> None:


        super().__init__()


        self.poisson = PoissonSolver(iterations)





    def forward(self, velocity: Tensor, position: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:


        divergence = graph_divergence(velocity, position, edge_index)


        potential = self.poisson(divergence, position, edge_index)


        projected = velocity - graph_gradient(potential, position, edge_index)


        residual = graph_divergence(projected, position, edge_index)


        return projected, residual








class WallEquilibrium(nn.Module):


    def __init__(self, slack: float = 0.05) -> None:


        super().__init__()


        self.slack = slack





    def forward(self, stress: Tensor, position: Tensor, edge_index: Tensor, normal: Tensor) -> Tensor:


        gradient = graph_gradient(stress, position, edge_index)


        tangential = gradient - (gradient * normal).sum(-1, keepdim=True) * normal


        magnitude = torch.linalg.vector_norm(tangential, dim=-1, keepdim=True)


        return torch.relu(magnitude - self.slack)








class HemodynamicTraces(nn.Module):


    def __init__(self, viscosity: float = 0.0035) -> None:


        super().__init__()


        self.viscosity = viscosity





    def forward(self, velocity: Tensor, normal: Tensor, position: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:


        source, target = edge_index


        displacement = position[target] - position[source]


        distance = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True).clamp_min(1e-8)


        delta = velocity[target] - velocity[source]


        shear_edge = self.viscosity * delta / distance


        shear = velocity.new_zeros(velocity.shape)


        shear.index_add_(0, source, shear_edge)


        counts = velocity.new_zeros((velocity.shape[0], 1))


        counts.index_add_(0, source, velocity.new_ones((source.shape[0], 1)))


        shear = shear / counts.clamp_min(1.0)


        shear = shear - (shear * normal).sum(-1, keepdim=True) * normal


        tawss = torch.linalg.vector_norm(shear, dim=-1, keepdim=True)


        mean_vector = shear


        osi = 0.5 * (1.0 - torch.linalg.vector_norm(mean_vector, dim=-1, keepdim=True) / tawss.clamp_min(1e-8))


        ecap = osi / tawss.clamp_min(1e-6)


        return shear, tawss, osi.clamp(0.0, 0.5), ecap


