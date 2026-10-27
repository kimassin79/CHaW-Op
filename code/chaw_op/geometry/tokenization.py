from typing import List, Tuple





import torch


from torch import Tensor, nn








class ScalarVectorGate(nn.Module):


    def __init__(self, scalar_width: int, vector_width: int) -> None:


        super().__init__()


        self.scalar = nn.Linear(scalar_width, scalar_width)


        self.vector = nn.Linear(vector_width, vector_width, bias=False)


        self.gate = nn.Linear(scalar_width, vector_width)





    def forward(self, scalar: Tensor, vector: Tensor) -> Tuple[Tensor, Tensor]:


        scalar_out = torch.nn.functional.silu(self.scalar(scalar))


        vector_out = self.vector(vector.transpose(-1, -2)).transpose(-1, -2)


        vector_out = vector_out * torch.sigmoid(self.gate(scalar_out)).unsqueeze(1)


        return scalar_out, vector_out








class GeometricProduct(nn.Module):


    def __init__(self, channels: int) -> None:


        super().__init__()


        self.dot_mix = nn.Linear(channels, channels)


        self.cross_mix = nn.Linear(channels, channels, bias=False)





    def forward(self, left: Tensor, right: Tensor) -> Tuple[Tensor, Tensor]:


        dot = (left * right).sum(dim=1)


        cross = torch.cross(left.transpose(1, 2), right.transpose(1, 2), dim=-1)


        cross = cross.transpose(1, 2)


        return self.dot_mix(dot), self.cross_mix(cross.transpose(1, 2)).transpose(1, 2)








class E3Tokenizer(nn.Module):


    def __init__(self, width: int, ga_channels: int) -> None:


        super().__init__()


        self.width = width


        self.ga_channels = ga_channels


        self.scalar_lift = nn.Sequential(


            nn.Linear(8, width),


            nn.SiLU(),


            nn.Linear(width, width),


        )


        self.position_lift = nn.Linear(1, ga_channels, bias=False)


        self.normal_lift = nn.Linear(1, ga_channels, bias=False)


        self.product = GeometricProduct(ga_channels)


        self.gate = ScalarVectorGate(width + ga_channels, ga_channels)


        self.vector_readout = nn.Linear(ga_channels, width, bias=False)


        self.norm = nn.LayerNorm(width)





    def forward(


        self,


        position: Tensor,


        normal: Tensor,


        curvature: Tensor,


        wall_distance: Tensor,


        batch: Tensor,


    ) -> Tensor:


        center = self._batch_center(position, batch)


        centered = position - center[batch]


        radius = torch.linalg.vector_norm(centered, dim=-1, keepdim=True)


        normal_norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)


        alignment = (centered * normal).sum(dim=-1, keepdim=True)


        scalar_input = torch.cat(


            [curvature, wall_distance, radius, normal_norm, alignment, curvature.square(), wall_distance.square(), radius.square()],


            dim=-1,


        )


        scalar = self.scalar_lift(scalar_input)


        pos_mv = centered.unsqueeze(-1) * self.position_lift.weight[:, 0]


        normal_mv = normal.unsqueeze(-1) * self.normal_lift.weight[:, 0]


        dot, cross = self.product(pos_mv, normal_mv)


        scalar, vector = self.gate(torch.cat([scalar, dot], dim=-1), cross)


        vector_norm = torch.linalg.vector_norm(vector, dim=1)


        return self.norm(scalar[..., : self.width] + self.vector_readout(vector_norm))





    def _batch_center(self, position: Tensor, batch: Tensor) -> Tensor:


        groups = int(batch.max().item()) + 1


        centers = position.new_zeros((groups, 3))


        counts = position.new_zeros((groups, 1))


        centers.index_add_(0, batch, position)


        counts.index_add_(0, batch, position.new_ones((position.shape[0], 1)))


        return centers / counts.clamp_min(1.0)








class InvariantDescriptor(nn.Module):


    def __init__(self, width: int) -> None:


        super().__init__()


        self.network = nn.Sequential(


            nn.Linear(12, width),


            nn.SiLU(),


            nn.Linear(width, width),


            nn.LayerNorm(width),


        )





    def forward(self, position: Tensor, normal: Tensor, edge_index: Tensor) -> Tensor:


        source, target = edge_index


        displacement = position[target] - position[source]


        distance = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)


        unit = displacement / distance.clamp_min(1e-8)


        normal_dot = (normal[source] * normal[target]).sum(-1, keepdim=True)


        source_projection = (unit * normal[source]).sum(-1, keepdim=True)


        target_projection = (unit * normal[target]).sum(-1, keepdim=True)


        features = torch.cat(


            [distance, distance.square(), normal_dot, source_projection, target_projection, unit, normal[source], normal[target, :1]],


            dim=-1,


        )


        return self.network(features)








def edge_frames(position: Tensor, normal: Tensor, edge_index: Tensor) -> Tensor:


    source, target = edge_index


    tangent = position[target] - position[source]


    tangent = torch.nn.functional.normalize(tangent, dim=-1)


    first = torch.cross(normal[source], tangent, dim=-1)


    first = torch.nn.functional.normalize(first, dim=-1)


    second = torch.cross(tangent, first, dim=-1)


    return torch.stack([tangent, first, second], dim=-2)








def radial_basis(distance: Tensor, radii: List[float], width: int) -> Tensor:


    centers = torch.linspace(0.0, max(radii), width, device=distance.device, dtype=distance.dtype)


    scale = centers.new_tensor(max(radii) / max(width - 1, 1)).clamp_min(1e-6)


    return torch.exp(-((distance.unsqueeze(-1) - centers) / scale).square())


