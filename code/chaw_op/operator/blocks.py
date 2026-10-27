from typing import List, Sequence





import torch


from torch import Tensor, nn








class EdgeKernel(nn.Module):


    def __init__(self, width: int) -> None:


        super().__init__()


        self.message = nn.Sequential(nn.Linear(width * 3, width * 2), nn.SiLU(), nn.Linear(width * 2, width))


        self.gate = nn.Sequential(nn.Linear(width * 3, width), nn.Sigmoid())


        self.update = nn.Sequential(nn.Linear(width * 2, width * 2), nn.SiLU(), nn.Linear(width * 2, width))


        self.norm = nn.LayerNorm(width)





    def forward(self, nodes: Tensor, edges: Tensor, edge_features: Tensor) -> Tensor:


        source, target = edges


        joined = torch.cat([nodes[source], nodes[target], edge_features], dim=-1)


        message = self.message(joined) * self.gate(joined)


        aggregate = nodes.new_zeros(nodes.shape)


        aggregate.index_add_(0, target, message)


        counts = nodes.new_zeros((nodes.shape[0], 1))


        counts.index_add_(0, target, nodes.new_ones((target.shape[0], 1)))


        aggregate = aggregate / counts.clamp_min(1.0)


        return self.norm(nodes + self.update(torch.cat([nodes, aggregate], dim=-1)))








class MultiScaleGNO(nn.Module):


    def __init__(self, width: int, scales: int) -> None:


        super().__init__()


        self.kernels = nn.ModuleList([EdgeKernel(width) for _ in range(scales)])


        self.scale_logits = nn.Parameter(torch.zeros(scales))


        self.fusion = nn.Sequential(nn.Linear(width * scales, width * 2), nn.SiLU(), nn.Linear(width * 2, width))


        self.norm = nn.LayerNorm(width)





    def forward(self, nodes: Tensor, edge_sets: Sequence[Tensor], edge_features: Sequence[Tensor]) -> Tensor:


        outputs: List[Tensor] = []


        weights = torch.softmax(self.scale_logits, dim=0)


        for weight, kernel, edges, features in zip(weights, self.kernels, edge_sets, edge_features):


            outputs.append(kernel(nodes, edges, features) * weight)


        return self.norm(nodes + self.fusion(torch.cat(outputs, dim=-1)))








class FeedForward(nn.Module):


    def __init__(self, width: int, expansion: int = 4) -> None:


        super().__init__()


        self.norm = nn.LayerNorm(width)


        self.network = nn.Sequential(


            nn.Linear(width, width * expansion),


            nn.GELU(),


            nn.Linear(width * expansion, width),


        )





    def forward(self, values: Tensor) -> Tensor:


        return values + self.network(self.norm(values))








class LatentAttention(nn.Module):


    def __init__(self, width: int, heads: int) -> None:


        super().__init__()


        self.norm = nn.LayerNorm(width)


        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)


        self.feedforward = FeedForward(width)





    def forward(self, values: Tensor, batch: Tensor) -> Tensor:


        output = values.clone()


        for group in torch.unique(batch):


            indices = torch.nonzero(batch == group, as_tuple=False).flatten()


            tokens = self.norm(values[indices]).unsqueeze(0)


            attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)


            output[indices] = values[indices] + attended.squeeze(0)


        return self.feedforward(output)








class OperatorLayer(nn.Module):


    def __init__(self, width: int, heads: int, scales: int) -> None:


        super().__init__()


        self.local = MultiScaleGNO(width, scales)


        self.global_attention = LatentAttention(width, heads)


        self.mixer = nn.Sequential(nn.Linear(width * 2, width), nn.SiLU(), nn.Linear(width, width))


        self.norm = nn.LayerNorm(width)





    def forward(self, values: Tensor, batch: Tensor, edge_sets: Sequence[Tensor], edge_features: Sequence[Tensor]) -> Tensor:


        local = self.local(values, edge_sets, edge_features)


        global_values = self.global_attention(local, batch)


        return self.norm(values + self.mixer(torch.cat([local, global_values], dim=-1)))








class ResidualDecoder(nn.Module):


    def __init__(self, width: int, output_width: int, depth: int = 3) -> None:


        super().__init__()


        layers: List[nn.Module] = []


        for _ in range(depth):


            layers.extend([nn.Linear(width, width), nn.SiLU(), nn.LayerNorm(width)])


        self.body = nn.Sequential(*layers)


        self.output = nn.Linear(width, output_width)





    def forward(self, values: Tensor) -> Tensor:


        return self.output(values + self.body(values))


