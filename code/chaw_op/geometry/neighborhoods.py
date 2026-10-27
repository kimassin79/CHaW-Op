from typing import Iterable, List, Sequence, Tuple





import torch


from torch import Tensor








def radius_edges(position: Tensor, batch: Tensor, radius: float, neighbors: int) -> Tensor:


    blocks: List[Tensor] = []


    for group in torch.unique(batch):


        indices = torch.nonzero(batch == group, as_tuple=False).flatten()


        points = position[indices]


        distances = torch.cdist(points, points)


        distances.fill_diagonal_(float("inf"))


        values, local = torch.topk(distances, k=min(neighbors, max(points.shape[0] - 1, 1)), largest=False)


        valid = values <= radius


        source = indices.unsqueeze(1).expand_as(local)[valid]


        target = indices[local[valid]]


        blocks.append(torch.stack([source, target]))


    if not blocks:


        return torch.empty((2, 0), dtype=torch.long, device=position.device)


    return torch.cat(blocks, dim=1)








def multiscale_edges(position: Tensor, batch: Tensor, radii: Sequence[float], neighbors: Sequence[int]) -> List[Tensor]:


    if len(radii) != len(neighbors):


        raise ValueError("radii and neighbors must have equal length")


    return [radius_edges(position, batch, radius, count) for radius, count in zip(radii, neighbors)]








def coalesce_edges(edge_index: Tensor, node_count: int) -> Tensor:


    linear = edge_index[0] * node_count + edge_index[1]


    unique = torch.unique(linear, sorted=True)


    return torch.stack([torch.div(unique, node_count, rounding_mode="floor"), unique.remainder(node_count)])








def symmetric_edges(edge_index: Tensor, node_count: int) -> Tensor:


    reverse = edge_index.flip(0)


    return coalesce_edges(torch.cat([edge_index, reverse], dim=1), node_count)








def edge_lengths(position: Tensor, edge_index: Tensor) -> Tensor:


    source, target = edge_index


    return torch.linalg.vector_norm(position[target] - position[source], dim=-1)








def degree(edge_index: Tensor, node_count: int, dtype: torch.dtype) -> Tensor:


    values = torch.ones(edge_index.shape[1], device=edge_index.device, dtype=dtype)


    result = torch.zeros(node_count, device=edge_index.device, dtype=dtype)


    result.index_add_(0, edge_index[1], values)


    return result








def normalized_edge_weights(edge_index: Tensor, node_count: int, dtype: torch.dtype) -> Tensor:


    deg = degree(edge_index, node_count, dtype).clamp_min(1.0)


    source, target = edge_index


    return deg[source].rsqrt() * deg[target].rsqrt()








def geodesic_shells(edge_index: Tensor, node_count: int, depth: int) -> List[Tensor]:


    adjacency = torch.zeros((node_count, node_count), dtype=torch.bool, device=edge_index.device)


    adjacency[edge_index[0], edge_index[1]] = True


    frontier = adjacency.clone()


    visited = torch.eye(node_count, dtype=torch.bool, device=edge_index.device)


    shells: List[Tensor] = []


    for _ in range(depth):


        frontier = frontier & ~visited


        shells.append(torch.nonzero(frontier, as_tuple=False).t().contiguous())


        visited = visited | frontier


        frontier = torch.matmul(frontier.float(), adjacency.float()) > 0


    return shells


