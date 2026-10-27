from typing import List





import torch


from torch import Tensor, nn








class ConditioningEmbedding(nn.Module):


    def __init__(self, width: int) -> None:


        super().__init__()


        self.network = nn.Sequential(


            nn.Linear(width + 2, width),


            nn.SiLU(),


            nn.Linear(width, width),


            nn.LayerNorm(width),


        )





    def forward(self, latent: Tensor, ilt: Tensor, ecap: Tensor) -> Tensor:


        return self.network(torch.cat([ilt, ecap, latent], dim=-1))








class CouplingLayer(nn.Module):


    def __init__(self, width: int, heads: int) -> None:


        super().__init__()


        self.query_norm = nn.LayerNorm(width)


        self.condition_norm = nn.LayerNorm(width)


        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)


        self.transition = nn.Sequential(


            nn.LayerNorm(width),


            nn.Linear(width, width * 4),


            nn.GELU(),


            nn.Linear(width * 4, width),


        )





    def forward(self, latent: Tensor, condition: Tensor, batch: Tensor) -> Tensor:


        result = latent.clone()


        for group in torch.unique(batch):


            indices = torch.nonzero(batch == group, as_tuple=False).flatten()


            query = self.query_norm(latent[indices]).unsqueeze(0)


            key_value = self.condition_norm(condition[indices]).unsqueeze(0)


            attended, _ = self.attention(query, key_value, key_value, need_weights=False)


            result[indices] = latent[indices] + attended.squeeze(0)


        return result + self.transition(result)








class ILTWallCoupling(nn.Module):


    def __init__(self, width: int, heads: int, layers: int) -> None:


        super().__init__()


        self.embedding = ConditioningEmbedding(width)


        self.layers = nn.ModuleList([CouplingLayer(width, heads) for _ in range(layers)])


        self.wall_head = nn.Sequential(


            nn.LayerNorm(width),


            nn.Linear(width, width),


            nn.SiLU(),


            nn.Linear(width, 2),


            nn.Softplus(),


        )





    def forward(self, latent: Tensor, ilt: Tensor, ecap: Tensor, batch: Tensor) -> Tensor:


        condition = self.embedding(latent, ilt, ecap)


        wall = latent


        for layer in self.layers:


            wall = layer(wall, condition, batch)


            condition = condition + wall


        return self.wall_head(wall)








class CouplingFreeWallHead(nn.Module):


    def __init__(self, width: int) -> None:


        super().__init__()


        self.network = nn.Sequential(


            nn.LayerNorm(width),


            nn.Linear(width, width * 2),


            nn.SiLU(),


            nn.Linear(width * 2, width),


            nn.SiLU(),


            nn.Linear(width, 2),


            nn.Softplus(),


        )





    def forward(self, latent: Tensor) -> Tensor:


        return self.network(latent)


