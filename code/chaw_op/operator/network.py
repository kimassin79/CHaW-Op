from typing import Dict, List, Sequence, Tuple





import torch


from torch import Tensor, nn





from chaw_op.contracts import FieldBattery, MeshBatch


from chaw_op.geometry.neighborhoods import multiscale_edges


from chaw_op.geometry.tokenization import E3Tokenizer, InvariantDescriptor


from chaw_op.operator.blocks import OperatorLayer, ResidualDecoder


from chaw_op.operator.coupling import CouplingFreeWallHead, ILTWallCoupling


from chaw_op.physics.projection import HemodynamicTraces, LerayProjection








class CHaWOperator(nn.Module):


    def __init__(


        self,


        width: int = 256,


        operator_layers: int = 6,


        attention_heads: int = 8,


        coupling_layers: int = 3,


        ga_channels: int = 32,


        radii: Sequence[float] = (0.015, 0.03, 0.06, 0.12),


        coupling_enabled: bool = True,


        conservation_projection: bool = True,


    ) -> None:


        super().__init__()


        self.radii = tuple(radii)


        self.neighbors = (16, 32, 48, 64)


        self.coupling_enabled = coupling_enabled


        self.conservation_projection = conservation_projection


        self.tokenizer = E3Tokenizer(width, ga_channels)


        self.edge_descriptor = InvariantDescriptor(width)


        self.layers = nn.ModuleList(


            [OperatorLayer(width, attention_heads, len(radii)) for _ in range(operator_layers)]


        )


        self.pressure_head = ResidualDecoder(width, 1)


        self.velocity_head = ResidualDecoder(width, 3)


        self.flow_correction_head = ResidualDecoder(width, 7)


        self.ilt_head = nn.Sequential(ResidualDecoder(width, 1), nn.Sigmoid())


        self.coupling = ILTWallCoupling(width, attention_heads, coupling_layers)


        self.coupling_free = CouplingFreeWallHead(width)


        self.quantile_head = ResidualDecoder(width, 27)


        self.projection = LerayProjection(24)


        self.traces = HemodynamicTraces()





    def forward(self, mesh: MeshBatch) -> FieldBattery:


        latent = self.tokenizer(mesh.position, mesh.normal, mesh.curvature, mesh.wall_distance, mesh.batch)


        edge_sets = multiscale_edges(mesh.position, mesh.batch, self.radii, self.neighbors)


        edge_features = [self.edge_descriptor(mesh.position, mesh.normal, edges) for edges in edge_sets]


        for layer in self.layers:


            latent = layer(latent, mesh.batch, edge_sets, edge_features)


        pressure = self.pressure_head(latent)


        velocity = self.velocity_head(latent)


        if self.conservation_projection:


            velocity, _ = self.projection(velocity, mesh.position, mesh.edge_index)


        wss, tawss, osi, ecap = self.traces(velocity, mesh.normal, mesh.position, mesh.edge_index)


        corrections = self.flow_correction_head(latent)


        wss = wss + corrections[:, :3]


        tawss = torch.nn.functional.softplus(tawss + corrections[:, 3:4])


        osi = torch.sigmoid(osi + corrections[:, 4:5]) * 0.5


        ecap = torch.nn.functional.softplus(ecap + corrections[:, 5:6])


        ilt = self.ilt_head(latent)


        if self.coupling_enabled:


            wall = self.coupling(latent, ilt, ecap, mesh.batch)


        else:


            wall = self.coupling_free(latent)


        quantile_values = self.quantile_head(latent).view(latent.shape[0], 9, 3)


        quantiles = self._quantile_map(quantile_values)


        return FieldBattery(


            pressure=pressure,


            velocity=velocity,


            wss=wss,


            tawss=tawss,


            osi=osi,


            ecap=ecap,


            ilt=ilt,


            pws=wall[:, :1],


            rpi=wall[:, 1:2],


            quantiles=quantiles,


        )





    def _quantile_map(self, values: Tensor) -> Dict[str, Tensor]:


        names = ("pressure", "velocity", "wss", "tawss", "osi", "ecap", "ilt", "pws", "rpi")


        ordered = torch.sort(values, dim=-1).values


        return {name: ordered[:, index] for index, name in enumerate(names)}


