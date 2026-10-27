from typing import Dict, Mapping, Tuple





import torch


from torch import Tensor, nn





from chaw_op.contracts import FieldBattery, MeshBatch


from chaw_op.physics.projection import WallEquilibrium, graph_divergence








class RunningVariance(nn.Module):


    def __init__(self, names: Tuple[str, ...], momentum: float = 0.01) -> None:


        super().__init__()


        self.names = names


        self.momentum = momentum


        self.register_buffer("variances", torch.ones(len(names)))


        self.register_buffer("initialized", torch.zeros(len(names), dtype=torch.bool))





    def forward(self, targets: Mapping[str, Tensor]) -> Dict[str, Tensor]:


        weights: Dict[str, Tensor] = {}


        for index, name in enumerate(self.names):


            variance = targets[name].detach().var().clamp_min(1e-8)


            if self.training:


                if not bool(self.initialized[index]):


                    self.variances[index] = variance


                    self.initialized[index] = True


                else:


                    self.variances[index].lerp_(variance, self.momentum)


            weights[name] = self.variances[index].reciprocal()


        return weights








class PinballLoss(nn.Module):


    def __init__(self, quantiles: Tuple[float, ...] = (0.05, 0.5, 0.95)) -> None:


        super().__init__()


        self.register_buffer("quantiles", torch.tensor(quantiles))





    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:


        error = target.unsqueeze(-1) - prediction


        quantiles = self.quantiles.to(error).view(*([1] * (error.ndim - 1)), -1)


        return torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean()








class FieldObjective(nn.Module):


    def __init__(self, quantile_weight: float = 0.05, wall_slack: float = 0.05) -> None:


        super().__init__()


        self.names = ("pressure", "velocity", "wss", "tawss", "osi", "ecap", "ilt", "pws", "rpi")


        self.variance = RunningVariance(self.names)


        self.pinball = PinballLoss()


        self.quantile_weight = quantile_weight


        self.wall_equilibrium = WallEquilibrium(wall_slack)





    def forward(


        self,


        prediction: FieldBattery,


        target: FieldBattery,


        mesh: MeshBatch,


        residual_weight: float,


    ) -> Tuple[Tensor, Dict[str, Tensor]]:


        predicted = prediction.tensors()


        expected = target.tensors()


        weights = self.variance(expected)


        components: Dict[str, Tensor] = {}


        regression = prediction.pressure.new_zeros(())


        for name in self.names:


            value = torch.mean((predicted[name] - expected[name]).square()) * weights[name]


            components[f"field/{name}"] = value


            regression = regression + value


        divergence = graph_divergence(prediction.velocity, mesh.position, mesh.edge_index).square().mean()


        wall_pws = self.wall_equilibrium(prediction.pws, mesh.position, mesh.edge_index, mesh.normal).square().mean()


        wall_rpi = self.wall_equilibrium(prediction.rpi, mesh.position, mesh.edge_index, mesh.normal).square().mean()


        physics = divergence + wall_pws + wall_rpi


        quantile = prediction.pressure.new_zeros(())


        if prediction.quantiles is not None:


            for name in self.names:


                values = prediction.quantiles[name]


                reference = expected[name]


                if reference.shape[-1] > 1:


                    reference = torch.linalg.vector_norm(reference, dim=-1, keepdim=True)


                quantile = quantile + self.pinball(values, reference)


        total = regression + residual_weight * physics + self.quantile_weight * quantile


        components["aggregate/regression"] = regression


        components["aggregate/physics"] = physics


        components["aggregate/quantile"] = quantile


        components["aggregate/total"] = total


        return total, components








class SymmetricConsistency(nn.Module):


    def forward(self, fields: FieldBattery) -> Tensor:


        low_shear = torch.exp(-fields.tawss)


        oscillatory = fields.osi / fields.tawss.clamp_min(1e-6)


        ilt_consistency = torch.nn.functional.mse_loss(fields.ilt, torch.sigmoid(low_shear + oscillatory))


        wall_scale = fields.pws / fields.pws.detach().mean().clamp_min(1e-6)


        rpi_scale = fields.rpi / fields.rpi.detach().mean().clamp_min(1e-6)


        wall_consistency = torch.nn.functional.mse_loss(wall_scale, rpi_scale)


        return ilt_consistency + wall_consistency








def residual_schedule(epoch: int, start: int = 50, end: int = 200, maximum: float = 0.1) -> float:


    if epoch <= start:


        return 0.0


    if epoch >= end:


        return maximum


    return maximum * float(epoch - start) / float(end - start)


