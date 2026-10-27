import logging


from dataclasses import dataclass


from pathlib import Path


from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Tuple





import torch


from torch import Tensor, nn


from torch.optim import AdamW, Optimizer


from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR





from chaw_op.contracts import FieldBattery, MeshBatch


from chaw_op.objectives.field_loss import FieldObjective, residual_schedule


from chaw_op.training.distributed import DistributedContext, reduce_mean


from chaw_op.training.state import atomic_save








@dataclass(frozen=True)


class TrainSettings:


    epochs: int


    learning_rate: float


    weight_decay: float


    gradient_accumulation: int


    gradient_clip_norm: float


    warmup_epochs: int


    residual_ramp_start: int


    residual_ramp_end: int


    residual_max_weight: float


    snapshots: int


    seed: int








class Trainer:


    def __init__(


        self,


        model: nn.Module,


        objective: FieldObjective,


        settings: TrainSettings,


        context: DistributedContext,


        output: Path,


        configuration: Mapping[str, object],


    ) -> None:


        self.model = model


        self.objective = objective


        self.settings = settings


        self.context = context


        self.output = output


        self.configuration = configuration


        self.optimizer = AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)


        warmup = LinearLR(self.optimizer, start_factor=1e-3, total_iters=settings.warmup_epochs)


        cosine = CosineAnnealingLR(self.optimizer, T_max=max(settings.epochs - settings.warmup_epochs, 1), eta_min=1e-8)


        self.scheduler = SequentialLR(self.optimizer, [warmup, cosine], milestones=[settings.warmup_epochs])


        self.logger = logging.getLogger("chaw_op.training")


        self.global_step = 0


        self.best_validation_loss = float("inf")





    def fit(self, training: Iterable[Tuple[MeshBatch, FieldBattery, object]], validation: Iterable[Tuple[MeshBatch, FieldBattery, object]]) -> None:


        for epoch in range(self.settings.epochs):


            train_metrics = self.train_epoch(training, epoch)


            validation_metrics = self.validate(validation, epoch)


            validation_loss = validation_metrics["aggregate/total"]


            self.scheduler.step()


            if self.context.rank == 0:


                self.logger.info("epoch=%d train_loss=%.8f validation_loss=%.8f", epoch, train_metrics["aggregate/total"], validation_loss)


                if validation_loss < self.best_validation_loss:


                    self.best_validation_loss = validation_loss


                    self.save(self.output / "best.pt", epoch)


                if (epoch + 1) % self.settings.snapshots == 0:


                    self.save(self.output / f"epoch_{epoch + 1:04d}.pt", epoch)





    def train_epoch(self, loader: Iterable[Tuple[MeshBatch, FieldBattery, object]], epoch: int) -> Dict[str, float]:


        self.model.train()


        self.objective.train()


        totals: Dict[str, float] = {}


        batches = 0


        self.optimizer.zero_grad(set_to_none=True)


        residual_weight = residual_schedule(


            epoch,


            self.settings.residual_ramp_start,


            self.settings.residual_ramp_end,


            self.settings.residual_max_weight,


        )


        for index, (mesh, target, _) in enumerate(loader):


            mesh = mesh.to(self.context.device)


            target = move_fields(target, self.context.device)


            prediction = self.model(mesh)


            loss, components = self.objective(prediction, target, mesh, residual_weight)


            scaled = loss / self.settings.gradient_accumulation


            scaled.backward()


            if (index + 1) % self.settings.gradient_accumulation == 0:


                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.gradient_clip_norm)


                self.optimizer.step()


                self.optimizer.zero_grad(set_to_none=True)


                self.global_step += 1


            for name, value in components.items():


                totals[name] = totals.get(name, 0.0) + float(reduce_mean(value, self.context).detach().cpu())


            batches += 1


        return {name: value / max(batches, 1) for name, value in totals.items()}





    def validate(self, loader: Iterable[Tuple[MeshBatch, FieldBattery, object]], epoch: int) -> Dict[str, float]:


        self.model.eval()


        self.objective.eval()


        totals: Dict[str, float] = {}


        batches = 0


        residual_weight = residual_schedule(epoch, self.settings.residual_ramp_start, self.settings.residual_ramp_end, self.settings.residual_max_weight)


        with torch.no_grad():


            for mesh, target, _ in loader:


                mesh = mesh.to(self.context.device)


                target = move_fields(target, self.context.device)


                prediction = self.model(mesh)


                _, components = self.objective(prediction, target, mesh, residual_weight)


                for name, value in components.items():


                    totals[name] = totals.get(name, 0.0) + float(reduce_mean(value, self.context).cpu())


                batches += 1


        return {name: value / max(batches, 1) for name, value in totals.items()}





    def save(self, path: Path, epoch: int) -> None:


        atomic_save(


            path,


            self.model,


            self.optimizer,


            self.scheduler,


            epoch,


            self.global_step,


            self.settings.seed,


            self.best_validation_loss,


            self.configuration,


        )








def move_fields(fields: FieldBattery, device: torch.device) -> FieldBattery:


    values = {name: value.to(device) for name, value in fields.tensors().items()}


    quantiles = None


    if fields.quantiles is not None:


        quantiles = {name: value.to(device) for name, value in fields.quantiles.items()}


    return FieldBattery(**values, quantiles=quantiles)


