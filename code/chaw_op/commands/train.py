import logging


from pathlib import Path


from typing import Any, Dict





import hydra


from omegaconf import DictConfig, OmegaConf


from torch.utils.data import DataLoader





from chaw_op.cohort.dataset import AAAMeshDataset, SiteBalancedSampler, collate_meshes


from chaw_op.objectives.field_loss import FieldObjective


from chaw_op.operator.network import CHaWOperator


from chaw_op.training.distributed import finalize, initialize, wrap


from chaw_op.training.engine import TrainSettings, Trainer


from chaw_op.training.state import set_seed








@hydra.main(version_base="1.2", config_path="../../../experiments", config_name="main")


def main(configuration: DictConfig) -> None:


    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


    context = initialize(str(configuration.distributed.backend))


    seed = int(configuration.seed) + context.rank


    set_seed(seed)


    root = Path(str(configuration.data.root))


    manifest = Path(str(configuration.data.manifest))


    training_data = AAAMeshDataset(root, manifest, "train")


    validation_data = AAAMeshDataset(root, manifest, "validation")


    sampler = SiteBalancedSampler(training_data.records, seed)


    training_loader = DataLoader(


        training_data,


        batch_size=int(configuration.optimization.batch_size_per_rank),


        sampler=sampler,


        num_workers=int(configuration.runtime.workers_per_rank),


        collate_fn=collate_meshes,


        pin_memory=True,


        prefetch_factor=int(configuration.runtime.prefetch_factor),


        persistent_workers=bool(configuration.runtime.persistent_workers),


    )


    validation_loader = DataLoader(


        validation_data,


        batch_size=int(configuration.optimization.batch_size_per_rank),


        shuffle=False,


        num_workers=int(configuration.runtime.workers_per_rank),


        collate_fn=collate_meshes,


        pin_memory=True,


        prefetch_factor=int(configuration.runtime.prefetch_factor),


        persistent_workers=bool(configuration.runtime.persistent_workers),


    )


    model = CHaWOperator(


        width=int(configuration.model.width),


        operator_layers=int(configuration.model.operator_layers),


        attention_heads=int(configuration.model.attention_heads),


        coupling_layers=int(configuration.model.coupling_layers),


        ga_channels=int(configuration.model.ga_channels),


        radii=tuple(float(value) for value in configuration.model.radii),


        coupling_enabled=bool(configuration.model.get("coupling_enabled", True)),


        conservation_projection=bool(configuration.model.get("conservation_projection", True)),


    ).to(context.device)


    model = wrap(model, context, bool(configuration.distributed.find_unused_parameters))


    objective = FieldObjective(float(configuration.optimization.quantile_weight)).to(context.device)


    settings = TrainSettings(


        epochs=int(configuration.optimization.epochs),


        learning_rate=float(configuration.optimization.learning_rate),


        weight_decay=float(configuration.optimization.weight_decay),


        gradient_accumulation=int(configuration.optimization.gradient_accumulation),


        gradient_clip_norm=float(configuration.optimization.gradient_clip_norm),


        warmup_epochs=int(configuration.optimization.warmup_epochs),


        residual_ramp_start=int(configuration.optimization.residual_ramp_start),


        residual_ramp_end=int(configuration.optimization.residual_ramp_end),


        residual_max_weight=float(configuration.optimization.residual_max_weight),


        snapshots=int(configuration.runtime.epochs_between_snapshots),


        seed=seed,


    )


    trainer = Trainer(model, objective, settings, context, Path("artifacts"), OmegaConf.to_container(configuration, resolve=True))


    trainer.fit(training_loader, validation_loader)


    finalize(context)








if __name__ == "__main__":


    main()


