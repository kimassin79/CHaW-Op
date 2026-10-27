import os


from dataclasses import dataclass


from typing import Optional





import torch


import torch.distributed as dist


from torch import nn


from torch.nn.parallel import DistributedDataParallel








@dataclass(frozen=True)


class DistributedContext:


    rank: int


    local_rank: int


    world_size: int


    distributed: bool


    device: torch.device








def initialize(backend: str = "nccl") -> DistributedContext:


    world_size = int(os.environ.get("WORLD_SIZE", "1"))


    rank = int(os.environ.get("RANK", "0"))


    local_rank = int(os.environ.get("LOCAL_RANK", "0"))


    distributed = world_size > 1


    if torch.cuda.is_available():


        device = torch.device("cuda", local_rank)


        torch.cuda.set_device(device)


    else:


        device = torch.device("cpu")


        backend = "gloo"


    if distributed and not dist.is_initialized():


        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


    return DistributedContext(rank, local_rank, world_size, distributed, device)








def wrap(model: nn.Module, context: DistributedContext, find_unused_parameters: bool = False) -> nn.Module:


    if not context.distributed:


        return model


    return DistributedDataParallel(


        model,


        device_ids=[context.local_rank] if context.device.type == "cuda" else None,


        output_device=context.local_rank if context.device.type == "cuda" else None,


        broadcast_buffers=True,


        find_unused_parameters=find_unused_parameters,


    )








def reduce_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:


    if not context.distributed:


        return value


    result = value.detach().clone()


    dist.all_reduce(result, op=dist.ReduceOp.SUM)


    return result / context.world_size








def barrier(context: DistributedContext) -> None:


    if context.distributed:


        dist.barrier()








def finalize(context: DistributedContext) -> None:


    if context.distributed and dist.is_initialized():


        dist.destroy_process_group()


