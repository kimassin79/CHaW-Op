import os


import random


from pathlib import Path


from typing import Any, Dict, Mapping, MutableMapping





import numpy as np


import torch


from torch import nn


from torch.optim import Optimizer








def set_seed(seed: int) -> None:


    random.seed(seed)


    np.random.seed(seed)


    torch.manual_seed(seed)


    torch.cuda.manual_seed_all(seed)


    torch.backends.cudnn.benchmark = False


    torch.backends.cudnn.deterministic = True








def capture_random_state() -> Dict[str, Any]:


    state: Dict[str, Any] = {


        "python": random.getstate(),


        "numpy": np.random.get_state(),


        "torch": torch.get_rng_state(),


    }


    if torch.cuda.is_available():


        state["cuda"] = torch.cuda.get_rng_state_all()


    return state








def restore_random_state(state: Mapping[str, Any]) -> None:


    random.setstate(state["python"])


    np.random.set_state(state["numpy"])


    torch.set_rng_state(state["torch"])


    if torch.cuda.is_available() and "cuda" in state:


        torch.cuda.set_rng_state_all(state["cuda"])








def atomic_save(


    path: Path,


    model: nn.Module,


    optimizer: Optimizer,


    scheduler: Any,


    epoch: int,


    global_step: int,


    seed: int,


    best_validation_loss: float,


    configuration: Mapping[str, Any],


) -> None:


    path.parent.mkdir(parents=True, exist_ok=True)


    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")


    payload = {


        "model": model.state_dict(),


        "optimizer": optimizer.state_dict(),


        "scheduler": scheduler.state_dict(),


        "epoch": epoch,


        "global_step": global_step,


        "seed": seed,


        "best_validation_loss": best_validation_loss,


        "configuration": dict(configuration),


        "random_state": capture_random_state(),


    }


    torch.save(payload, temporary)


    os.replace(temporary, path)








def resume(path: Path, model: nn.Module, optimizer: Optimizer, scheduler: Any, device: torch.device) -> Dict[str, Any]:


    payload = torch.load(path, map_location=device)


    model.load_state_dict(payload["model"])


    optimizer.load_state_dict(payload["optimizer"])


    scheduler.load_state_dict(payload["scheduler"])


    restore_random_state(payload["random_state"])


    return payload


