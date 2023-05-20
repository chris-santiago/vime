import torch


def set_device():
    device = {True: torch.device("mps"), False: torch.device("cpu")}
    return device[torch.backends.mps.is_available()]
