import os
import torch
from grl.utils.log import log


def save_model(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    prefix="checkpoint",
):
    """
    Overview:
        Save model state_dict, optimizer state_dict and training iteration to disk, name as 'prefix_iteration.pt'.
    Arguments:
        path (:obj:`str`): path to save model
        model (:obj:`torch.nn.Module`): model to save
        optimizer (:obj:`torch.optim.Optimizer`): optimizer to save
        iteration (:obj:`int`): iteration to save
        prefix (:obj:`str`): prefix of the checkpoint file
    """

    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            iteration=iteration,
        ),
        f=os.path.join(path, f"{prefix}_{iteration}.pt"),
    )


def load_model(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    prefix="checkpoint",
) -> int:
    """
    Overview:
        Load model state_dict, optimizer state_dict and training iteration from disk, load the latest checkpoint file named as 'prefix_iteration.pt'.
    Arguments:
        path (:obj:`str`): path to load model
        model (:obj:`torch.nn.Module`): model to load
        optimizer (:obj:`torch.optim.Optimizer`): optimizer to load
        prefix (:obj:`str`): prefix of the checkpoint file
    Returns:
        - last_iteraion (:obj:`int`): the iteration of the loaded checkpoint
    """

    last_iteraion = -1
    checkpoint_path = path
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path) or not os.listdir(checkpoint_path):
            log.warning(f"Checkpoint path {checkpoint_path} does not exist or is empty")
            return last_iteraion

        checkpoint_files = sorted(
            [
                f
                for f in os.listdir(checkpoint_path)
                if f.endswith(".pt") and f.startswith(prefix)
            ],
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        if not checkpoint_files:
            log.warning(f"No checkpoint files found in {checkpoint_path}")
            return last_iteraion

        checkpoint_file = os.path.join(checkpoint_path, checkpoint_files[-1])

        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        last_iteraion = checkpoint.get("iteration", -1)
        ori_state_dict = {
            k.replace("module.", ""): v for k, v in checkpoint["model"].items()
        }
        ori_state_dict = {
            k.replace("_orig_mod.", ""): v for k, v in ori_state_dict.items()
        }
        model.load_state_dict(ori_state_dict)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        log.warning(f"{last_iteraion}_checkpoint files has been loaded")
        return last_iteraion
    return last_iteraion
