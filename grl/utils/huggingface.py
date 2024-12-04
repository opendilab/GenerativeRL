import os
import tempfile
import numpy as np
import torch
import logging
from pathlib import Path
from datetime import date
from easydict import EasyDict
import gym
import json

try:
    from huggingface_hub import ModelCard, ModelCardData
    from huggingface_hub import HfApi
    from huggingface_hub import hf_hub_download
except:
    print("Please install huggingface_hub by running `pip install huggingface_hub`")
import grl


def _calculate_model_params(model):
    Total_params = 0
    for param_tensor in model:
        mulValue = np.prod(model[param_tensor].size())
        Total_params += mulValue
    return Total_params


def _huggingface_api_upload_file(
    huggingface_api, path_or_fileobj, path_in_repo, repo_id, retry=5
):
    while retry > 0:
        try:
            file_url = huggingface_api.upload_file(
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
            )
            return file_url
        except:
            retry -= 1
            assert retry > 0, "Huggingface Hub upload retry exceeded limit."


def push_model_to_hub(
    model,
    config,
    env_name,
    task_name,
    algo_name,
    repo_id,
    score=None,
    video_path=None,
    wandb_url=None,
    usage_file=None,
    train_file=None,
    github_repo_url=None,
    github_doc_model_url=None,
    github_doc_env_url=None,
    model_description=None,
    installation_guide=None,
    platform_info=None,
    create_repo=True,
    template_path=None,
):
    """
    Overview:
        Push model to Huggingface Hub.
    Arguments:
        model (:obj:`torch.nn.Module`): the model to be pushed to Huggingface Hub.
        config (:obj:`dict`): the configuration of the model.
        env_name (:obj:`str`): the environment name of the model.
        task_name (:obj:`str`): the task name of the model.
        algo_name (:obj:`str`): the algorithm name of the model.
        repo_id (:obj:`str`): the repository id of Huggingface Hub where the model is stored.
        score (:obj:`float`, optional): the score of the model.
        video_path (:obj:`str`, optional): the path of the video file.
        wandb_url (:obj:`str`, optional): the wandb url of the model.
        usage_file (:obj:`str`, optional): the usage file of the model.
        train_file (:obj:`str`, optional): the train file of the model.
        github_repo_url (:obj:`str`, optional): the github repo url of the model.
        github_doc_model_url (:obj:`str`, optional): the github doc model url of the model.
        github_doc_env_url (:obj:`str`, optional): the github doc env url of the model.
        model_description (:obj:`str`, optional): the model description of the model.
        installation_guide (:obj:`str`, optional): the installation guide of the model.
        platform_info (:obj:`str`, optional): the platform info of the model.
        create_repo (:obj:`bool`, optional): whether to create a new repo.
        template_path (:obj:`str`, optional): the template markdown file path of the model.
    """

    with tempfile.TemporaryDirectory() as workfolder:
        huggingface_api = HfApi()

        if template_path is None:
            template_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "modelcard_template.md"
            )

        torch.save(
            model.state_dict(), os.path.join(Path(workfolder), "pytorch_model.bin")
        )
        json.dump(
            config,
            open(os.path.join(Path(workfolder), "policy_config.json"), "w"),
            indent=4,
        )

        with open(os.path.join(Path(workfolder), "policy_config.json"), "r") as file:
            json_config = file.read()

        if usage_file is not None and os.path.exists(usage_file):
            with open(usage_file, "r") as file:
                usage = file.read()
        else:
            usage = ""

        if train_file is not None and os.path.exists(train_file):
            with open(train_file, "r") as file:
                python_code_for_train = file.read()
        else:
            python_code_for_train = ""

        model_size = (
            str(round(_calculate_model_params(model.state_dict()) / 256.0, 2)) + " KB"
        )

        if model_description is None:
            model_description = ""

        if installation_guide is None:
            installation_guide = "<TODO>"

        if wandb_url is None:
            wandb_url = "<TODO>"

        if github_repo_url is None:
            github_repo_url = "<TODO>"

        if github_doc_model_url is None:
            github_doc_model_url = "<TODO>"

        if github_doc_env_url is None:
            github_doc_env_url = "<TODO>"

        if create_repo:
            try:
                huggingface_api.create_repo(
                    repo_id=repo_id,
                    repo_type="model",
                    private=True,
                )
            except:
                logging.warning("repo already exists")

        model_file_url = _huggingface_api_upload_file(
            huggingface_api=huggingface_api,
            path_or_fileobj=os.path.join(Path(workfolder), "pytorch_model.bin"),
            path_in_repo="pytorch_model.bin",
            repo_id=repo_id,
        )

        if video_path is not None and os.path.exists(video_path):
            # copy to workfolder and rename the video file as deploy.mp4
            demo_file_url = _huggingface_api_upload_file(
                huggingface_api=huggingface_api,
                path_or_fileobj=video_path,
                path_in_repo="replay.mp4",
                repo_id=repo_id,
            )
        else:
            demo_file_url = None

        config_file_url = _huggingface_api_upload_file(
            huggingface_api=huggingface_api,
            path_or_fileobj=os.path.join(Path(workfolder), "policy_config.json"),
            path_in_repo="policy_config.json",
            repo_id=repo_id,
        )

        metric = [
            {
                "name": "mean_reward",
                "value": str(score),
                "type": "mean_reward",
                "verified": False,
            }
        ]

        card_data = ModelCardData(
            language="en",
            license="apache-2.0",
            library_name="pytorch",
            benchmark_name=env_name,
            task_name=task_name,
            tags=[
                "reinforcement-learning",
                "Generative Model",
                "GenerativeRL",
                task_name,
            ],
            **{
                "pipeline_tag": "reinforcement-learning",
                "model-index": [
                    {
                        "name": algo_name,
                        "results": [
                            {
                                "task": {
                                    "name": "reinforcement-learning",
                                    "type": "reinforcement-learning",
                                },
                                "dataset": {
                                    "name": task_name,
                                    "type": task_name,
                                },
                                "metrics": metric,
                            },
                        ],
                    },
                ],
            }
        )

        card = ModelCard.from_template(
            card_data,
            model_id="{}-{}-{}".format(env_name, task_name, algo_name),
            algo_name=algo_name,
            platform_info=platform_info,
            model_description=model_description,
            installation_guide=installation_guide,
            developers="OpenDILab",
            config_file_url=config_file_url,
            library_version=grl.__version__,
            gym_version=gym.__version__,
            pytorch_version=torch.__version__,
            date=date.today(),
            video_demo_url=demo_file_url,
            parameters_total_size=model_size,
            wandb_url=wandb_url,
            github_repo_url=github_repo_url,
            github_doc_model_url=github_doc_model_url,
            github_doc_env_url=github_doc_env_url,
            python_config=config,
            json_config=json_config,
            usage=usage,
            python_code_for_train=python_code_for_train,
            template_path=template_path,
        )

        try:
            card.validate()
            # card.save("README.md")
            card.push_to_hub(repo_id=repo_id)
        except:
            raise ValueError("model card info is invalid. please check.")


def pull_model_from_hub(repo_id: str):
    """
    Overview:
        Pull public available models from Huggingface Hub
    Arguments:
        repo_id (:obj:`str`): the repository id of Huggingface Hub where the model is stored.
    """
    with tempfile.TemporaryDirectory() as workfolder:

        model_file = hf_hub_download(
            repo_id=repo_id, filename="pytorch_model.bin", cache_dir=Path(workfolder)
        )
        policy_state_dict = torch.load(model_file, map_location=torch.device("cpu"))

        config_file = hf_hub_download(
            repo_id=repo_id, filename="policy_config.json", cache_dir=Path(workfolder)
        )
        config = EasyDict(json.load(open(config_file, "r")))

    return policy_state_dict, config
