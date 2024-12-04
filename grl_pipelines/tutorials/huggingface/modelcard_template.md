---
{{ card_data }}
---

# Play **{{ task_name | default("[More Information Needed]", true)}}** with **{{ algo_name | default("[More Information Needed]", true)}}** Policy

## Model Description
<!-- Provide a longer summary of what this model is. -->

This implementation applies **{{ algo_name | default("[More Information Needed]", true)}}** to the {{ benchmark_name | default("[More Information Needed]", true)}} **{{ task_name | default("[More Information Needed]", true)}}** environment using {{ platform_info | default("[GenerativeRL](https://github.com/opendilab/di-engine)", true)}}.

{{ model_description | default("**GenerativeRL** is a Python library for various generative model based reinforcement learning algorithms and benchmarks. Built on PyTorch, it supports both academic research and prototype applications, offering customization of training pipelines.", false)}}

## Model Usage
### Install the Dependencies
<details close>
<summary>(Click for Details)</summary>

```shell
# install GenerativeRL with huggingface support
pip3 install GenerativeRL[huggingface]
# install environment dependencies if needed
{{ installation_guide | default("", false)}}
```
</details>

### Download Model from Huggingface and Run the Model

<details close>
<summary>(Click for Details)</summary>

```shell
# running with trained model
python3 -u run.py
```
**run.py**
```python
{{ usage | default("# [More Information Needed]", true)}}
```
</details>

## Model Training

### Train the Model and Push to Huggingface_hub

<details close>
<summary>(Click for Details)</summary>

```shell
#Training Your Own Agent
python3 -u train.py
```
**train.py**
```python
{{ python_code_for_train | default("# [More Information Needed]", true)}}
```
</details>

**Configuration**
<details close>
<summary>(Click for Details)</summary>


```python
{{ python_config | default("# [More Information Needed]", true)}}
```

```json
{{ json_config | default("# [More Information Needed]", true)}}
```

</details>

**Training Procedure** 
<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->
- **Weights & Biases (wandb):** [monitor link]({{ wandb_url | default("[More Information Needed]", true)}})

## Model Information
<!-- Provide the basic links for the model. -->
- **Github Repository:** [repo link]({{ github_repo_url | default("[More Information Needed]", true)}})
- **Doc**: [Algorithm link]({{ github_doc_model_url | default("[More Information Needed]", true)}})
- **Configuration:** [config link]({{ config_file_url | default("[More Information Needed]", true)}})
- **Demo:** [video]({{ video_demo_url | default("[More Information Needed]", true)}})
<!-- Provide the size information for the model. -->
- **Parameters total size:** {{ parameters_total_size | default("[More Information Needed]", true)}}
- **Last Update Date:** {{ date | default("[More Information Needed]", true)}}

## Environments
<!-- Address questions around what environment the model is intended to be trained and deployed at, including the necessary information needed to be provided for future users. -->
- **Benchmark:** {{ benchmark_name | default("[More Information Needed]", true)}}
- **Task:** {{ task_name | default("[More Information Needed]", true)}}
- **Gym version:** {{ gym_version | default("[More Information Needed]", true)}}
- **GenerativeRL version:** {{ library_version | default("[More Information Needed]", true)}}
- **PyTorch version:** {{ pytorch_version | default("[More Information Needed]", true)}}
- **Doc**: [Environments link]({{ github_doc_env_url | default("[More Information Needed]", true)}})

