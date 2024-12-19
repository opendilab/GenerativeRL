# GenerativeRL Tutorials

English | [简体中文(Simplified Chinese)](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/README.zh.md)

## Train a generative model

### Diffusion model

We provide a simple colab notebook to demonstrate how to build a diffusion model using the `GenerativeRL` library. You can access the notebook [here](https://colab.research.google.com/drive/18yHUAmcMh_7xq2U6TBCtcLKX2y4YvNyk#scrollTo=aqtDAvG6cQ1V).

### Flow model

We provide a simple colab notebook to demonstrate how to build a flow model using the `GenerativeRL` library. You can access the notebook [here](https://colab.research.google.com/drive/1vrxREVXKsSbnsv9G2CnKPVvrbFZleElI?usp=drive_link).

## Evaluate a generative model

### Sample generation

We provide a simple colab notebook to demonstrate how to generate samples from a trained generative model using the `GenerativeRL` library. You can access the notebook [here](https://colab.research.google.com/drive/16jQhf1BDjtToxMZ4lDxB4IwGdRmr074j?usp=sharing).

### Density estimation

We provide a simple colab notebook to demonstrate how to estimate the density of samples using a trained generative model using the `GenerativeRL` library. You can access the notebook [here](https://colab.research.google.com/drive/1zHsW13n338YqX87AIWG26KLC4uKQL1ZP?usp=sharing).

## Tutorials via toy examples

We provide several toy examples to demonstrate the features of the `GenerativeRL` library. You can access the examples [here](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/).

### Diverse generative models

- [Diffusion Model](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/generative_models/swiss_roll_diffusion.py)
- [Energy condition Diffusion Model](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/generative_models/swiss_roll_energy_condition.py)
- [Independent Conditional Flow Matching Model](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/generative_models/swiss_roll_icfm.py)
- [Optimal Transport Conditional Flow Matching Model](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/generative_models/swiss_roll_otcfm.py)
- [SF2M](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/generative_models/swiss_roll_otcfm.py)

### Generative model applications

- [World Model](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/applications/swiss_roll_world_model.py)

### Generative model evaluation

- [Likelihood Evaluation](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/metrics/swiss_roll_likelihood.py)

### ODE/SDE solvers usages

- [DPM Solver](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/solvers/swiss_roll_dpmsolver.py)
- [SDE Solver](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/solvers/swiss_roll_sdesolver.py)

### Special usages in GenerativeRL

- [Customized Neural Network Modules](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/special_usages/customized_modules.py)
- [Dict-like Structure Data Generation](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/special_usages/dict_tensor_ode.py)

## Use Hugging Face website to push and pull models

### Push a model

We provide an [example](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/huggingface/lunarlander_continuous_qgpo_huggingface_push.py) to push a trained model to the Hugging Face website.

In this example, we push a trained LunarLanderContinuous model to the Hugging Face website, and automatically generate a model card using the [template](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/huggingface/modelcard_template.md) to showcase the model's [detailed information](https://huggingface.co/OpenDILabCommunity/LunarLanderContinuous-v2-QGPO).

### Pull a model

We provide an [example](https://github.com/opendilab/GenerativeRL/tree/main/grl_pipelines/tutorials/huggingface/lunarlander_continuous_qgpo_huggingface_pull.py) to pull a model from the Hugging Face website, and test the model's performance in the environment.
