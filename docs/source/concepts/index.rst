Concepts
=========================================================

Frameworks consist of code and APIs designed for data transformation, model training, and deployment.
GenerativeRL is a framework that provides user-friendly APIs for training and deploying generative models and reinforcement learning (RL) agents.
In this section, we will explore the core concepts of GenerativeRL, including generative models, reinforcement learning, and their integration.
We will discuss the key design principles that underpin the GenerativeRL library and how they can be leveraged to address complex problems in the field of reinforcement learning.
Additionally, we will explain why these concepts are important and what makes GenerativeRL unique and adaptable across a wide range of applications.

Concepts Overview
-----------------

Generative Models
~~~~~~~~~~~~~~~~~

Generative models are a class of machine learning models used to generate new data samples from a given distribution, typically learned from a training dataset.
Most generative models are trained using unsupervised learning techniques and can be applied to tasks such as image, video, or audio generation, data augmentation, and interpolation.
GenerativeRL focuses on models that use continuous-time dynamics to model data distributions, such as diffusion models and flow models.
These models have a high capacity to capture complex data distributions and have demonstrated promising results in a variety of applications.
They are typically trained using maximum likelihood estimation or its variants, such as score matching, and can generate high-quality samples by solving an ordinary differential equation (ODE) or a stochastic differential equation (SDE).

.. math::

    dX_t = f(X_t, t) dt + \sigma(X_t, t) dW_t

GenerativeRL provides unified APIs for training and deploying generative models based on continuous-time dynamics.
However, different generative models vary in their definitions of the drift function :math:`f` and the diffusion function :math:`\sigma`.
Some of these can be unified under common APIs, while others may require specific implementations.
There are four key differences between generative models implemented across different open-source libraries:

- **Model Definitions**: The neural network used to parameterize certain parts of the model, such as the drift function, score function, data denoiser, or potential function.
- **Path Definitions**: The definition of the stochastic process path, which determines whether the model is a diffusion model, a flow model, or a specific type of diffusion or flow model.
- **Training Procedure**: The fundamental training objective used to optimize the model parameters to maximize the likelihood of the training data. This can include pretraining methods like score matching, flow matching, or bridge matching, and fine-tuning techniques such as advantage-weighted regression, policy gradients, or adjoint matching.
- **Sampling Procedure**: The method used to generate new data samples from the model, which can involve forward or reverse sampling depending on the path and the numerical method used (e.g., Euler-Maruyama or Runge-Kutta).

GenerativeRL offers maximum flexibility, allowing users to customize and extend generative models to suit their specific needs across these four dimensions.
For instance, users can easily define their own neural network architectures, paths, training procedures, and sampling methods to create new generative models tailored to specific applications and data formats.

Reinforcement Learning
~~~~~~~~~~~~~~~~~~~~~~~

Reinforcement learning (RL) is a class of machine learning algorithms that learn to make decisions by interacting with an environment and receiving rewards or penalties based on their actions.
RL agents learn to maximize a cumulative reward signal by exploring the environment, taking actions, and updating their policies or value functions based on the observed rewards.
RL algorithms can be categorized into model-free and model-based methods, depending on whether they learn a model of the environment dynamics or directly optimize a policy or value function.
RL algorithms can also be categorized into online and offline methods, depending on whether they learn from interactions with the environment or from a fixed dataset.
Online RL algorithms can also be classified based on their exploration strategies, such as on-policy or off-policy methods, and their optimization objectives, such as policy gradients, value functions, or actor-critic methods.

Generative model integration with RL is a promising research direction that leverages generative models to improve sample efficiency, generalization, and exploration in RL.
For example, generative models can be used to learn a model of the environment dynamics, generate synthetic data for offline RL, or provide a learned Generative model policy or value function.

GenerativeRL provides a decoupled architecture that allows users to easily integrate generative models with RL algorithms.
Different generative models can be trained independently of the RL algorithms with unified APIs with little modifications to configurations.

Design Principles
-----------------

GenerativeRL is designed with the following principles, ranked from most important to least important:

- **Automatic Differentiation**: GenerativeRL leverages automatic differentiation libraries, such as PyTorch, `torchdiffeq`, and `torchdyn`, to efficiently and accurately compute gradients.
- **Unification**: GenerativeRL unifies the training and deployment of various generative models and reinforcement learning agents within a single framework.
- **Simplicity**: GenerativeRL provides a simple and intuitive interface for training and deploying generative models and RL agents.
- **Flexibility**: GenerativeRL is designed to be flexible and extensible, enabling users to easily customize and extend the library to suit their specific needs for different applications and data formats, such as tensors or dictionaries.
- **Modularity**: GenerativeRL is built on a modular architecture that allows users to mix and match different components, such as generative models, RL algorithms, and neural network architectures.
- **Reproducibility**: GenerativeRL ensures reproducible training and evaluation procedures through configurations, random seed initialization, logging, and checkpointing, making it possible to reproduce results across different runs and environments.
- **Minimal Dependencies**: GenerativeRL seeks to minimize external dependencies, providing a lightweight library that can be easily installed and used on various platforms and environments.
- **Compatibility with Existing RL Frameworks**: GenerativeRL is designed to work seamlessly with existing RL frameworks like OpenAI Gym and TorchRL, leveraging their functionality and environments.