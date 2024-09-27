How to train generative models
-------------------------------------------------

GenerativeRL provides a set of generative models that can be used to generate data samples for diverse applications.
These models include diffusion models, flow models, and bridge models.

Stochastic path
~~~~~~~~~~~~~~~~

Diffusion models take the form of a stochastic differential equation (SDE) that describes the diffusion process evolution of a data distribution over time.
A typical diffusion process is defined by the following SDE:

.. math::

    dX_t = f(X_t, t) dt + \sigma(X_t, t) dW_t

where :math:`X_t` is the data distribution at time :math:`t`, :math:`f` is the drift function, :math:`\sigma` is the diffusion function, and :math:`dW_t` is the Wiener process increment.

The drift function :math:`f` and the diffusion function :math:`\sigma` are defined in class ``GaussianConditionalProbabilityPath`` in ``grl.numerical_methods.probability_path``
Every kind of diffusion path defines a different kind of diffusion model, such as the Variance-Preserving Stochastic Differential Equation (VP-SDE).

GenerativeRL currently supports the following diffusion models:

- Variance-Preserving Stochastic Differential Equation (VP-SDE)
- Generalized Variance-Preserving Stochastic Differential Equation (GVP)
- Linear Stochastic Differential Equation (Linear-SDE)

We usally take the diffusion time :math:`T` to be from 0 to 1, and the initial distribution :math:`X_0` to be the data distribution we want to model, while :math:`X_1` to be a standard normal distribution.

Model parameterization
~~~~~~~~~~~~~~~~~~~~~~~

The reverse-time diffusion process is defined as the generation process of the data distribution from a standard normal distribution to the target distribution.
By using Fokker-Planck-Kolmogorov (FPK) equation, we can derive the reverse-time diffusion process from the forward-time diffusion process.

.. math::

    dX_t = f(X_t, t) dt - \frac{1}{2}(g^2(t)+g'^2(t))\nabla_{x_t}\log p(x_t) dt + g'(t) d\hat{W}_t

where :math:`g(t) = \sigma(X_t, t)`, :math:`g'(t) = \sigma'(X_t, t)`, which is the noise coefficient used at reverse-time diffusion process, and :math:`d\hat{W}_t` is the Wiener process increment in the reverse-time diffusion process.

The score function :math:`\nabla_{x_t}\log p(x_t)` can be parameterized by a neural network, which is also denoted as `s_{\theta}` in the codebase.
In the work of DDPM, the score function is parameterized by a transformation into :math:`-\sigma(t)\nabla_{x_t}\log p(x_t)`, which has a comparable scale as standard gaussian noise and is called the noise function.

Or, we can also parameterize the velocity field :math:`v(x_t, t) = f(X_t, t) -\frac{1}{2}(g^2(t)+g'^2(t))\nabla_{x_t}\log p(x_t)` by neural networks.

GenerativeRL currently supports the following nerual network parameterizations method:

- Score function
- Noise function
- Velocity
- Data denoiser

User can independently define which parameterization method to use for neural networks in the generative model.

These functions has a same input-output shape, which is the same as the data distribution shape. Therefore, in GenerativeRL, we use a unified class ``grl.generative_models.IntrinsicModel`` to define the neural network parameterization of these functions.
This class takes time :math:`t`, data :math:`X_t` and condition :math:`c` as input, and returns the output of the neural network.
Any neural network architecture can be used in this class, such as MLP, CNN, U-net, or Transformer.
GenerativeRL provides a set of neural network architectures in the ``grl.neural_network`` module, which can be called from the configurations, for example:

.. code-block:: python

    diffusion_model=EasyDict(dict(
        device=device,
        x_size=x_size,
        alpha=1.0,
        solver=dict(
            type="ODESolver",
            args=dict(
                library="torchdyn",
            ),
        ),
        path=dict(
            type="linear_vp_sde",
            beta_0=0.1,
            beta_1=20.0,
        ),
        model=dict(
            type="noise_function",
            args=dict(
                t_encoder=t_encoder,
                backbone=dict(
                    type="TemporalSpatialResidualNet",
                    args=dict(
                        hidden_sizes=[512, 256, 128],
                        output_dim=x_size,
                        t_dim=t_embedding_dim,
                    ),
                ),
            ),
        ),
    ))

    from grl.generative_models import DiffusionModel
    diffusion_model = DiffusionModel(diffusion_model)

Here, neural network of class ``TemporalSpatialResidualNet`` is used to parameterize the noise function.

Customized neural network
~~~~~~~~~~~~~~~~~~~~~~~~~

For customized neural network architectures, GenerativeRL provide API ``grl.neural_network.register_module`` for registering new neural network classes.
User can define their own neural network class and register it in the ``grl.neural_network`` module, so that it can be called from the configurations, for example:

.. code-block:: python

    from grl.neural_network import register_module

    class MyModule(nn.Module):
        def __init__(self, hidden_sizes, output_dim):
            super(MyModule, self).__init__()
            self.layers = nn.ModuleList()
            for i in range(len(hidden_sizes) - 1):
                self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_sizes[-1], output_dim))

        def forward(self, t, x, c:None):
            for layer in self.layers:
                x = layer(x)
            return x

    register_module("MyModule", MyModule)

    diffusion_model=EasyDict(dict(
        device=device,
        x_size=x_size,
        alpha=1.0,
        solver=dict(
            type="ODESolver",
            args=dict(
                library="torchdyn",
            ),
        ),
        path=dict(
            type="linear_vp_sde",
            beta_0=0.1,
            beta_1=20.0,
        ),
        model=dict(
            type="noise_function",
            args=dict(
                t_encoder=t_encoder,
                backbone=dict(
                    type="MyModule",
                    args=dict(
                        hidden_sizes=[512, 256, 128],
                        output_dim=x_size,
                    ),
                ),
            ),
        ),
    ))

    from grl.generative_models import DiffusionModel
    diffusion_model = DiffusionModel(diffusion_model)

Usually, the generative model is trained by maximum likelihood estimation (MLE) or its variants, such as score matching, bridge matching, or flow matching.

Training objective for different generative models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GenerativeRL currently supports both score matching and flow matching for training diffusion models. Flow model can not obtain its score function directly, so it can only be trained by flow matching.

Score matching loss is defined as a weighted mean squared error between the score function and the neural network parameterization of the score function.

.. math::

    \mathcal{L}_{\text{DSM}} = \frac{1}{2}\int_{0}^{1}{\mathbb{E}_{p(x_t,x_0)}\left[\lambda(t)\|s_{\theta}(x_t)-\nabla_{x_t}\log p(x_t|x_0)\|^2\right]\mathrm{d}t}


Flow matching loss is defined as a mean squared error between the velocity field and the neural network parameterization of the velocity field.

.. math::

    \mathcal{L}_{\text{CFM}} = \frac{1}{2}\int_{0}^{1} \mathbb{E}_{p(x_t, x_0, x_1)}\left[\|v_{\theta}(x_t) - v(x_t|x_0, x_1)\|^2\right] \mathrm{d}t


GenerativeRL provides a unified API for training with score matching loss and flow matching loss, simply calling the ``score_matching_loss`` or ``flow_matching_loss`` method in the generative model class.
Here is the example code for training a diffusion model with score matching loss:

.. code-block:: python


    from grl.generative_models import DiffusionModel

    # Create a diffusion model
    diffusion_model = DiffusionModel(config)

    # Train the diffusion model with score matching loss
    score_matching_loss = diffusion_model.score_matching_loss(data)
    # Train the diffusion model with flow matching loss
    flow_matching_loss = diffusion_model.flow_matching_loss(data)

