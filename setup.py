from setuptools import setup, find_packages

setup(
    name='GenerativeRL',
    version='0.0.1',
    description='PyTorch implementations of generative reinforcement learning algorithms',
    author='OpenDILab',

    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        'gym',
        'numpy<=1.26.4',
        'torch>=2.2.0',
        'opencv-python',
        'tensordict',
        'torchrl',
        'di-treetensor',
        'matplotlib',
        'wandb',
        'rich',
        'easydict',
        'tqdm',
        'torchdyn',
        'torchsde',
        'scipy',
        'POT',
        'beartype',
        'diffusers',
        'av',
        'moviepy',
        'imageio[ffmpeg]',
    ],
    dependency_links=[
        'git+https://github.com/rtqichen/torchdiffeq.git#egg=torchdiffeq',
    ],
    extras_require={
        'd4rl': [
            'gym==0.23.1',
            'mujoco_py',
            'Cython<3.0',
        ],
        'DI-engine': [
            'DI-engine',
        ],
        'formatter': [
            'black',
            'isort',
        ],
    }
)
