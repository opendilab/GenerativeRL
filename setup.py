from setuptools import setup, find_packages

with open('README.md', 'r', encoding="utf-8") as f:
    readme = f.read()

setup(
    name='GenerativeRL',
    version='0.0.1',
    description='PyTorch implementations of generative reinforcement learning algorithms',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='OpenDILab',
    author_email="opendilab@pjlab.org.cn",
    url="https://github.com/opendilab/GenerativeRL",

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
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache-2.0",
)
