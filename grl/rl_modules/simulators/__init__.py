from .gym_env_simulator import GymEnvSimulator


def get_simulator(type: str):
    if type.lower() not in SIMULATORS:
        raise KeyError(f"Invalid simulator type: {type}")
    return SIMULATORS[type.lower()]


def create_simulator(config):
    return get_simulator(config.type)(**config.args)


SIMULATORS = {
    "GymEnvSimulator".lower(): GymEnvSimulator,
}
