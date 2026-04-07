import numpy as np
import pcgym


def get_environment(
    problem: str = None,
    T: int = None,
    n_steps: int = None,
    initial_state: np.ndarray = None,
    *args,
    **kwargs,
):
    assert not (problem is None), "No problem defined"
    assert not (T is None), "No time horizon defined"
    assert not (n_steps is None), "No number of steps defined"
    assert not (initial_state is None), "No initial_state defined"
    if problem == "cstr":
        parameters = create_cstr_environment(n_steps, T, initial_state)
    else:
        parameters = dict()
    return pcgym.make_env(parameters), parameters["o_space"]


def create_cstr_environment(n_steps, T, initial_state):
    # Setpoint
    SP = {
        "Ca": [0.85 for i in range(int(n_steps / 2))]
        + [0.9 for i in range(int(n_steps / 2))]
    }

    # Action and observation Space
    action_space = {"low": np.array([295]), "high": np.array([302])}
    observation_space = {
        "low": np.array([0.7, 300, 0.8]),
        "high": np.array([1, 350, 0.9]),
    }

    # Construct the environment parameter dictionary
    return {
        "N": n_steps,  # Number of time steps
        "tsim": T,  # Simulation Time
        "SP": SP,
        "o_space": observation_space,
        "a_space": action_space,
        "x0": initial_state,  # Initial conditions [Ca, T, Ca_SP]
        "model": "cstr",  # Select the model
    }
