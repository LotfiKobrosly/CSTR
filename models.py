import numpy as np
import pcgym


def get_environment(
    problem: str = None,
    T: int = None,
    n_steps: int = None,
    initial_state: np.ndarray = None,
    set_points: dict() = None,
    *args,
    **kwargs,
):
    assert not (problem is None), "No problem defined"
    assert not (T is None), "No time horizon defined"
    assert not (n_steps is None), "No number of steps defined"
    assert not (initial_state is None), "No initial_state defined"
    assert not (set_points is None), "No set points defined"
    match problem:
        case "cstr":
            parameters = create_cstr_environment(n_steps, T, initial_state, set_points)
        case "multistage_extraction":
            parameters = create_extraction_column_environment(
                n_steps, T, initial_state, set_points
            )
        case "nonsmooth_control":
            parameters = create_nonsmooth_control_environment(
                n_steps, T, initial_state, set_points
            )
        case "photo_production":
            parameters = create_photo_production_environment(
                n_steps, T, initial_state, set_points
            )
        case _:
            raise ValueError("Unknown environment")
    return pcgym.make_env(parameters), parameters["o_space"]


def create_cstr_environment(n_steps, T, initial_state, set_points):

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
        "SP": set_points,
        "o_space": observation_space,
        "a_space": action_space,
        "x0": initial_state,  # Initial conditions [Ca, T, Ca_SP]
        "model": "cstr",  # Select the model
    }


def create_extraction_column_environment(n_steps, T, initial_state, set_points):
    action_space = {"low": np.array([5, 10]), "high": np.array([500, 1000])}
    observation_bounds = np.array(
        [
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0.3, 0.4],
        ]
    ).reshape(-1, 2)
    observation_space = {
        "low": observation_bounds[0, :],
        "high": observation_bounds[1, :],
    }

    return {
        "N": n_steps,
        "tsim": T,
        "SP": set_points,
        "o_space": observation_space,
        "a_space": action_space,
        "x0": initial_state,
        "model": "multistage_extraction",
    }


# TODO
def create_nonsmooth_control_environment(n_steps, T, initial_state, set_points):
    action_space = {"low": np.array([-1]), "high": np.array([1])}
    observation_space = {
        "low": None,
        "high": None,
    }

    return {
        "N": n_steps,
        "tsim": T,
        "SP": set_points,
        "o_space": observation_space,
        "a_space": action_space,
        "x0": initial_state,
        "model": "nonsmooth_control",
    }


# TODO
def create_photo_production_environment(n_steps, T, initial_state, set_points):
    """
    See Bradford, Eric, et al. "Stochastic data-driven model predictive control
    using gaussian processes." Computers & Chemical Engineering 139 (2020): 106844.
    """
    action_space = {"low": None, "high": None}
    observation_space = {
        "low": None,
        "high": None,
    }

    return {
        "N": n_steps,
        "tsim": T,
        "SP": set_points,
        "o_space": observation_space,
        "a_space": action_space,
        "x0": initial_state,
        "model": "photo_production",
    }

def code(control_instance):
    control_instance_tuple = ()
    for element in control_instance:
        control_instance_tuple += ((round(element, 3)),)
    return control_instance_tuple
