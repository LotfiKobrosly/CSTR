import numpy as np
import pcgym


CURRENTLY_AVAILABLE_MODELS = [
    "cstr",
    "first_order_system",
    "multistage_extraction",
    "nonsmooth_control",
    "cstr_series_recycle",
    "distillation_column",
    "multistage_extraction_reactive",
    "four_tank",
    "heat_exchanger",
    "biofilm_reactor",
    "polymerisation_reactor",
    "crystallization",
]


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
            function = create_cstr_environment
        case "multistage_extraction":
            function = create_extraction_column_environment
        case "nonsmooth_control":
            function = create_nonsmooth_control_environment
        case "crystallization":
            function = create_potassium_sulfate_crystallization_environment
        case "four_tank":
            function = create_four_tank_environment
        case "biofilm_reactor":
            function = create_biofilm_reactor_environment
        # case "photo_production":
        #     parameters = create_photo_production_environment(
        #         n_steps, T, initial_state, set_points
        #     )
        case _:
            raise ValueError("Unknown environment: " + problem)
    parameters = function(n_steps, T, initial_state, set_points)
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
        "low": observation_bounds[:, 0],
        "high": observation_bounds[:, 1],
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
        "low": np.array([-1, -1, 0.2]),
        "high": np.array([1, 1, 0.4]),
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


def create_potassium_sulfate_crystallization_environment(
    n_steps, T, initial_state, set_points
):
    action_space = {"low": np.array([-1]), "high": np.array([1])}
    observation_bounds = np.array(
        [
            [0, 1e20],
            [0, 1e20],
            [0, 1e20],
            [0, 1e20],
            [0, 0.5],
            [0, 2],
            [0, 20],
            [0.9, 1.1],
            [14, 16],
        ]
    ).reshape(-1, 2)
    observation_space = {
        "low": observation_bounds[:, 0],
        "high": observation_bounds[:, 1],
    }
    return {
        "N": n_steps,
        "tsim": T,
        "SP": set_points,
        "o_space": observation_space,
        "a_space": action_space,
        "x0": initial_state,
        "model": "crystallization",
        "r_scale": {"CV": 0.5**2, "Ln": 1 / 20**2},
    }


def create_four_tank_environment(n_steps, T, initial_state, set_points):
    action_space = {"low": np.array([0, 0]), "high": np.array([10, 10])}
    observation_space = {
        "low": np.array([0] * 6),
        "high": np.array([0.6] * 6),
    }
    return {
        "N": n_steps,
        "tsim": T,
        "SP": set_points,
        "o_space": observation_space,
        "a_space": action_space,
        "x0": initial_state,
        "model": "four_tank",
        "r_scale": {"h3": 1 / 0.6**2, "h4": 1 / 0.6**2},
    }


def create_biofilm_reactor_environment(n_steps, T, initial_state, set_points):
    action_space = {
        "low": np.array([0, 1, 0.05, 0.05, 0.05]),
        "high": np.array([10, 30, 1, 1, 1]),
    }
    observation_bounds = np.array(
        [
            [0, 10],
            [0, 10],
            [0, 10],
            [0, 500],
            [0, 10],
            [0, 10],
            [0, 10],
            [0, 500],
            [0, 10],
            [0, 10],
            [0, 10],
            [0, 500],
            [0, 10],
            [0, 10],
            [0, 10],
            [0, 500],
            [0.9, 1.1],
        ]
    )
    observation_space = {
        "low": observation_bounds[:, 0],
        "high": observation_bounds[:, 1],
    }
    return {
        "N": n_steps,
        "tsim": T,
        "SP": set_points,
        "o_space": observation_space,
        "a_space": action_space,
        "x0": initial_state,
        "model": "biofilm_reactor",
        "r_scale": {"S1_A": 1 / 10**2},
    }


def code(control_instance):
    if isinstance(control_instance, np.ndarray):
        control_instance = control_instance.flatten()
    control_instance_tuple = ()
    for element in control_instance:
        control_instance_tuple += ((round(element, 3)),)
    return control_instance_tuple
