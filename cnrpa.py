import itertools
from copy import deepcopy
import numpy as np

from environment import EnvironmentWrapper
from continuous_dictionary import *
from gaussian_kernel import GaussianKernel
from constants import RANDOM_STATE, RELEVANCE_THRESHOLD, HALF_LIFE_DIVIDER


def get_region_area(region: tuple):
    """
    region is a tuple of 2 tuples
    First tuple contains the lower bounds of the region
    Second tuple contains the upper bounds of the region
    """
    area = 1
    for component_index in range(len(region[0])):
        area *= region[1][component_index] - region[0][component_index]

    return area


def point_is_in_region(point: tuple, region: tuple):
    return np.all(
        (np.array(point) >= np.array(region[0]))
        & (np.array(point) <= np.array(region[1]))
    )


def subdivide_region(region: tuple):
    middle_bounds = (np.array(region[0]) + np.array(region[1])) / 2
    new_bounds = list()
    for component_index in range(len(middle_bounds)):
        new_bounds.append(
            [
                (region[0][component_index], middle_bounds[component_index]),
                (middle_bounds[component_index], region[1][component_index]),
            ]
        )

    return list(itertools.product(*tuple(new_bounds)))


def instantiate_policy(policy_type: str = "by_region", kernel_radius=None):
    if policy_type == "gaussian":
        assert not (
            kernel_radius is None
        ), "Value of kernel_radius must be specified when Gaussian policy is chosen"
        return ContinuousGaussianDictionary(kernel_radius)
    elif policy_type == "by_region":
        return ContinuousByRegionDictionary()
    else:
        raise ValueError("Policy type ill-defined")


def adapt_policy(
    policy,
    actions,
    sequence,
    learning_rate,
    n_visits: dict = None,
    full_area: float = None,
):
    if isinstance(policy, ContinuousByRegionDictionary):
        assert not (
            (n_visits is None) or (full_area is None)
        ), "'n_visits' and 'full_area' must be defined"

        # Update number of visits per regions
        for region in n_visits.keys():
            for point in sequence[:-1]:
                if point_is_in_region(point, region):
                    n_visits[region]["n_visits"] += 1
                    break

        # Subdivie regions if necessary
        temporary_n_visits = deepcopy(n_visits)
        for region in temporary_n_visits.keys():
            if (
                temporary_n_visits[region]["n_visits"]
                >= temporary_n_visits[region]["threshold"]
            ):
                new_regions = subdivide_region(region)
                for new_region in new_regions:
                    lower_bound = tuple([round(bound[0], 3) for bound in new_region])
                    upper_bound = tuple([round(bound[1], 3) for bound in new_region])
                    policy[(lower_bound, upper_bound)] = policy[region]
                    n_visits[(lower_bound, upper_bound)] = {
                        "n_visits": n_visits[region]["n_visits"],
                        "threshold": full_area
                        / get_region_area((lower_bound, upper_bound)),
                    }
                del policy[region]
                del n_visits[region]

        # Adapt policy
        ## Going through the trajectory, affecting them to their corresponding regions
        regional_division_of_points = {key: list() for key in policy.keys()}
        for point_index, point in enumerate(sequence[:-1]):
            for key in regional_division_of_points.keys():
                if point_is_in_region(point, key):
                    regional_division_of_points[key].append(actions[point_index])

        ## Changing the values of teh policy
        for region in regional_division_of_points.keys():
            whole_list = list(regional_division_of_points[region])
            if whole_list:
                policy[region] += learning_rate * (
                    np.mean(whole_list, axis=0) - policy[region]
                )

    elif isinstance(policy, ContinuousGaussianDictionary):
        if policy:
            # Get dimension of the keys
            dimension = len(sequence[0])

            # Get all values
            kernel_radius = 1 / len(policy)

            # Adapting visited states
            for state_index, state in enumerate(sequence[:-1]):
                new_move = actions[state_index]
                if state in policy.keys():
                    policy[state] += learning_rate * (new_move - policy[state])
                else:
                    policy[state] = new_move

            # Adapting nearby states
            for state in policy.keys():
                if state not in sequence[:-1]:
                    weights = np.zeros(len(actions))
                    gaussian_filter = GaussianKernel(state, kernel_radius)
                    weights = [
                        gaussian_filter.pdf(point)
                        for point in sequence[:-1]
                    ]
                    if np.sum(weights) >= RELEVANCE_THRESHOLD:
                        weights = np.array(weights) / np.sum(weights)
                        try:
                            new_move = np.array(list(actions)).T @ weights
                        except:
                            for action in actions:

                                print(action)
                            raise ValueError("Whatever")
                        policy[state] += learning_rate * (new_move - policy[state])
        else:
            for state_index, state in enumerate(sequence[:-1]):
                policy[state] = actions[state_index]

    else:
        raise ValueError("Policy type ill-defined")


def cnrpa(
    environment: EnvironmentWrapper,
    level: int = 1,
    n_policies: int = 100,
    policy: dict = None,
    n_visits: dict = None,
    current_iteration: int = 0,
    full_area: float = None,
):
    if level == 0:
        sampling_radius = np.exp(-current_iteration / (n_policies / HALF_LIFE_DIVIDER))
        environment.reset()
        while not environment.is_final():
        
            if len(policy) > 0:
                action = RANDOM_STATE.normal(
                    loc=policy[environment.code(environment.current_state)],
                    scale=sampling_radius,
                )
            else:
                action = environment.sample_random_action()
            environment.step(action)
        #print(environment.score)
        return environment.score, environment.sequence, environment.actions

    else:
        new_policy = deepcopy(policy)
        new_n_visits = deepcopy(n_visits)
        best_score = environment.best_score
        best_sequence = environment.best_sequence[:]
        best_actions = environment.best_actions[:]
        learning_rate = np.sqrt(1 / environment.horizon)
        for iteration in range(n_policies):
            new_environment = deepcopy(environment.environment)
            new_environment.reset()
            new_environment = EnvironmentWrapper(
                new_environment, environment.horizon, environment.penalty_factor
            )
            score, sequence, actions = cnrpa(
                new_environment,
                level - 1,
                n_policies,
                new_policy,
                new_n_visits,
                iteration,
                full_area,
            )
            if score < best_score:
                best_score = score
                best_sequence = sequence
                best_actions = actions
            adapt_policy(
                new_policy,
                best_actions,
                best_sequence,
                learning_rate,
                new_n_visits,
                full_area,
            )
        adapt_policy(
            policy, best_actions, best_sequence, learning_rate, n_visits, full_area
        )

        return best_score, best_sequence, best_actions


def run_cnrpa(
    environment: EnvironmentWrapper,
    level: int = 1,
    n_policies: int = 100,
    policy_type: str = "by_region",
    observation_space: dict = None,
):
    policy = instantiate_policy(policy_type, kernel_radius=0.2)
    if policy_type == "by_region":
        start_region = (
            (-2, -2, -2), (2, 2, 2)
        )
        policy[start_region] = RANDOM_STATE.uniform(-1, 1)
        n_visits = {
            start_region : {"n_visits": 0, "threshold": 1}
        }
        full_area = get_region_area(start_region)
    else:
        n_visits = None
        full_area = None
    best_score, best_sequence, best_actions = cnrpa(
        environment, level, n_policies, policy, n_visits, 0, full_area
    )
    return best_sequence, best_actions, best_score
