import itertools
from copy import deepcopy
import numpy as np

from environment import EnvironmentWrapper
from continuous_dictionary import *
from gaussian_kernel import GaussianKernel
from models import code
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


def instantiate_policy(policy_type: str = "by_region", kernel_radius: float = None, temporal_kernel_radius: float = None):
    if policy_type == "gaussian":
        assert not (
            kernel_radius is None
        ), "Value of kernel_radius must be specified when Gaussian policy is chosen"
        assert not (
            kernel_radius is None
        ), "Value of kernel_radius must be specified when Gaussian policy is chosen"
        return ContinuousGaussianDictionary(kernel_radius, temporal_kernel_radius)
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
        for timestamp, point in enumerate(sequence[:-1]):
            for region in n_visits.keys():
                if point_is_in_region((*point, timestamp), region):
                    n_visits[region]["n_visits"] += 1

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
        ## Conglomerating actions chosen in each region
        regional_subdivisions_new_moves = {region: list() for region in policy.keys()}
        for region in regional_subdivisions_new_moves.keys():
            for timestamp in range(int(region[0][-1]), min(int(region[1][-1]), len(actions) - 1)):
                if point_is_in_region((*sequence[timestamp], timestamp), region):
                    try:
                        regional_subdivisions_new_moves[region].append(actions[timestamp])
                    except IndexError:
                        print("Index:", timestamp)
                        print("Range:", int(region[0][-1]), int(region[1][-1]))
                        print("N° actions:", len(actions))
                        raise IndexError("Index out of range")
            regional_subdivisions_new_moves[region] = np.mean(regional_subdivisions_new_moves[region], axis=0)

        ## Changing values
        for timestamp, point in enumerate(sequence[:-1]):
            for key in policy.keys():
                if point_is_in_region((*point, timestamp), key):
                    #print("For region", key, "value:", policy[key])
                    policy[key] += learning_rate * (regional_subdivisions_new_moves[key].reshape(policy[key].shape) - policy[key])
                break

    elif isinstance(policy, ContinuousGaussianDictionary):
        if policy:
            # Get dimension of the keys
            dimension = len(sequence[0])

            # Get all values
            kernel_radius = 1 / len(policy)

            # Adapting visited states
            for timestamp, state in enumerate(sequence[:-1]):
                new_move = actions[timestamp]
                if (*state, timestamp) in policy.keys():
                    new_move = np.reshape(new_move, policy[(*state, timestamp)].shape)
                    policy[(*state, timestamp)] += learning_rate * (new_move - policy[(*state, timestamp)])
                else:
                    policy[(*state, timestamp)] = new_move

            # Adapting nearby states
            for state in policy.keys():
                if state[:-1] not in sequence[:-1]:
                    weights = np.zeros(len(actions))
                    gaussian_filter = GaussianKernel(state[:-1], kernel_radius)
                    temporal_gaussian_filter = GaussianKernel(np.array([state[-1]]), kernel_radius)
                    weights = [
                        gaussian_filter.pdf(point)
                        for point in sequence[:-1]
                    ]
                    if np.sum(weights) >= RELEVANCE_THRESHOLD:
                        weights = 0.5 * (np.array(weights) + np.array([temporal_gaussian_filter.pdf(timestamp) for timestamp in range(len(actions))]))
                        weights /= np.sum(weights)
                        try:
                            new_move = np.array(list(actions)).T @ weights
                        except:
                            raise ValueError("Can't compute new action")
                        new_move = np.reshape(new_move, policy[state].shape)
                        policy[state] += learning_rate * (new_move - policy[state])
        else:
            for timestamp, state in enumerate(sequence[:-1]):
                policy[(*state, timestamp)] = actions[timestamp]

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
    half_life_divider: int = None,
):
    if level == 0:
        sampling_radius = np.exp(-current_iteration / (n_policies / half_life_divider))
        environment.reset()
        if isinstance(policy, ContinuousByRegionDictionary):
            environment.allow_truncation()
        while not environment.is_final():
        
            if len(policy) > 0:
                action = RANDOM_STATE.normal(
                    loc=policy[(*code(environment.current_state), environment.current_timestamp)],
                    scale=sampling_radius,
                )
            else:
                action = environment.sample_random_action()
            #print(action)
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
                half_life_divider,
            )
            if score < best_score:
                best_score = score
                best_sequence = sequence[:]
                best_actions = actions[:]
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
    half_life_divider: int = 10,
):
    dimension = len(environment.current_state)
    kernel_radius = 0.3
    temporal_kernel_radius = 0.05 * environment.horizon
    policy = instantiate_policy(policy_type, kernel_radius=kernel_radius, temporal_kernel_radius=temporal_kernel_radius)
    if policy_type == "by_region":

        start_region = (
            tuple([-2 for _ in range(dimension)] + [0]),
            tuple([2 for _ in range(dimension)] + [environment.horizon + 1]),
        )
        n_visits = dict()
        policy[start_region] = environment.sample_random_action()
        n_visits[start_region] = {"n_visits": 0, "threshold": 1}
        full_area = get_region_area(start_region)
        environment.allow_truncation()
    else:
        n_visits = None
        full_area = None
    best_score, best_sequence, best_actions = cnrpa(
        environment, level, n_policies, policy, n_visits, 0, full_area, half_life_divider
    )
    environment.disallow_truncation()
    return best_sequence, best_actions, best_score
