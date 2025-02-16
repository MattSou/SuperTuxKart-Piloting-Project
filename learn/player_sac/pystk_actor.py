from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym

from gymnasium.wrappers import FilterObservation
# Imports our Actor class
# IMPORTANT: note the relative import
from .actors import Actor, MyWrapper, ArgmaxActor, SamplingActor, NaiveActor, CenterPathFilter

#: The base environment name
env_name = "supertuxkart/flattened_continuous_actions-v0"

#: Player name
player_name = "player_flattened_continous_v0"


def get_actor(
    state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
) -> Agent:
    actor = Actor(observation_space, action_space)
    # print('action_space', action_space)

    # Returns a dummy actor
    # if state is None:
    #     return SamplingActor(action_space)

    # if state is None:
    #     return PolicyBasedActor('model_discrete.pth', 'model_continuous.pth', observation_space)

    # if state is None:
    #     return NaiveActor(484)

    actor.load_state_dict(state)
    return Agents(actor, ArgmaxActor())


def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return [
        # Example of a custom wrapper
        # lambda env: FilterObservation(env, filter_keys=['continuous']),
        # lambda env: CenterPathFilter(env)
    ]
