from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym

from pystk2_gymnasium.stk_wrappers import PolarObservations, ConstantSizedObservations, OnlyContinuousActionsWrapper
from pystk2_gymnasium.wrappers import FlattenerWrapper
# Imports our Actor class
# IMPORTANT: note the relative import
from .actors import ContinuousDeterministicActor

#: The base environment name
env_name = "supertuxkart/flattened_continuous_actions-v0"

#: Player name
player_name = "player_flattened_continous_v0"


def get_actor(
    state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
) -> Agent:
    actor = ContinuousDeterministicActor()
    # print('action_space', action_space)

    # Returns a dummy actor
    # if state is None:
    #     return SamplingActor(action_space)

    # if state is None:
    #     return PolicyBasedActor('model_discrete.pth', 'model_continuous.pth', observation_space)

    # if state is None:
    #     return NaiveActor(484)

    actor.load_state_dict(state)
    return actor


def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return [
        # Example of a custom wrapper
        lambda env: PolarObservations(ConstantSizedObservations(env)),
        lambda env: OnlyContinuousActionsWrapper(env),
        lambda env: FlattenerWrapper(env, flatten_observations=False),
        #lambda env: MyWrapper(env),
    ]