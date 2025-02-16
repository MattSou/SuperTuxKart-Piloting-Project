from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym

# from gymnasium.stk_wrappers import PolarObservations, ConstantSizedObservations, OnlyContinuousActionsWrapper, DiscreteActionsWrapper
from pystk2_gymnasium.stk_wrappers import PolarObservations, ConstantSizedObservations, OnlyContinuousActionsWrapper, DiscreteActionsWrapper
# Imports our Actor class
# IMPORTANT: note the relative import
from .actors import DiscretePolicy

#: The base environment name
env_name = "supertuxkart/full-v0"

#: Player name
player_name = "Gus_PPO"


def get_actor(
    state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
) -> Agent:
    actor = DiscretePolicy(16, [512, 512, 512], 7)
    

    actor.load_state_dict(state)
    return actor


def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return [
        # Example of a custom wrapper
        lambda env: PolarObservations(ConstantSizedObservations(env)),
        lambda env: OnlyContinuousActionsWrapper(env),
        lambda env: DiscreteActionsWrapper(env),
        #lambda env: MyWrapper(env),
    ]
