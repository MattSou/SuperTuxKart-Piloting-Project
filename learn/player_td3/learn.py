from pathlib import Path
from pystk2_gymnasium import AgentSpec
from functools import partial
import torch
import inspect
from bbrl.agents.gymnasium import ParallelGymAgent, make_env

# Note the use of relative imports
from .pystk_actor import env_name, get_wrappers, player_name
from .td3 import TD3, run_td3
from omegaconf import OmegaConf


params = {
    "save_best": False,
    "base_dir": "${gym_env.env_name}/ddpg-S${algorithm.seed}_$",
    "collect_stats": False,
    # Set to true to have an insight on the learned policy
    # (but slows down the evaluation a lot!)
    "plot_agents": True,
    "algorithm": {
        "seed": 1,
        "n_envs": 8,
        "n_steps": 256,
        "buffer_size": 1e6,
        "batch_size": 256,
        "max_grad_norm": 0.5,
        "nb_evals": 5,
        "eval_interval": 100_000,
        "learning_starts": 5_000,
        "max_epochs": 200,
        "discount_factor": 0.98,
        "entropy_mode": "auto",  # "auto" or "fixed"
        "init_entropy_coef": 2e-3,
        "tau_target": 0.05,
        "architecture": {
            "actor_hidden_size": [64, 64, 64],
            "critic_hidden_size": [256, 256, 256],
        },
    },
    "gym_env": {
        "env_name": env_name,
    },
    "actor_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 1e-5,
    },
    "critic_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 1e-5,
    },
    "entropy_coef_optimizer": {
        "classname": "torch.optim.Adam",
        "lr": 1e-5,
    },
    # "state" : None,
}
cfg = OmegaConf.create(params)

if __name__ == "__main__":
    # Setup the environment
    make_stkenv = partial(
        make_env,
        env_name,
        wrappers=get_wrappers(),
        render_mode=None,
        autoreset=True,
        agent=AgentSpec(use_ai=False, name=player_name),
    )

    # env_agent = ParallelGymAgent(make_stkenv, 1)
    # env = env_agent.envs[0]

    # (2) Learn

    # actor = Actor(env.observation_space, env.action_space)
    td3 = TD3(cfg, make_stkenv)
    run_td3(td3)
    # sac.visualize_best()

    # (3) Save the actor sate
    # mod_path = Path(inspect.getfile(get_wrappers)).parent
    # torch.save(actor.state_dict(), mod_path / "pystk_actor.pth")
