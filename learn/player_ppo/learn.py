from pathlib import Path
from pystk2_gymnasium import AgentSpec
from functools import partial
import torch
import inspect
from bbrl.agents.gymnasium import ParallelGymAgent, make_env

# Note the use of relative imports
from .pystk_actor import env_name, get_wrappers, player_name
from .ppo import PPOClip, run
from omegaconf import OmegaConf


params = {
    "base_dir": "${gym_env.env_name}/ppo-clip-S${algorithm.seed}",
    "save_best": False,
    "logger": {
        "classname": "bbrl.utils.logger.TFLogger",
        "cache_size": 10000,
        "every_n_seconds": 10,
        "verbose": False,
    },
    "algorithm": {
        "seed": 12,
        "max_grad_norm": 0.3,
        "n_envs": 1,
        "n_steps": 800,
        "eval_interval": 10000,
        "loss_interval": 2000,
        "nb_evals": 10,
        "gae": 0.8,
        "discount_factor": 0.90,
        "normalize_advantage": False,
        "max_epochs": 80,
        "opt_epochs": 10,
        "batch_size": 256,
        "clip_range": 0.2,
        "clip_range_vf": 0,
        "entropy_coef": 2e-5,
        "policy_coef": 1,
        "critic_coef": 1.0,
        "policy_type": "DiscretePolicy",
        "architecture": {
            "actor_hidden_size": [512, 512, 512],
            "critic_hidden_size": [512, 512, 512],
        },
    },
    "gym_env": {
        "env_name": env_name,
    },
    "optimizer": {
        "classname": "torch.optim.AdamW",
        "lr": 1e-5,
        "eps": 1e-5,
    },
}    
    
cfg = OmegaConf.create(params)

if __name__ == "__main__":
    # Setup the environment
    make_stkenv = partial(
        make_env,
        cfg.gym_env.env_name,
        wrappers=get_wrappers(),
        render_mode=None,
        autoreset=True,
        agent=AgentSpec(use_ai=False, name="Gus"),
        num_kart=5,
        difficulty=1,
    )

    # env_agent = ParallelGymAgent(make_stkenv, 1)
    # env = env_agent.envs[0]

    # (2) Learn

    # actor = Actor(env.observation_space, env.action_space)
    ppo_clip =PPOClip(cfg, make_stkenv, 'pystk_actor.pth')
    run(ppo_clip)
    # sac.visualize_best()

    # (3) Save the actor sate
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    torch.save(ppo_clip.train_policy.state_dict(), mod_path / "pystk_actor.pth")
