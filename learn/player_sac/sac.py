import torch
import torch.nn as nn
from bbrl.agents import TemporalAgent, KWAgentWrapper
from bbrl_utils.algorithms import EpochBasedAlgo
from bbrl_utils.nn import setup_optimizer, soft_update_params

import copy
from bbrl.agents.gymnasium import ParallelGymAgent
from functools import partial
from .actors import ContinuousQAgent, SquashedGaussianActor
import gymnasium as gym
import numpy as np


def get_obs_and_actions_sizes(env):
        obs_space = env.get_observation_space()
        act_space = env.get_action_space()
        # print(type(obs_space))
        # print(type(act_space))
        def parse_space(space):
            if isinstance(space, gym.spaces.dict.Dict):
                n=0
                for k, v in space.spaces.items():
                    n+=parse_space(v)
                return n
            else:
                if len(space.shape) > 0:
                    if len(space.shape) > 1:
                        # warnings.warn(
                        #     "Multi dimensional space, be careful, a tuple (shape) "
                        #     "is returned, maybe youd like to flatten or simplify it first"
                        # )
                        return space.shape
                    return space.shape[0]
                else:
                    return space.n
        
        return parse_space(obs_space), parse_space(act_space)
# Create the SAC algorithm environment
class SACAlgo(EpochBasedAlgo):
    def __init__(self, cfg, make_env):
        super().__init__(cfg)
        self.train_env = ParallelGymAgent(
            partial(
                make_env,
                autoreset=True,
            ),
            cfg.algorithm.n_envs,
        ).seed(cfg.algorithm.seed)


        self.eval_env = ParallelGymAgent(make_env, cfg.algorithm.nb_evals).seed(
            cfg.algorithm.seed
        )

        self.make_env = make_env

        obs_size, act_size = get_obs_and_actions_sizes(self.train_env)

        assert (
            self.train_env.is_continuous_action()
        ), "SAC code dedicated to continuous actions"

        # We need an actor
        self.actor = SquashedGaussianActor(
            obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
        )

        # Builds the critics
        self.critic_1 = ContinuousQAgent(
            obs_size,
            cfg.algorithm.architecture.critic_hidden_size,
            act_size,
        ).with_prefix("critic-1/")
        self.target_critic_1 = copy.deepcopy(self.critic_1).with_prefix(
            "target-critic-1/"
        )

        self.critic_2 = ContinuousQAgent(
            obs_size,
            cfg.algorithm.architecture.critic_hidden_size,
            act_size,
        ).with_prefix("critic-2/")
        self.target_critic_2 = copy.deepcopy(self.critic_2).with_prefix(
            "target-critic-2/"
        )

        # Train and evaluation policies
        self.train_policy = self.actor
        self.eval_policy = KWAgentWrapper(self.actor, stochastic=False)

def setup_entropy_optimizers(cfg):
    if cfg.algorithm.entropy_mode == "auto":
        # Note: we optimize the log of the entropy coef which is slightly different from the paper
        # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
        # Comment and code taken from the SB3 version of SAC
        log_entropy_coef = nn.Parameter(
            torch.log(torch.ones(1) * cfg.algorithm.init_entropy_coef)
        )
        entropy_coef_optimizer = setup_optimizer(
            cfg.entropy_coef_optimizer, log_entropy_coef
        )
        return entropy_coef_optimizer, log_entropy_coef
    else:
        return None, None
    

def compute_critic_loss(
    cfg,
    reward: torch.Tensor,
    must_bootstrap: torch.Tensor,
    q_values: torch.Tensor, 
    target_q_values: torch.Tensor,
    ent_coef: torch.Tensor,
    action_logprobs: torch.Tensor,
):
    r"""Computes the critic loss for a set of $S$ transition samples

    Args:
        cfg: The experimental configuration
        reward: Tensor (2xS) of rewards
        must_bootstrap: Tensor (2xS) of indicators
        t_actor: The actor agent
        t_q_agents: The critics
        t_target_q_agents: The target of the critics
        rb_workspace: The transition workspace
        ent_coef: The entropy coefficient $\alpha$

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The two critic losses (scalars)
    """

    qvals = q_values[0]
    target = reward[1] + cfg.algorithm.discount_factor * (target_q_values[1] - ent_coef * action_logprobs[1]) * must_bootstrap[1]

    mse = nn.MSELoss()
    critic_loss = mse(target, qvals)
    return critic_loss


def compute_actor_loss(
    q_values: torch.Tensor,
    ent_coef: torch.Tensor,
    action_logprobs: torch.Tensor,
):
    r"""
    Actor loss computation
    :param ent_coef: The entropy coefficient $\alpha$
    :param t_actor: The actor agent (temporal agent)
    :param t_q_agents: The critics (as temporal agent)
    :param rb_workspace: The replay buffer (2 time steps, $t$ and $t+1$)
    """

    # Recompute the action with the current actor (at $a_t$)
    actor_loss = -q_values[0] + ent_coef * action_logprobs
   
    return actor_loss.mean()


def run_sac(sac: SACAlgo):
    cfg = sac.cfg
    logger = sac.logger


    # If entropy_mode is not auto, the entropy coefficient ent_coef remains
    # fixed. Otherwise, computes the target entropy
    if cfg.algorithm.entropy_mode == "auto":
        # target_entropy is \mathcal{H}_0 in the SAC and aplications paper.
        target_entropy = -np.prod(sac.train_env.action_space.shape).astype(np.float32)
    else:
        target_entropy = None

    actor_optimizer = setup_optimizer(cfg.actor_optimizer, sac.actor)
    critic_optimizer_1 = setup_optimizer(cfg.critic_optimizer, sac.critic_1)
    critic_optimizer_2 = setup_optimizer(cfg.critic_optimizer, sac.critic_2)
    entropy_coef_optimizer, log_entropy_coef = setup_entropy_optimizers(cfg)

    t_critic_1 = TemporalAgent(sac.critic_1)
    t_critic_2 = TemporalAgent(sac.critic_2)
    t_target_critic_1 = TemporalAgent(sac.target_critic_1)
    t_target_critic_2 = TemporalAgent(sac.target_critic_2)

    t_actor = TemporalAgent(sac.actor)
    
    # Loops over successive replay buffers
    for rb in sac.iter_replay_buffers():
        # Implement the SAC algorithm
        rb_workspace = rb.get_shuffled(sac.cfg.algorithm.batch_size)


        action_logprobs_rb = rb_workspace["action_logprobs"].detach()[1]
        # Compute the critic loss
        t_critic_1(rb_workspace, t=0, n_steps=2)
        t_critic_2(rb_workspace, t=0, n_steps=2)
        with torch.no_grad():
            t_target_critic_1(rb_workspace, t=0, n_steps=2)
            t_target_critic_2(rb_workspace, t=0, n_steps=2)

        q_values_1, q_values_2, terminated, reward, target_q_values_1, target_q_values_2 = rb_workspace[
            "critic-1/q_value", "critic-2/q_value", "env/terminated", "env/reward", "target-critic-1/q_value", "target-critic-2/q_value"
        ]

        must_bootstrap = ~terminated

        target_q_values = torch.min(target_q_values_1, target_q_values_2)

        ent_coef = log_entropy_coef.exp().detach()

        critic_loss_1 = compute_critic_loss(
            sac.cfg, reward, must_bootstrap, q_values_1, target_q_values, ent_coef, action_logprobs_rb
        )


        critic_loss_2 = compute_critic_loss(
            sac.cfg, reward, must_bootstrap, q_values_2, target_q_values, ent_coef, action_logprobs_rb
        )
        
        # Gradient step (critic_1)
        logger.add_log("critic_loss_1", critic_loss_1, sac.nb_steps)
        critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        torch.nn.utils.clip_grad_norm_(
            sac.critic_1.parameters(), sac.cfg.algorithm.max_grad_norm
        )
        critic_optimizer_1.step()

        # Gradient step (critic_2)
        logger.add_log("critic_loss_2", critic_loss_2, sac.nb_steps)
        critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        torch.nn.utils.clip_grad_norm_(
            sac.critic_2.parameters(), sac.cfg.algorithm.max_grad_norm
        )
        critic_optimizer_2.step()



        # Compute the actor loss

        
        action_logprobs_rb = rb_workspace["action_logprobs"].detach()[0]

        t_actor(rb_workspace, t=0, n_steps=1, stochastic=True)

        t_critic_1(rb_workspace, t=0, n_steps=2)
        t_critic_2(rb_workspace, t=0, n_steps=2)
        
        q_values_1, q_values_2 = rb_workspace["critic-1/q_value", "critic-2/q_value"]

        q_values = torch.min(q_values_1, q_values_2)

        actor_loss = compute_actor_loss(
            q_values, ent_coef, action_logprobs_rb
        )

        # Gradient step (actor)
        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            sac.actor.parameters(), sac.cfg.algorithm.max_grad_norm
        )
        actor_optimizer.step()
        logger.add_log("actor_loss", actor_loss, sac.nb_steps)

        # Entropy optimizer part
        if entropy_coef_optimizer is not None:
            # See Eq. (17) of the SAC and Applications paper. The log
            # probabilities *must* have been computed when computing the actor
            # loss.
            action_logprobs_rb = rb_workspace["action_logprobs"].detach()[0]
            
            entropy_coef_loss = -(
                log_entropy_coef.exp() * (action_logprobs_rb + target_entropy)
            ).mean()
            entropy_coef_optimizer.zero_grad()
            entropy_coef_loss.backward()
            entropy_coef_optimizer.step()
            logger.add_log("entropy_coef_loss", entropy_coef_loss, sac.nb_steps)
            logger.add_log("entropy_coef", log_entropy_coef.exp(), sac.nb_steps)
 
        ####################################################
        print(f"Actor loss: {actor_loss}, Critic loss 1: {critic_loss_1}, Critic loss 2: {critic_loss_2}, Entropy coef loss: {entropy_coef_loss}")
        # Soft update of target q function
        soft_update_params(sac.critic_1, sac.target_critic_1, cfg.algorithm.tau_target)
        soft_update_params(sac.critic_2, sac.target_critic_2, cfg.algorithm.tau_target)

        sac.evaluate()
        torch.save(sac.actor.state_dict(), "sac_actor_state.pth")
        torch.save(sac.actor, "sac_actor.pth")
    
    return sac.actor
