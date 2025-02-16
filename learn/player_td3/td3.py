import torch
import torch.nn as nn
from bbrl.agents import TemporalAgent, Agents
import subprocess
from bbrl_utils.algorithms import EpochBasedAlgo
from bbrl_utils.nn import setup_optimizer, soft_update_params

import copy
from bbrl.agents.gymnasium import ParallelGymAgent

from functools import partial
from .actors import ContinuousQAgent, ContinuousDeterministicActor, AddGaussianNoise



class TD3(EpochBasedAlgo):
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

        # Define the agents and optimizers for TD3
        obs_size, act_size = 16, 2
        self.critic_1 = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        ).with_prefix("critic_1/")

        self.critic_2 = ContinuousQAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
        ).with_prefix("critic_2/")

        self.target_critic_1 = copy.deepcopy(self.critic_1).with_prefix("target-critic_1/")
        self.target_critic_2 = copy.deepcopy(self.critic_2).with_prefix("target-critic_2/")

        self.actor = ContinuousDeterministicActor()

        # As an alternative, you can use `AddOUNoise`
        noise_agent = AddGaussianNoise(torch.tensor(cfg.algorithm.action_noise))

        self.train_policy = Agents(self.actor, noise_agent)
        self.eval_policy = self.actor

        # Define agents over time
        self.t_actor = TemporalAgent(self.actor)
        self.t_critic_1 = TemporalAgent(self.critic_1)
        self.t_critic_2 = TemporalAgent(self.critic_2)
        self.t_target_critic_1 = TemporalAgent(self.target_critic_1)
        self.t_target_critic_2 = TemporalAgent(self.target_critic_2)

        # Configure the optimizer
        self.actor_optimizer = setup_optimizer(cfg.actor_optimizer, self.actor)
        self.critic_optimizer_1 = setup_optimizer(cfg.critic_optimizer, self.critic_1)
        self.critic_optimizer_2 = setup_optimizer(cfg.critic_optimizer, self.critic_2)
        
       
def run_td3(td3: TD3):
    loss_interval =  td3.cfg.algorithm.loss_interval
    eval_interval =  td3.cfg.algorithm.eval_interval
    maj_actor = 0
    for rb in td3.iter_replay_buffers():
        rb_workspace = rb.get_shuffled(td3.cfg.algorithm.batch_size)

        # Compute the critic loss
        td3.t_critic_1(rb_workspace, t=0, n_steps=2)
        td3.t_critic_2(rb_workspace, t=0, n_steps=2)
        with torch.no_grad():
            td3.t_target_critic_1(rb_workspace, t=0, n_steps=2)
            td3.t_target_critic_2(rb_workspace, t=0, n_steps=2)

        q_values_1, q_values_2, terminated, reward, target_q_values_1, target_q_values_2 = rb_workspace[
            "critic_1/q_value", "critic_2/q_value", "env/terminated", "env/reward", "target-critic_1/q_value", "target-critic_2/q_value"
        ]

        # Determines whether values of the critic should be propagated
        must_bootstrap = ~terminated

        target_q_values = torch.min(target_q_values_1, target_q_values_2)
        
        # Compute critic loss
        critic_loss_1 = compute_critic_loss(
            td3.cfg, reward, must_bootstrap, q_values_1, target_q_values
        )

        critic_loss_2 = compute_critic_loss(
            td3.cfg, reward, must_bootstrap, q_values_2, target_q_values
        )
        
        # Gradient step (critic_1)
        td3.logger.add_log("critic_loss_1", critic_loss_1, td3.nb_steps)
        td3.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        torch.nn.utils.clip_grad_norm_(
            td3.critic_1.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        td3.critic_optimizer_1.step()

        # Gradient step (critic_2)
        td3.logger.add_log("critic_loss_2", critic_loss_2, td3.nb_steps)
        td3.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        torch.nn.utils.clip_grad_norm_(
            td3.critic_2.parameters(), td3.cfg.algorithm.max_grad_norm
        )
        td3.critic_optimizer_2.step()

        # Compute the actor loss
        td3.t_actor(rb_workspace, t=0, n_steps=2)
        td3.t_critic_1(rb_workspace, t=0, n_steps=2)
        q_values = rb_workspace["critic_1/q_value"]
        actor_loss = compute_actor_loss(q_values)


        if maj_actor % 3 == 0:
            maj_actor += 1
            # Gradient step (actor)
            td3.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                td3.actor.parameters(), td3.cfg.algorithm.max_grad_norm
            )
            td3.actor_optimizer.step()

        # Soft update of target q function
        soft_update_params(
            td3.critic_1, td3.target_critic_1, td3.cfg.algorithm.tau_target
        )
        soft_update_params(
            td3.critic_2, td3.target_critic_2, td3.cfg.algorithm.tau_target
        )

        if td3.nb_steps > loss_interval:
            loss_interval += td3.cfg.algorithm.loss_interval
            print(f"Step: {td3.nb_steps}, Actor loss: {actor_loss}, Critic loss: {critic_loss_1}, {critic_loss_2}")
        
        # Evaluate the actor if needed
        if td3.nb_steps > eval_interval:
            eval_interval += td3.cfg.algorithm.eval_interval
            print("nb_steps", td3.nb_steps)
            torch.save(td3.actor.model.state_dict(), 'crab_model.pth')
            for i in range(3):
                subprocess.run(['master-dac', 'rld', 'stk-race', 'actor0.zip', '--hide', f'--num-karts={5}'])
        
        if td3.evaluate():
            continue


def compute_critic_loss(cfg, reward: torch.Tensor, must_bootstrap: torch.Tensor, q_values: torch.Tensor, target_q_values: torch.Tensor):
    """Compute the DDPG critic loss from a sample of transitions

    :param cfg: The configuration
    :param reward: The reward (shape 2xB)
    :param must_bootstrap: Must bootstrap flag (shape 2xB)
    :param q_values: The computed Q-values (shape 2xB)
    :param target_q_values: The Q-values computed by the target critic (shape 2xB)
    :return: the loss (a scalar)
    """
    # Compute temporal difference
    qvals = q_values[0]
    target = reward[1] + cfg.algorithm.discount_factor * target_q_values[1] * must_bootstrap[1]

    # Compute critic loss (no need to use must_bootstrap here since we are dealing with "full" transitions)
    mse = nn.MSELoss()
    critic_loss = mse(target, qvals)
    return critic_loss
def compute_actor_loss(q_values):
    """Returns the actor loss

    :param q_values: The q-values (shape 2xB)
    :return: A scalar (the loss)
    """

    loss = - q_values[0].mean()

    return loss