import torch
from bbrl.agents import TemporalAgent, KWAgentWrapper
from bbrl_utils.algorithms import EpisodicAlgo, iter_partial_episodes
from bbrl_utils.nn import build_mlp, setup_optimizer, soft_update_params

import copy
from bbrl_utils.notebook import setup_tensorboard
from bbrl_utils.nn import copy_parameters
from bbrl.agents.gymnasium import ParallelGymAgent
from bbrl_utils.notebook import video_display
from functools import partial
from .actors import VAgent, DiscretePolicy
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from bbrl.utils.functional import gae
import subprocess


class PPOClip(EpisodicAlgo):
    def __init__(self, cfg, make_env, model_str='pystk_actor.pth'):
        super().__init__(cfg, autoreset=True)

        self.train_env = ParallelGymAgent(
            partial(
                make_env,
                autoreset=True,
            ),
            cfg.algorithm.n_envs,
        ).seed(cfg.algorithm.seed)

        obs_size, act_size = 16, 7

        # self.train_policy = globals()[cfg.algorithm.policy_type](
        #     obs_size,
        #     cfg.algorithm.architecture.actor_hidden_size,
        #     act_size,
        # ).with_prefix("current_policy/")
        # self.train_policy.model.load_state_dict(torch.load("crab_model4.pth"))
        self.train_policy = globals()[cfg.algorithm.policy_type](
            obs_size,
            cfg.algorithm.architecture.actor_hidden_size,
            act_size,
        ).with_prefix("current_policy/")
        # self.train_policy.load_state_dict(torch.load(model_str))

        self.eval_policy = KWAgentWrapper(
            self.train_policy, 
            stochastic=False,
            predict_proba=False,
            compute_entropy=False,
        )

        self.critic_agent = VAgent(
            obs_size, cfg.algorithm.architecture.critic_hidden_size
        ).with_prefix("critic/")
        # self.critic_agent.model.load_state_dict(torch.load("critic4.pth"))
        self.old_critic_agent = copy.deepcopy(self.critic_agent).with_prefix("old_critic/")

        self.old_policy = copy.deepcopy(self.train_policy)
        self.old_policy.with_prefix("old_policy/")

        self.policy_optimizer = setup_optimizer(
            cfg.optimizer, self.train_policy
        )
        # self.policy_optimizer.load_state_dict(torch.load("policy_optimizer3.pth"))
        self.critic_optimizer = setup_optimizer(
            cfg.optimizer, self.critic_agent
        )
        # self.critic_optimizer.load_state_dict(torch.load("critic_optimizer3.pth"))

def run(ppo_clip: PPOClip):
    """
    Run the PPO algorithm with a discrete policy
    """
    cfg = ppo_clip.cfg
    loss_interval =  ppo_clip.cfg.algorithm.loss_interval
    eval_interval =  ppo_clip.cfg.algorithm.eval_interval

    timenow = str(datetime.now())[0:-10]
    timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
    writer = SummaryWriter('./runs/'+timenow)

    t_policy = TemporalAgent(ppo_clip.train_policy)
    t_old_policy = TemporalAgent(ppo_clip.old_policy)
    t_critic = TemporalAgent(ppo_clip.critic_agent)
    t_old_critic = TemporalAgent(ppo_clip.old_critic_agent)
    decay_rate = (0.5)**(1/50)
    decay = 1

    for train_workspace in iter_partial_episodes(
        ppo_clip, cfg.algorithm.n_steps
    ):
        # Run the current policy and evaluate the proba of its action according
        # to the old policy The old_policy can be run after the train_agent on
        # the same workspace because it writes a logprob_predict and not an
        # action. That is, it does not determine the action of the old_policy,
        # it just determines the proba of the action of the current policy given
        # its own probabilities

        with torch.no_grad():
            t_old_policy(
                train_workspace,
                t=0,
                n_steps=cfg.algorithm.n_steps,
                # Just computes the probability of the old policy's action
                # to get the ratio of probabilities
                predict_proba=True,
                compute_entropy=False,
            )

        # Compute the critic value over the whole workspace
        t_critic(train_workspace, t=0, n_steps=cfg.algorithm.n_steps)
        with torch.no_grad():
            t_old_critic(train_workspace, t=0, n_steps=cfg.algorithm.n_steps)

        ws_terminated, ws_reward, ws_v_value, ws_old_v_value = train_workspace[
            "env/terminated",
            "env/reward",
            "critic/v_values",
            "old_critic/v_values",
        ]
        center_path_distance = train_workspace["env/env_obs/center_path_distance"].squeeze(1)
        # print(center_path_distance.shape)
        # the critic values are clamped to move not too far away from the values of the previous critic
        if cfg.algorithm.clip_range_vf > 0:
            # Clip the difference between old and new values
            # NOTE: this depends on the reward scaling
            ws_v_value = ws_old_v_value + torch.clamp(
                ws_v_value - ws_old_v_value,
                -cfg.algorithm.clip_range_vf,
                cfg.algorithm.clip_range_vf,
            )
        
        rew = ws_reward[1:] - decay * center_path_distance[1:].abs()
        decay *= decay_rate

        # Compute the advantage using the (clamped) critic values
        with torch.no_grad():
            advantage = gae(
                # ws_reward[1:],
                rew,
                ws_v_value[1:],
                ~ws_terminated[1:],
                ws_v_value[:-1],
                cfg.algorithm.discount_factor,
                cfg.algorithm.gae,
            )

        ppo_clip.critic_optimizer.zero_grad()
        # target = ws_reward[1:] + cfg.algorithm.discount_factor * ws_old_v_value[1:].detach() * (1 - ws_terminated[1:].int())
        target = rew + cfg.algorithm.discount_factor * ws_old_v_value[1:].detach() * (1 - ws_terminated[1:].int())
        critic_loss = torch.nn.functional.mse_loss(ws_v_value[:-1], target) * cfg.algorithm.critic_coef
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            ppo_clip.critic_agent.parameters(), cfg.algorithm.max_grad_norm
        )
        ppo_clip.critic_optimizer.step()

        # We store the advantage into the transition_workspace
        if cfg.algorithm.normalize_advantage and advantage.shape[1] > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        train_workspace.set_full("advantage", torch.cat(
            (advantage, torch.zeros(1, advantage.shape[1]))
        ))
        transition_workspace = train_workspace.get_transitions()

        writer.add_scalar('Advantage Mean', advantage.mean().item(), ppo_clip.nb_steps)
        writer.add_scalar('Reward Mean', ws_reward.mean().item(), ppo_clip.nb_steps)

        # Inner optimization loop: we sample transitions and use them to learn
        # the policy
        for opt_epoch in range(cfg.algorithm.opt_epochs):
            if cfg.algorithm.batch_size > 0:
                sample_workspace = transition_workspace.select_batch_n(
                    cfg.algorithm.batch_size
                )
            else:
                sample_workspace = transition_workspace

            # Compute the policy loss

            # Compute the probability of the played actions according to the current policy
            # We do not replay the action: we use the one stored into the dataset
            # Hence predict_proba=True
            # Note that the policy is not wrapped into a TemporalAgent, but we use a single step
            # Compute the ratio of action probabilities
            # Compute the policy loss
            # (using cfg.algorithm.clip_range and torch.clamp)
            policy_advantage = sample_workspace["advantage"][0]
            ppo_clip.train_policy(sample_workspace, t=0, n_steps=1, predict_proba=True, compute_entropy=True)
            log_prob = sample_workspace["current_policy/logprob_predict"]
            old_log_prob = sample_workspace["old_policy/logprob_predict"]
            ratio = (log_prob - old_log_prob[0]).exp().squeeze(0)
            policy_loss = torch.min(
                ratio * policy_advantage, torch.clamp(ratio, 1.0 - cfg.algorithm.clip_range, 1.0 + cfg.algorithm.clip_range) * policy_advantage
            )
            policy_loss = policy_loss.mean()

            loss_policy = -cfg.algorithm.policy_coef * policy_loss

            # Entropy loss favors exploration Note that the standard PPO
            # algorithms do not have an entropy term, they don't need it because
            # the KL term is supposed to deal with exploration So, to run the
            # standard PPO algorithm, you should set
            # cfg.algorithm.entropy_coef=0
            entropy = sample_workspace["current_policy/entropy"]
            assert len(entropy) == 1, f"{entropy.shape}"
            entropy_loss = entropy[0].mean()
            loss_entropy = -cfg.algorithm.entropy_coef * entropy_loss

            # Store the losses for tensorboard display
            ppo_clip.logger.log_losses(
                critic_loss, entropy_loss, policy_loss, ppo_clip.nb_steps
            )
            ppo_clip.logger.add_log(
                "advantage", policy_advantage[0].mean(), ppo_clip.nb_steps
            )

            loss = loss_policy + loss_entropy

            ppo_clip.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                ppo_clip.train_policy.parameters(), cfg.algorithm.max_grad_norm
            )
            ppo_clip.policy_optimizer.step()

            writer.add_scalar('Loss', loss.item(), ppo_clip.nb_steps+opt_epoch)
            writer.add_scalar('Policy Loss', policy_loss.item(), ppo_clip.nb_steps+opt_epoch)
            writer.add_scalar('Critic Loss', critic_loss.item(), ppo_clip.nb_steps+opt_epoch)
            writer.add_scalar('Entropy Loss', entropy_loss.item(), ppo_clip.nb_steps+opt_epoch)
            # writer.add_scalar('Advantage', policy_advantage[0].mean(), ppo_clip.nb_steps+opt_epoch)


        # Copy parameters
        copy_parameters(ppo_clip.train_policy, ppo_clip.old_policy)
        copy_parameters(ppo_clip.critic_agent, ppo_clip.old_critic_agent)

        # Evaluates our current algorithm if needed
        #ppo_clip.evaluate()
        if ppo_clip.nb_steps > loss_interval:
            loss_interval += ppo_clip.cfg.algorithm.loss_interval
            print(f"Step: {ppo_clip.nb_steps}, Loss: {loss.item()} Critic Loss: {critic_loss.item()}")
        
        # Evaluate the actor if needed
        if ppo_clip.nb_steps > eval_interval:
            eval_interval += ppo_clip.cfg.algorithm.eval_interval
            print("nb_steps", ppo_clip.nb_steps)
            # torch.save(ppo_clip.train_policy.model.state_dict(), 'crab_model.pth')
            torch.save(ppo_clip.train_policy.state_dict(), 'pystk_actor.pth')
            #needing an actor0.zip zip file well formated for live evaluation
            for i in range(2):
                subprocess.run(['master-dac', 'rld', 'stk-race', 'actor0.zip', '--hide', f'--num-karts={5}'])

def run_double_discrete(ppo_clip: PPOClip):
    """
    Run the PPO algorithm with a double discrete policy
    """
    cfg = ppo_clip.cfg
    loss_interval =  ppo_clip.cfg.algorithm.loss_interval
    eval_interval =  ppo_clip.cfg.algorithm.eval_interval

    t_policy = TemporalAgent(ppo_clip.train_policy)
    t_old_policy = TemporalAgent(ppo_clip.old_policy)
    t_critic = TemporalAgent(ppo_clip.critic_agent)
    t_old_critic = TemporalAgent(ppo_clip.old_critic_agent)

    for train_workspace in iter_partial_episodes(
        ppo_clip, cfg.algorithm.n_steps
    ):
        # Run the current policy and evaluate the proba of its action according
        # to the old policy The old_policy can be run after the train_agent on
        # the same workspace because it writes a logprob_predict and not an
        # action. That is, it does not determine the action of the old_policy,
        # it just determines the proba of the action of the current policy given
        # its own probabilities

        with torch.no_grad():
            t_old_policy(
                train_workspace,
                t=0,
                n_steps=cfg.algorithm.n_steps,
                # Just computes the probability of the old policy's action
                # to get the ratio of probabilities
                predict_proba=True,
                compute_entropy=False,
            )

        # Compute the critic value over the whole workspace
        t_critic(train_workspace, t=0, n_steps=cfg.algorithm.n_steps)
        with torch.no_grad():
            t_old_critic(train_workspace, t=0, n_steps=cfg.algorithm.n_steps)

        ws_terminated, ws_reward, ws_v_value, ws_old_v_value = train_workspace[
            "env/terminated",
            "env/reward",
            "critic/v_values",
            "old_critic/v_values",
        ]
        # ws_reward = ws_reward - abs(train_workspace["env/env_obs/velocity"])[:,:,0] + abs(train_workspace["env/env_obs/velocity"])[:,:,2] / 200

        # the critic values are clamped to move not too far away from the values of the previous critic
        if cfg.algorithm.clip_range_vf > 0:
            # Clip the difference between old and new values
            # NOTE: this depends on the reward scaling
            ws_v_value = ws_old_v_value + torch.clamp(
                ws_v_value - ws_old_v_value,
                -cfg.algorithm.clip_range_vf,
                cfg.algorithm.clip_range_vf,
            )

        # Compute the advantage using the (clamped) critic values
        with torch.no_grad():
            advantage = gae(
                ws_reward[1:],
                ws_v_value[1:],
                ~ws_terminated[1:],
                ws_v_value[:-1],
                cfg.algorithm.discount_factor,
                cfg.algorithm.gae,
            )

        ppo_clip.critic_optimizer.zero_grad()
        target = ws_reward[1:] + cfg.algorithm.discount_factor * ws_old_v_value[1:].detach() * (1 - ws_terminated[1:].int())
        critic_loss = torch.nn.functional.mse_loss(ws_v_value[:-1], target) * cfg.algorithm.critic_coef
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            ppo_clip.critic_agent.parameters(), cfg.algorithm.max_grad_norm
        )
        ppo_clip.critic_optimizer.step()

        # We store the advantage into the transition_workspace
        if cfg.algorithm.normalize_advantage and advantage.shape[1] > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        train_workspace.set_full("advantage", torch.cat(
            (advantage, torch.zeros(1, advantage.shape[1]))
        ))
        transition_workspace = train_workspace.get_transitions()

        # Inner optimization loop: we sample transitions and use them to learn
        # the policy
        for opt_epoch in range(cfg.algorithm.opt_epochs):
            if cfg.algorithm.batch_size > 0:
                sample_workspace = transition_workspace.select_batch_n(
                    cfg.algorithm.batch_size
                )
            else:
                sample_workspace = transition_workspace

            # Compute the policy loss

            # Compute the probability of the played actions according to the current policy
            # We do not replay the action: we use the one stored into the dataset
            # Hence predict_proba=True
            # Note that the policy is not wrapped into a TemporalAgent, but we use a single step
            # Compute the ratio of action probabilities
            # Compute the policy loss
            # (using cfg.algorithm.clip_range and torch.clamp)
            policy_advantage = sample_workspace["advantage"][0]
            ppo_clip.train_policy(sample_workspace, t=0, n_steps=1, predict_proba=True, compute_entropy=True)
            log_prob = sample_workspace["current_policy/logprob_predict"]
            old_log_prob = sample_workspace["old_policy/logprob_predict"]
            ratio = (log_prob - old_log_prob[0]).exp().squeeze(0)
            policy_loss = torch.min(
                ratio * policy_advantage, torch.clamp(ratio, 1.0 - cfg.algorithm.clip_range, 1.0 + cfg.algorithm.clip_range) * policy_advantage
            )
            policy_loss = policy_loss.mean()

            loss_policy = -cfg.algorithm.policy_coef * policy_loss

            # Entropy loss favors exploration Note that the standard PPO
            # algorithms do not have an entropy term, they don't need it because
            # the KL term is supposed to deal with exploration So, to run the
            # standard PPO algorithm, you should set
            # cfg.algorithm.entropy_coef=0
            entropy = sample_workspace["current_policy/entropy"]
            assert len(entropy) == 1, f"{entropy.shape}"
            entropy_loss = entropy[0].mean()
            loss_entropy = -cfg.algorithm.entropy_coef * entropy_loss

            # Store the losses for tensorboard display
            ppo_clip.logger.log_losses(
                critic_loss, entropy_loss, policy_loss, ppo_clip.nb_steps
            )
            ppo_clip.logger.add_log(
                "advantage", policy_advantage[0].mean(), ppo_clip.nb_steps
            )

            loss = loss_policy + loss_entropy

            ppo_clip.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                ppo_clip.train_policy.parameters(), cfg.algorithm.max_grad_norm
            )
            ppo_clip.policy_optimizer.step()

        # Copy parameters
        copy_parameters(ppo_clip.train_policy, ppo_clip.old_policy)
        copy_parameters(ppo_clip.critic_agent, ppo_clip.old_critic_agent)

        # Evaluates our current algorithm if needed
        #ppo_clip.evaluate()
        if ppo_clip.nb_steps > loss_interval:
            loss_interval += ppo_clip.cfg.algorithm.loss_interval
            print(f"Step: {ppo_clip.nb_steps}, Loss: {loss.item()} Critic Loss: {critic_loss.item()}")
        
        # Evaluate the actor if needed
        if ppo_clip.nb_steps > eval_interval:
            eval_interval += ppo_clip.cfg.algorithm.eval_interval
            print("nb_steps", ppo_clip.nb_steps)
            from datetime import datetime
            now = datetime.now()
            torch.save(ppo_clip.train_policy.state_dict(), f'model_ppo/actor_{now.strftime("%Y%m%d%H%M%S")}.pth')
            torch.save(ppo_clip.train_policy.state_dict(), f'crab_model.pth')
            torch.save(ppo_clip.critic_agent.model.state_dict(), f'model_ppo/critic_{now.strftime("%Y%m%d%H%M%S")}.pth')
            for i in range(2):
                subprocess.run(['master-dac', 'rld', 'stk-race', 'actor0.zip', '--hide', f'--num-karts={5}'])