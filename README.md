# SuperTuxKart-Piloting-Project
Proposed work on building an agent to play to SuperTuxKart, on a Reinforcement Learning at M2A (Sorbonne Université)

## Structure

- `agents` contain our 2 main agents `bc_final.zip` from behavioral cloning and `ppo_final.zip` from our PPO training.
- `datasets` contains an example of dataset for behavioral cloning training.
- `learn` contains our three main RL algorithms: Soft Actor Critic (sac), Deep Deterministic Policy Gradient (ddpg) and Proximal Policy Optimization (ppo), in the form of a folder `player_{METHOD}`.
- `logs` contains an example of a expert trajectory used for behavioral cloning.
- `scrpits` contains BC-dataset creation scripts, and all training scripts for Imitation learning (BC and GAIL).


## Main usages

### Learn

```sh
# To be run from the learn directory
PYTHONPATH=. python -m player_{METHOD}.learn
```

### Test agents
```sh
# Usage: master-dac rld stk-race [OPTIONS] [ZIP_FILES|MODULE]...
master-dac rld stk-race --hide {ACTOR}.zip
```
