# Model-free Deep Reinforcement Learning Algorithms Implementation

### Overview

This is a repository about the implementation of many model-free deep reinforcement learning algorithms.

- Policy gradient ( On-Policy, Monte Carlo, Continuous Action Space )
- Actor-Critic ( Off-Policy, Temporal Difference, Continuous Action Space )
- Deep Q-Learning ( Off-Policy, Temporal Difference, Discrete Action Space )
- Deep Deterministic Policy Gradient ( Off-Policy, Temporal Difference, Continuous Action Space )
- Twin Delayed Deep Deterministic Policy Gradient ( Off-Policy, Temporal Difference, Continuous Action Space )
- Proximal Policy Optimization ( On-Policy, Monte Carlo, Discrete Action Space )


### Usage

Test :
``` 
python algos/DDPG/main.py --test
```
Train:
```
python algos/DDPG/main.py --train
```