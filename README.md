# Spinning Up RL

This repo consists of my implementations on core RL algorithms including:
- Vanilla Policy Gradient (VPG)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971)
- [Twin Delayed DDPG (TD3)](https://arxiv.org/abs/1802.09477v3)
- [Soft Actor Critic (SAC)](https://arxiv.org/abs/1801.01290)
- [Hindsight Experience Replay (HER)](https://arxiv.org/abs/1707.01495)

The implementations are based on the tutorial from OpenAI's [SpinningUp](https://spinningup.openai.com/en/latest/). In order to run the algorithms, follow the [installation instructions](https://spinningup.openai.com/en/latest/user/installation.html) from SpinningUp.
I also implemented HER that wraps around the [skrl](https://skrl.readthedocs.io/en/latest/) library to train for [FetchPush-v1 and FetchSlide-v1](https://github.com/geyang/gym-fetch) tasks. In order to run this, you additionally need to [install the skrl library](https://skrl.readthedocs.io/en/latest/intro/installation.html). 

Python executables:
- `basic/`
    - `q_learning.py`
    - `pg_simple.py`
    - `pg_reward_to_go.py`
    - `vpg.py`
    - `trpo.py`
    - `ppo_clip.py`
    - `ddpg.py`
    - `td3.py`
    - `sac.py`
- `fetch/`
    - `ddpg_fetch.py`
    - `sac_fetch.py`

