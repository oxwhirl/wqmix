# Weighted QMIX: Expanding Monotonic Value Function Factorisation (NeurIPS 2020)

Based on PyMARL (https://github.com/oxwhirl/pymarl/). Please refer to that repo for more documentation.

This repo contains the cleaned-up code that was used in "Weighted QMIX: Expanding Monotonic Value Function Factorisation" (https://arxiv.org/abs/2006.10800).

## Included in this repo

In particular implementations for:
- OW-QMIX
- CW-QMIX
- Versions of DDPG & SAC used in the paper

We thank the authors of "QPLEX: Duplex Dueling Multi-Agent Q-Learning" (https://arxiv.org/abs/2008.01062) for their implementation of QPLEX (https://github.com/wjh720/QPLEX/), whose implementation we used. The exact implementation we used is included in this repo.

Note that in the repository the naming of certain hyper-parameters and concepts is a little different to the paper:
- &alpha; in the paper is `w` in the code
- Optimistic Weighting (OW) is referred to as `hysteretic_qmix`

## For all SMAC experiments we used SC2.4.6.2.69232 (not SC2.4.10). The underlying dynamics are sufficiently different that you **cannot** compare runs across the 2 versions!
The `install_sc2.sh` script will install SC2.4.6.2.69232.

## Running experiments

The config files (`src/config/algs/*.yaml`) contain default hyper-parameters for the respective algorithms.
These were changed when running the experiments for the paper (`epsilon_anneal_time = 1000000` for the robustness to exploration experiments, and `w=0.1` for the predator prey punishment experiments for instance).
Please see the Appendix of the paper for the exact hyper-parameters used. 

Set `central_mixer=atten` to get the modified mixing network architecture that was used for the final experiment on `corridor` in the paper.

As an example, to run the OW-QMIX on 3s5z with epsilon annealed over 1mil timesteps using docker:
```shell
bash run.sh $GPU python3 src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=3s5z w=0.5 epsilon_anneal_time=1000000
```

## Citing

Bibtex:
```
@inproceedings{rashid2020weighted,
  title={Weighted QMIX: Expanding Monotonic Value Function Factorisation},
  author={Rashid, Tabish and Farquhar, Gregory and Peng, Bei and Whiteson, Shimon},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```
