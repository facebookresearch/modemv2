# MoDem-V2: Visuo-Motor World Models for Real-World Robot Manipulation

Original PyTorch implementation of [ MoDem-V2: Visuo-Motor World Models for Real-World Robot Manipulation](#) by

[Patrick Lancaster](https://palanc.github.io), [Nicklas Hansen](https://nicklashansen.github.io/), [Aravind Rajeswaran](https://aravindr93.github.io/) [Vikash Kumar](https://vikashplus.github.io/) (Meta AI, UC San Diego)

<p align="center">
  <img width="24.5%" src="https://i.imgur.com/X5jL6eq.gif">
  <img width="24.5%" src="https://i.imgur.com/IdAassv.gif">
  <img width="24.5%" src="https://i.imgur.com/nATNRHA.gif">
  <img width="24.5%" src="https://i.imgur.com/o8bPihp.gif">
   <a href="https://arxiv.org/abs/2309.14236">[Paper]</a>&emsp;<a href="https://sites.google.com/view/modem-v2">[Website]</a>
</p>


## Method

**MoDem-V2** combines the sample efficiency of the original **MoDem** with conservative exploration in order to quickly and safely learn manipulation skills on real robots.

<p align="center">
  <img width="80%" src="https://i.imgur.com/mC2K8Qj.jpeg">
</p>


## Citation

If you use this repo in your research, please consider citing the paper as follows:

```
@article{lancaster2023modem,
  title={MoDem-V2: Visuo-Motor World Models for Real-World Robot Manipulation},
  author={Lancaster, Patrick and Hansen, Nicklas and Rajeswaran, Aravind and Kumar, Vikash},
  journal={arXiv preprint arXiv:2309.14236},
  year={2023}
}
```


## Instructions

We assume that your machine has a CUDA-enabled GPU, a local copy of MuJoCo 2.1.x installed, and at least 80GB of memory. Then, create a conda environment with `conda env create -f environment.yml`, and add `modemv2/tasks/robohive` to your `PYTHONPATH`. Activate the new environment with `conda activate modemv2` and then install mujoco_py with `pip install -e ./mujoco_py`. You will also need to configure `wandb_entity` in `modemv2/cfgs/config.yaml`. Demonstrations are made available [here](https://github.com/palanc/modem/releases/tag/0.1.0); untar them into `modemv2/demonstrations`. 

Launch MoDem-V2 training with the scripts in `scripts/franka`. Note that the scripts should be executed from the `modemv2` directory. For example, to train a single seed of MoDem-V2 on the bin picking task:
```
sh scripts/franka/bin_pick/modemv2.sh
```

Append an argument of 1 in order to train 5 seeds on the cluster, for example:
```
sh scripts/franka/bin_pick/modemv2.sh 1
```

Alternatively, append an argument of 2 in order to truncate each stage of training and verify that the code has been setup correctly, for example:
```
sh scripts/franka/bin_pick/modemv2.sh 2
```


## License & Acknowledgements

This codebase is based on the original [MoDem](https://github.com/facebookresearch/modem) implementation. The majority of MoDem-V2 is licensed under CC-BY-NC, however portions of the project are available under separate license terms: mujoco-py is licensed under the following license: https://github.com/openai/mujoco-py/blob/master/LICENSE.md; robohive is licensed under the following license: https://github.com/vikashplus/robohive/blob/main/LICENSE.
