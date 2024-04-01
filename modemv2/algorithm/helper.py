# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from pathlib import Path
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from env import make_env
import multiprocessing as mp
from copy import deepcopy
from tasks.franka import recompute_real_rwd, FrankaTask
from robohive.utils.tensor_utils import stack_tensor_dict_list
from robohive.logger.grouped_datasets import Trace
import git
import hydra

__REDUCE__ = lambda b: "mean" if b else "none"


def l1(pred, target, reduce=False):
    """Computes the L1-loss between predictions and targets."""
    return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
    """Computes the MSE loss between predictions and targets."""
    return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))


def _get_out_shape(in_shape, layers):
    """Utility function. Returns the output shape of a network for a given input shape."""
    x = torch.randn(*in_shape).unsqueeze(0)
    return (
        (nn.Sequential(*layers) if isinstance(layers, list) else layers)(x)
        .squeeze(0)
        .shape
    )


def orthogonal_init(m):
    """Orthogonal layer initialization."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def soft_update_params(m, m_target, tau):
    """Update slow-moving average of online network (target network) at rate tau."""
    with torch.no_grad():
        for p, p_target in zip(m.parameters(), m_target.parameters()):
            p_target.data.lerp_(p.data, tau)


def set_requires_grad(net, value):
    """Enable/disable gradients for a given (sub)network."""
    for param in net.parameters():
        param.requires_grad_(value)


class TruncatedNormal(pyd.Normal):
    """Utility class implementing the truncated normal distribution."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class NormalizeImg(nn.Module):
    """Module that divides (pixel) observations by 255."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.div(255.0)


class Flatten(nn.Module):
    """Module that flattens its input to a (batched) vector."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

def enc(cfg):
    """Returns our MoDem encoder that takes a stack of 224x224 frames as input."""
    if cfg.img_size <= 0:
        return None
        
    C = int(cfg.obs_shape[1])
    encoders = []
    for _ in range(cfg.obs_shape[0]):
        layers = [
            NormalizeImg(),
            nn.Conv2d(C, cfg.num_channels, 7, stride=2), nn.ReLU(),
            nn.Conv2d(cfg.num_channels, cfg.num_channels, 5, stride=2), nn.ReLU(),
            nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU(),
            nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU(),
            nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU(),
            nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2), nn.ReLU(),
        ]
        out_shape = _get_out_shape((C, cfg.img_size, cfg.img_size), layers)
        layers.extend([Flatten(), nn.Linear(np.prod(out_shape), cfg.latent_dim)])
        encoders.append(nn.Sequential(*layers))
    return encoders


def state_enc(cfg):
    """Returns a proprioceptive state encoder + modality fuse."""
    return (
        nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.enc_dim),
            nn.ELU(),
            nn.Linear(cfg.enc_dim, cfg.latent_dim),
        ),
        nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.enc_dim),
            nn.ELU(),
            nn.Linear(cfg.enc_dim, cfg.latent_dim),
        ),
    )


def mlp(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
    """Returns an MLP."""
    if isinstance(mlp_dim, int):
        mlp_dim = [mlp_dim, mlp_dim]
    return nn.Sequential(
        nn.Linear(in_dim, mlp_dim[0]),
        act_fn,
        nn.Linear(mlp_dim[0], mlp_dim[1]),
        act_fn,
        nn.Linear(mlp_dim[1], out_dim),
    )


def q(cfg, act_fn=nn.ELU()):
    """Returns a Q-function that uses Layer Normalization."""
    return nn.Sequential(
        nn.Linear(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim),
        nn.LayerNorm(cfg.mlp_dim),
        nn.Tanh(),
        nn.Linear(cfg.mlp_dim, cfg.mlp_dim),
        act_fn,
        nn.Linear(cfg.mlp_dim, 1),
    )


class RandomShiftsAug(nn.Module):
    """
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """

    def __init__(self, cfg):
        super().__init__()
        self.pad = int(
            cfg.img_size / 21
        )  # maintain same padding ratio as in original implementation

    def forward(self, x):
        n, v, c, h, w = x.size()
        assert h == w

        x = x.reshape(-1,*x.shape[-3:])

        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n*v, 1, 1, 1)
        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n*v, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False).reshape((n,v,c,h,w))


class Episode(object):
    """Storage object for a single episode."""

    def __init__(self, cfg, init_obs, init_state=None):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.obs = torch.empty(
            (cfg.episode_length + 1, *init_obs.shape),
            dtype=torch.uint8,
            device=self.device,
        )
        self.obs[0] = torch.tensor(init_obs, dtype=torch.uint8, device=self.device)
        self.state = torch.empty(
            (cfg.episode_length + 1, *init_state.shape),
            dtype=torch.float32,
            device=self.device,
        )
        self.state[0] = torch.tensor(
            init_state, dtype=torch.float32, device=self.device
        )
        self.action = torch.empty(
            (cfg.episode_length, cfg.action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.reward = torch.empty(
            (cfg.episode_length,), dtype=torch.float32, device=self.device
        )
        self.cumulative_reward = 0
        self.done = False
        self._idx = 0

    @classmethod
    def from_trajectory(cls, cfg, obs, states, action, reward, done=None):
        """Constructs an episode from a trajectory."""
        episode = cls(cfg, obs[0], states[0])
        episode.obs[1:] = torch.tensor(
            obs[1:], dtype=episode.obs.dtype, device=episode.device
        )
        episode.state[1:] = torch.tensor(
            states[1:], dtype=episode.state.dtype, device=episode.device
        )
        episode.action = torch.tensor(
            action, dtype=episode.action.dtype, device=episode.device
        )
        episode.reward = torch.tensor(
            reward, dtype=episode.reward.dtype, device=episode.device
        )
        episode.cumulative_reward = torch.sum(episode.reward)
        episode.done = True
        episode._idx = cfg.episode_length
        return episode

    def __len__(self):
        return self._idx

    @property
    def first(self):
        return len(self) == 0

    def __add__(self, transition):
        self.add(*transition)
        return self

    def add(self, obs, state, action, reward, done):
        self.obs[self._idx + 1] = torch.tensor(
            obs, dtype=self.obs.dtype, device=self.obs.device
        )
        self.state[self._idx + 1] = torch.tensor(
            state, dtype=self.state.dtype, device=self.state.device
        )
        self.action[self._idx] = action
        self.reward[self._idx] = reward
        self.cumulative_reward += reward
        self.done = done
        self._idx += 1


def trace2episodes(cfg, env, trace, exclude_fails=False, is_demo=False):

    episodes = []

    for pname, pdata in trace.trace.items():                  
        successful_trial = True

        if not is_demo or exclude_fails:
            assert('success' in pdata)
            assert(pdata['success'].all() or not pdata['success'].any())
            successful_trial = pdata['success'].all()
        
        if exclude_fails and not successful_trial:
            print('skipping trial')
            continue

        # Get images
        views = []            
        if cfg.img_size <= 0:
            views.append(np.zeros((cfg.episode_length+1,3,10,10)))
        else:
            for i, cam in enumerate(cfg.camera_views):
                rgb_key = 'env_infos/visual_dict/rgb:'+cam+':'+str(cfg.img_size)+'x'+str(cfg.img_size)+':2d'
                d_key = 'env_infos/visual_dict/d:'+cam+':'+str(cfg.img_size)+'x'+str(cfg.img_size)+':2d'
                if (rgb_key not in pdata) or (d_key not in pdata):
                    rgb_key = 'env_infos/visual_dict/rgb:'+cam+':240x424:2d'
                    d_key = 'env_infos/visual_dict/d:'+cam+':240x424:2d'                      
                assert(rgb_key in pdata
                    and d_key in pdata)
                lc = cfg.left_crops[i]
                tc = cfg.top_crops[i]                
                rgb_imgs = pdata[rgb_key][:].transpose(0,3,1,2)
                rgb_imgs = rgb_imgs[:cfg.episode_length+1,:,tc:tc+cfg.img_size,lc:lc+cfg.img_size]
                depth_imgs = pdata[d_key][:]
                depth_imgs = depth_imgs[:cfg.episode_length+1,:,tc:tc+cfg.img_size,lc:lc+cfg.img_size]
                views.append(np.concatenate([rgb_imgs, depth_imgs], axis=1))
        obs = np.stack(views, axis=1)

        if 'BinPush' in cfg.task:
            franka_task = FrankaTask.BinPush
        elif 'HangPush' in cfg.task:
            franka_task = FrankaTask.HangPush
        elif 'PlanarPush' in cfg.task:
            franka_task = FrankaTask.PlanarPush
        elif 'BinPick' in cfg.task:
            franka_task = FrankaTask.BinPick
        elif 'BinReorient' in cfg.task:
            franka_task = FrankaTask.BinReorient
        else:
            raise NotImplementedError()

        qp = pdata['env_infos/obs_dict/qp'][:cfg.episode_length+1]
        qv = pdata['env_infos/obs_dict/qv'][:cfg.episode_length+1]
        grasp_pos = pdata['env_infos/obs_dict/grasp_pos'][:cfg.episode_length+1]
        grasp_rot = pdata['env_infos/obs_dict/grasp_rot'][:cfg.episode_length+1]
        obj_err = pdata['env_infos/obs_dict/object_err'][:cfg.episode_length+1]
        tar_err = pdata['env_infos/obs_dict/target_err'][:cfg.episode_length+1]
        assert((np.abs(qp[1:]-qp[0]) > 1e-5).any())
        assert((np.abs(grasp_pos[1:]-grasp_pos[0]) > 1e-5).any())
        assert((np.abs(grasp_rot[1:]-grasp_rot[0]) > 1e-5).any())

        if cfg.img_size > 0:    
            if franka_task == FrankaTask.BinReorient:
                state = np.concatenate([qp[:,:17],
                                        qv[:,:17],
                                        grasp_pos,
                                        grasp_rot], axis=1)
            elif not cfg.real_robot:      
                state = np.concatenate([qp[:,:8],
                                        qv[:,:8],
                                        grasp_pos,
                                        grasp_rot], axis=1)
            else:
                state = np.concatenate([qp[:,:9],
                                        qv[:,:9],
                                        grasp_pos,
                                        grasp_rot], axis=1)                
        else:   
            assert((np.abs(obj_err[1:]-obj_err[0]) > 1e-5).any())
            assert((np.abs(tar_err[1:]-tar_err[0]) > 1e-5).any())

            if len(tar_err.shape) < 2:
                tar_err = tar_err[:,np.newaxis]

            state = np.concatenate([qp,
                                    qv,
                                    grasp_pos,
                                    grasp_rot,
                                    obj_err,
                                    tar_err], axis=1)

        state = torch.tensor(state, dtype=torch.float32)


        actions = np.array(pdata['actions'])[:cfg.episode_length]
        
        if not cfg.real_robot and (cfg.task.startswith('franka-FrankaPlanarPush') or cfg.task.startswith('franka-FrankaBinPush')):
            actions = np.clip(actions,-1.0,1.0)

        if not((actions >= -1.0).all() and (actions <= 1.0).all()):
            print('Found ep w/ oob actions, min {}, max {}'.format(actions.min(), actions.max()))
        actions = np.clip(actions,-1.0,1.0)
        assert (franka_task == FrankaTask.BinReorient and actions.shape[1] == 16) or actions.shape[1] == 7  

        if franka_task == FrankaTask.BinPick:
            aug_actions = np.zeros((actions.shape[0],6),dtype=np.float32)
            aug_actions[:,:3] = actions[:,:3]
            yaw = (0.5*actions[:,5]+0.5)*(env.unwrapped.pos_limits['eef_high'][5]-env.unwrapped.pos_limits['eef_low'][5])+env.unwrapped.pos_limits['eef_low'][5]
            aug_actions[:,3] = np.cos(yaw)
            aug_actions[:,4] = np.sin(yaw)
            aug_actions[:,5] = actions[:,6]
        elif franka_task == FrankaTask.BinPush or franka_task == FrankaTask.HangPush: 
            aug_actions = np.zeros((actions.shape[0],3),dtype=np.float32)
            aug_actions[:,:3] = actions[:,:3]
        elif franka_task == FrankaTask.PlanarPush:
            aug_actions = np.zeros((actions.shape[0],5),dtype=np.float32)
            aug_actions[:,:3] = actions[:,:3]
            yaw = (0.5*actions[:,5]+0.5)*(env.unwrapped.pos_limits['eef_high'][5]-env.unwrapped.pos_limits['eef_low'][5])+env.unwrapped.pos_limits['eef_low'][5]
            aug_actions[:,3] = np.cos(yaw)
            aug_actions[:,4] = np.sin(yaw)
        elif franka_task == FrankaTask.BinReorient:
            aug_actions = np.zeros((actions.shape[0],15),dtype=np.float32)
            aug_actions[:,:3] = actions[:,:3]
            yaw = (0.5*actions[:,5]+0.5)*(env.unwrapped.pos_limits['eef_high'][5]-env.unwrapped.pos_limits['eef_low'][5])+env.unwrapped.pos_limits['eef_low'][5]
            aug_actions[:,3] = np.cos(yaw)
            aug_actions[:,4] = np.sin(yaw)   
            aug_actions[:,5:] = actions[:,6:]         
        else:
            raise NotImplementedError()

        actions = aug_actions

        if cfg.real_robot:
            if not cfg.dense_reward:
                rewards = torch.zeros((cfg.episode_length,), 
                                        dtype=torch.float32, 
                                        device=state.device)#-1.0            
                if franka_task == FrankaTask.BinPick or franka_task==FrankaTask.BinReorient:
                    if successful_trial:
                        rewards = recompute_real_rwd(cfg, state, obs[:,0,:3], None)                
                elif franka_task == FrankaTask.BinPush or franka_task == FrankaTask.PlanarPush: 
                    rewards = recompute_real_rwd(cfg, state, obs[:,0,:3], env.col_thresh) 
                else:
                    raise NotImplementedError()  
            else:
                assert(False, 'Dense rewards not available')
                      
        else:         
            if not cfg.dense_reward:
                rewards = np.array(pdata['env_infos/solved'][:cfg.episode_length], dtype=np.float32)#-1.
            else:
                rewards = np.array(pdata['env_infos/rwd_dense'][:cfg.episode_length], dtype=np.float32)
            
        episode = Episode.from_trajectory(cfg, obs, state, actions, rewards)
        episodes.append(episode)
    
    return episodes

def get_demos(cfg, env):
    demo_dir = git.Repo(hydra.utils.get_original_cwd(), search_parent_directories=True).working_tree_dir + "/modemv2"
    fps = glob.glob(str(Path(demo_dir) / "demonstrations" / f"{cfg.task}/*.pickle"))
    if len(fps) == 0:
        fps = glob.glob(str(Path(demo_dir) / "demonstrations" / f"{cfg.task}/*.h5"))
    episodes = []
    assert(cfg.task.startswith('franka-'))

    if 'BinPush' in cfg.task:
        franka_task = FrankaTask.BinPush
    elif 'HangPush' in cfg.task:
        franka_task = FrankaTask.HangPush
    elif 'PlanarPush' in cfg.task:
        franka_task = FrankaTask.PlanarPush
    elif 'BinPick' in cfg.task:
        franka_task = FrankaTask.BinPick
    elif 'BinReorient' in cfg.task:
        franka_task = FrankaTask.BinReorient
    else:
        raise NotImplementedError()
    exclude_fails = cfg.real_robot and (franka_task == FrankaTask.BinPush or franka_task == FrankaTask.PlanarPush)

    for fp in fps:   
        if len(episodes) >= cfg.demos:
            break        
        paths = Trace.load(fp)

        paths_episodes = trace2episodes(cfg=cfg,
                                        env=env,
                                        trace=paths,
                                        exclude_fails=exclude_fails,
                                        is_demo=True)
        for i in range(len(paths_episodes)):
            if len(episodes) >= cfg.demos:
                break
            episodes.append(paths_episodes[i])      
            if len(episodes) > 0 and len(episodes) % 1 == 0:
                print('Loaded demo {} of {}, reward {}'.format(len(episodes),cfg.demos, torch.sum(episodes[-1].reward).item()))
    
    assert(len(episodes)==cfg.demos)
    return episodes


def gather_paths_parallel(
    cfg,
    start_state: dict,
    act_list: np.ndarray,
    base_seed: int,
    paths_per_cpu: int,
    num_cpu: int = 1,
):
    """Parallel wrapper around the gather paths function."""

    if num_cpu == 1:
        input_dict = dict(
            cfg=cfg,
            start_state=start_state,
            act_list=act_list,
            base_seed=base_seed
        )
        return generate_paths(**input_dict)

    # do multiprocessing only if necessary
    input_dict_list = []

    for i in range(num_cpu):
        cpu_seed = base_seed + i * paths_per_cpu
        input_dict = dict(
            cfg=cfg,
            start_state=start_state,
            act_list=act_list[:,i*act_list.shape[1]//num_cpu:(i+1)*act_list.shape[1]//num_cpu],
            base_seed=cpu_seed
        )
        input_dict_list.append(input_dict)

    results = _try_multiprocess_mp(
        func=generate_paths,
        input_dict_list=input_dict_list,
        num_cpu=num_cpu,
        max_process_time=300,
        max_timeouts=4,
    )
    paths = []
    for result in results:
        for path in result:
            paths.append(path)

    return paths


def _try_multiprocess_mp(
    func: callable,
    input_dict_list: list,
    num_cpu: int = 1,
    max_process_time: int = 500,
    max_timeouts: int = 4,
    *args,
    **kwargs,
):
    """Run multiple copies of provided function in parallel using multiprocessing."""

    # Base case
    if max_timeouts == 0:
        return None

    pool = mp.Pool(processes=num_cpu, maxtasksperchild=None)
    parallel_runs = [
        pool.apply_async(func, kwds=input_dict) for input_dict in input_dict_list
    ]
    try:
        results = [p.get(timeout=max_process_time) for p in parallel_runs]
    except Exception as e:
        print(str(e))
        print("Timeout Error raised... Trying again")
        pool.close()
        #pool.terminate()
        pool.join()
        return _try_multiprocess_mp(
            func, input_dict_list, num_cpu, max_process_time, max_timeouts - 1
        )

    pool.close()
    #pool.terminate()
    pool.join()
    return results

def generate_paths(
    cfg,
    start_state: dict,
    act_list: np.ndarray,
    base_seed: int
) -> list:
    """Generates perturbed action sequences and then performs rollouts.

    Args:
        env:
            Environment, either as a gym env class, callable, or env_id string.
        start_state:
            Dictionary describing a single start state. All rollouts begin from this state.
        base_act:
            A numpy array of base actions to which we add noise to generate action sequences for rollouts.
        filter_coefs:
            We use these coefficients to generate colored for action perturbation
        base_seed:
            Seed for generating random actions and rollouts.
        num_paths:
            Number of paths to rollout.
        env_kwargs:
            Any kwargs needed to instantiate the environment. Used only if env is a callable.

    Returns:
        A list of paths that describe the resulting rollout. Each path is a list.
    """
    set_seed(base_seed)

    paths = do_env_rollout(cfg, start_state, act_list)
    return paths

def set_seed(seed=None):
    """
    Set all seeds to make results reproducible
    :param seed: an integer to your choosing (default: None)
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

def do_policy_rollout(
    cfg,
    start_states,
    agent
):
    e = make_env(cfg)
    

    H = cfg.episode_length
    N = len(start_states)  # number of rollout trajectories (per process)
    ep_states = []
    ep_actions = []
    for i in range(N):
        e.reset()
      
        if cfg.task.startswith('adroit-') or cfg.task.startswith('franka-'):
            #e.unwrapped.set_env_state(start_states[i])
            e.set_env_state(start_states[i])
        else:
            e.unwrapped.physics.set_state(start_states[i])
        obs = e.observation
        state = e.state

        observations = [obs]
        actions = []
        rewards = []
        env_infos = []
        states = [state]

        for k in range(H):
            action = agent.act(torch.tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0), 
                               torch.tensor(e.state, dtype=torch.float32, device=agent.device).unsqueeze(0),
                               0.0).squeeze(0).cpu().numpy()
            obs, r, d, ifo = e.step(action)
            state = e.state

            actions.append(e.base_env().last_eef_cmd)
            observations.append(obs)
            rewards.append(r)
            env_infos.append(ifo)
            states.append(state)
        
        ep_actions.append(np.stack(actions, axis=0))
        ep_states.append(np.stack(states, axis=0))

    return ep_states, ep_actions    

def do_env_rollout(
    cfg,
    start_state: dict,
    act_list: np.ndarray
) -> list:
    """Rollout action sequence in env from provided initial states.

    Instantiates requested environment. Sets environment to provided initial states.
    Then rollouts out provided action sequence. Returns result in paths format (list of dicts).

    Args:
        env:
            Environment, either as a gym env class, callable, or env_id string.
        start_state:
            Dictionary describing a single start state. All rollouts begin from this state.
        act_list:
            List of numpy arrays containing actions that will be rolled out in open loop.
        env_kwargs:
            Any kwargs needed to instantiate the environment. Used only if env is a callable.

    Returns:
        A list of paths that describe the resulting rollout. Each path is a list.
    """

    # Not all low-level envs are picklable. For generalizable behavior,
    # we have the option to instantiate the environment within the rollout function.
    cfg_copy = deepcopy(cfg)
    cfg_copy.dense_reward = True
    e = make_env(cfg_copy)

    e.unwrapped.real_step = False  # indicates simulation for purpose of trajectory optimization
    paths = []
    H = act_list.shape[0]  # horizon
    N = act_list.shape[1]  # number of rollout trajectories (per process)
    for i in range(N):
        e.reset()
        if cfg.task.startswith('adroit-') or cfg.task.startswith('franka-'):
            #e.unwrapped.set_env_state(start_state)
            e.set_env_state(start_state)
        else:
            e.unwrapped.physics.set_state(start_state)
        obs = []
        act = []
        rewards = []
        env_infos = []
        states = []

        for k in range(H):
            s, r, d, ifo = e.step(act_list[k,i])

            act.append(act_list[k,i])
            obs.append(s)
            rewards.append(r)
            env_infos.append(ifo)            
            if cfg.task.startswith('adroit-') or cfg.task.startswith('franka-'):
                states.append(e.unwrapped.get_env_state())
            else:
                states.append(e.unwrapped.physics.get_state())
            
        path = dict(
            observations=np.array(obs),
            actions=np.array(act),
            rewards=np.array(rewards),
            env_infos=stack_tensor_dict_list(env_infos),
            states=states,
        )
        paths.append(path)

    return paths

def stack_tensor_dict_list(tensor_dict_list):
    """
    Stack a list of dictionaries of {tensors or dictionary of tensors}.
    :param tensor_dict_list: a list of dictionaries of {tensors or dictionary of tensors}.
    :return: a dictionary of {stacked tensors or dictionary of stacked tensors}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = stack_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret

def stack_tensor_list(tensor_list):
    return np.array(tensor_list)

class ReplayBuffer(object):
    """
    Storage and sampling functionality for training MoDem.
    Uses prioritized experience replay by default.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.capacity = 2 * cfg.train_steps + 1
        obs_shape = (cfg.obs_shape[0], cfg.obs_shape[1]//cfg.frame_stack, *cfg.obs_shape[-2:])
        self._state_dim = cfg.state_dim
        self._obs = torch.empty(
            (self.capacity + 1, *obs_shape), dtype=torch.uint8, device=self.device
        )
        self._last_obs = torch.empty(
            (self.capacity // cfg.episode_length, *cfg.obs_shape),
            dtype=torch.uint8,
            device=self.device,
        )
        self._action = torch.empty(
            (self.capacity, cfg.action_dim), dtype=torch.float32, device=self.device
        )
        self._reward = torch.empty(
            (self.capacity,), dtype=torch.float32, device=self.device
        )
        self._state = torch.empty(
            (self.capacity, self._state_dim), dtype=torch.float32, device=self.device
        )
        self._last_state = torch.empty(
            (self.capacity // cfg.episode_length, self._state_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self._priorities = torch.ones(
            (self.capacity,), dtype=torch.float32, device=self.device
        )
        self._eps = 1e-6
        self.idx = 0
        self.full = False

    def __len__(self):
        return self.idx if not self.full else self.capacity

    def __add__(self, episode: Episode):
        self.add(episode)
        return self

    def add(self, episode: Episode):
        assert(len(episode.obs.shape)==5)

        if self.idx + self.cfg.episode_length > self.capacity:
            self._priorities[self.idx:] = 0.0
            self.idx = 0
            self.full = True

        obs = episode.obs[:-1, :, -(self.cfg.obs_shape[1]//self.cfg.frame_stack):]
        if episode.obs.shape[2] == (self.cfg.obs_shape[1]//self.cfg.frame_stack):
            last_obs = torch.split(episode.obs[-self.cfg.frame_stack:],1,dim=0)
            last_obs = torch.cat(last_obs, dim=2)
            last_obs = torch.squeeze(last_obs, dim=0)
        else:
            assert(episode.obs.shape[2] == self.cfg.obs_shape[1])
            last_obs = episode.obs[-1]
        self._obs[self.idx : self.idx + self.cfg.episode_length] = obs
        self._last_obs[self.idx // self.cfg.episode_length] = last_obs
        self._action[self.idx : self.idx + self.cfg.episode_length] = episode.action
        self._reward[self.idx : self.idx + self.cfg.episode_length] = episode.reward
        states = torch.tensor(episode.state, dtype=torch.float32)
        self._state[
            self.idx : self.idx + self.cfg.episode_length, : self._state_dim
        ] = states[:-1]
        self._last_state[
            self.idx // self.cfg.episode_length, : self._state_dim
        ] = states[-1]
        max_priority = (
            1.0
            if self.idx == 0
            else self._priorities[: self.idx].max().to(self.device).item()
        )
        mask = (
            torch.arange(self.cfg.episode_length)
            >= self.cfg.episode_length - self.cfg.horizon
        )
        new_priorities = torch.full(
            (self.cfg.episode_length,), max_priority, device=self.device
        )
        new_priorities[mask] = 0
        self._priorities[self.idx : self.idx + self.cfg.episode_length] = new_priorities

        self.idx += self.cfg.episode_length

    def update_priorities(self, idxs, priorities):
        self._priorities[idxs] = priorities.squeeze(1).to(self.device) + self._eps

    def _get_obs(self, arr, idxs):
        obs = torch.empty(
            (self.cfg.batch_size, self.cfg.obs_shape[0], self.cfg.obs_shape[1], *arr.shape[-2:]),
            dtype=arr.dtype,
            device=torch.device("cuda"),
        )
        obs[:, :,-(self.cfg.obs_shape[1]//self.cfg.frame_stack):] = arr[idxs].cuda(non_blocking=True)
        _idxs = idxs.clone()
        mask = torch.ones_like(_idxs, dtype=torch.bool)
        for i in range(1, self.cfg.frame_stack):
            mask[_idxs % self.cfg.episode_length == 0] = False
            _idxs[mask] -= 1
            obs[:,:, -(i + 1) * (self.cfg.obs_shape[1]//self.cfg.frame_stack) : -i * (self.cfg.obs_shape[1]//self.cfg.frame_stack)] = arr[_idxs].cuda(non_blocking=True)
        return obs.float()

    def sample(self):
        if not self.full:
            probs = self._priorities[: self.idx] ** self.cfg.per_alpha
        else:
            probs = self._priorities[:] ** self.cfg.per_alpha
        probs /= probs.sum()
        total = len(probs)
        idxs = torch.from_numpy(
            np.random.choice(
                total, self.cfg.batch_size, p=probs.cpu().numpy(), replace=True
            )
        ).to(self.device)
        weights = (total * probs[idxs]) ** (-self.cfg.per_beta)
        weights /= weights.max()
        obs = (
            self._get_obs(self._obs, idxs)
            if self.cfg.frame_stack > 1
            else self._obs[idxs].cuda(non_blocking=True)
        )
        next_obs_shape = (self.cfg.obs_shape[0], self.cfg.obs_shape[1], *self._last_obs.shape[-2:])
        next_obs = torch.empty(
            (self.cfg.horizon + 1, self.cfg.batch_size, *next_obs_shape),
            dtype=obs.dtype,
            device=obs.device,
        )
        action = torch.empty(
            (self.cfg.horizon + 1, self.cfg.batch_size, *self._action.shape[1:]),
            dtype=torch.float32,
            device=self.device,
        )
        reward = torch.empty(
            (self.cfg.horizon + 1, self.cfg.batch_size),
            dtype=torch.float32,
            device=self.device,
        )
        state = self._state[idxs, : self._state_dim]
        next_state = torch.empty(
            (self.cfg.horizon + 1, self.cfg.batch_size, *state.shape[1:]),
            dtype=state.dtype,
            device=state.device,
        )
        for t in range(self.cfg.horizon + 1):
            _idxs = idxs + t
            next_obs[t] = (
                self._get_obs(self._obs, _idxs + 1)
                if self.cfg.frame_stack > 1
                else self._obs[_idxs + 1].cuda(non_blocking=True)
            )
            action[t] = self._action[_idxs]
            reward[t] = self._reward[_idxs]
            next_state[t] = self._state[_idxs + 1, : self._state_dim]

        mask = (_idxs + 1) % self.cfg.episode_length == 0
        next_obs[-1, mask] = (
            self._last_obs[_idxs[mask] // self.cfg.episode_length]
            .to(next_obs.device, non_blocking=True)
            .float()
        )
        state = state.cuda(non_blocking=True)
        next_state[-1, mask] = (
            self._last_state[_idxs[mask] // self.cfg.episode_length, : self._state_dim]
            .to(next_state.device)
            .float()
        )
        next_state = next_state.cuda(non_blocking=True)
        next_obs = next_obs.cuda(non_blocking=True)
        action = action.cuda(non_blocking=True)
        reward = reward.unsqueeze(2).cuda(non_blocking=True)
        idxs = idxs.cuda(non_blocking=True)
        weights = weights.cuda(non_blocking=True)

        return obs, next_obs, action, reward, state, next_state, idxs, weights


def linear_schedule(schdl, step):
    """Outputs values following a linear decay schedule"""
    try:
        return float(schdl)
    except ValueError:
        try:
            match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
            if match:
                init, final, duration = [float(g) for g in match.groups()]
                mix = np.clip(step / duration, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final
        except ValueError:
            match = re.match(r"linear\((.+),(.+),(.+),(.+)\)", schdl)
            if match:
                init, final, start, end = [float(g) for g in match.groups()]
                assert(end>start)
                mix = np.clip((step-start)/(end-start),0.0,1.0)
                return (1.0-mix)*init + mix*final
    raise NotImplementedError(schdl)
