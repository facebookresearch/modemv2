# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
import torch
import numpy as np
import gym
from gym.wrappers import TimeLimit
import matplotlib.pyplot as plt
from enum import Enum 
from robohive.envs.arms.bin_pick_v0 import BinPickPolicy
from robohive.utils.quat_math import mat2quat
import hydra
import time
from pathlib import Path
import git
import os

class FrankaTask(Enum):
    BinPick=1
    BinPush=2
    HangPush=3
    PlanarPush=4
    BinReorient=5

class FrankaWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self._num_frames = cfg.get("frame_stack", 1)
        self._frames = deque([], maxlen=self._num_frames)
        img_size = max(cfg.img_size, 10)
        self.camera_names = cfg.get("camera_views", ["view_1"])
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(len(self.camera_names), self._num_frames * (3 if cfg.img_size <= 0 else 4), img_size, img_size),
            dtype=np.uint8,
        )
        
        if 'BinPick' in cfg.task:
            self.franka_task = FrankaTask.BinPick
        elif 'BinPush' in cfg.task:
            self.franka_task = FrankaTask.BinPush
        elif 'HangPush' in cfg.task:
            self.franka_task = FrankaTask.HangPush
        elif 'PlanarPush' in cfg.task:
            self.franka_task = FrankaTask.PlanarPush
        elif 'BinReorient' in cfg.task:
            self.franka_task = FrankaTask.BinReorient
        else:
            raise NotImplementedError()

        if self.franka_task == FrankaTask.BinPick:
            act_low = -np.ones(6)
            act_high = np.ones(6)        
        elif self.franka_task == FrankaTask.PlanarPush:
            act_low = -np.ones(5)
            act_high = np.ones(5)
        elif self.franka_task == FrankaTask.BinReorient:
            act_low = -np.ones(15)
            act_high = np.ones(15)
        else:
            act_low = -np.ones(3)
            act_high = np.ones(3)
        self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)
        if self.env.robot.is_hardware:
            if cfg.task.startswith('franka-FrankaBinPick'):

                self.consec_train_fails = 0
                self.RESET_PI_THRESH = 1
                self.reset_pi = BinPickPolicy(env=self.env, seed=self.cfg.seed, is_reset_policy=True)
            elif cfg.task.startswith('franka-FrankaPlanarPush') or cfg.task.startswith('franka-FrankaBinPush'):
                from modemv2.utils.color_threshold import ColorThreshold
                self.col_thresh = ColorThreshold(cam_name=self.camera_names[0], 
                                                left_crop=self.cfg.success_mask_left, 
                                                right_crop=self.cfg.success_mask_right, 
                                                top_crop=self.cfg.success_mask_top, 
                                                bottom_crop=self.cfg.success_mask_bottom, 
                                                thresh_val=self.cfg.success_thresh, 
                                                target_uv=np.array(self.cfg.success_uv),
                                                render=False)

    @property
    def state(self):
        return self._state_obs.astype(np.float32)

    @property
    def observation(self):
        return self._stacked_obs()

    def _get_state_obs(self, obs):

        qp = self.env.sim.data.qpos.ravel()
        qv = self.env.sim.data.qvel.ravel()
        grasp_pos = self.env.sim.data.site_xpos[self.env.grasp_sid].ravel()
        grasp_rot = mat2quat(self.env.sim.data.site_xmat[self.env.grasp_sid].reshape(3,3).transpose())
        
        
        if self.franka_task == FrankaTask.BinReorient:
            obj_err = self.env.sim.data.site_xpos[self.env.object_sid]-self.env.sim.data.site_xpos[self.env.hand_sid]
            tar_err = np.array([np.abs(self.env.sim.data.site_xmat[self.env.object_sid][-1] - 1.0)],dtype=float)
        else:
            obj_err = self.env.sim.data.site_xpos[self.env.object_sid]-self.env.sim.data.site_xpos[self.env.grasp_sid]
            tar_err = self.env.sim.data.site_xpos[self.env.target_sid]-self.env.sim.data.site_xpos[self.env.object_sid]

        if self.cfg.img_size <= 0:
            manual = np.concatenate([qp,qv,grasp_pos,grasp_rot,obj_err,tar_err])
            assert(np.isclose(obs[:manual.shape[0]], manual).all())
        else:
            if self.franka_task == FrankaTask.BinReorient:
                manual = np.concatenate([qp[:17],
                                         qv[:17],
                                         grasp_pos,
                                         grasp_rot])
                assert(np.isclose(obs[:17], qp[:17]).all())
                assert(np.isclose(obs[qp.shape[0]:qp.shape[0]+17], qv[:17]).all())
                assert(np.isclose(obs[qp.shape[0]+qv.shape[0]:qp.shape[0]+qv.shape[0]+3], grasp_pos).all())
                assert(np.isclose(obs[qp.shape[0]+qv.shape[0]+grasp_pos.shape[0]:qp.shape[0]+qv.shape[0]+grasp_pos.shape[0]+4],grasp_rot).all())                
            elif self.cfg.real_robot:
                manual = np.concatenate([qp[:9], qv[:9], grasp_pos,grasp_rot])
                assert(np.isclose(obs[:9], qp[:9]).all())
                assert(np.isclose(obs[qp.shape[0]:qp.shape[0]+9], qv[:9]).all())
                assert(np.isclose(obs[qp.shape[0]+qv.shape[0]:qp.shape[0]+qv.shape[0]+3], grasp_pos).all())
                assert(np.isclose(obs[qp.shape[0]+qv.shape[0]+grasp_pos.shape[0]:qp.shape[0]+qv.shape[0]+grasp_pos.shape[0]+4],grasp_rot).all())
            else:
                manual = np.concatenate([qp[:8], qv[:8], grasp_pos,grasp_rot])
                assert(np.isclose(obs[:8], qp[:8]).all())
                assert(np.isclose(obs[qp.shape[0]:qp.shape[0]+8], qv[:8]).all())
                assert(np.isclose(obs[qp.shape[0]+qv.shape[0]:qp.shape[0]+qv.shape[0]+3], grasp_pos).all())
                assert(np.isclose(obs[qp.shape[0]+qv.shape[0]+grasp_pos.shape[0]:qp.shape[0]+qv.shape[0]+grasp_pos.shape[0]+4],grasp_rot).all())

        return manual


    def _get_pixel_obs(self):
        if self.cfg.img_size <= 0:
            return np.zeros((self.observation_space.shape[0],
                             self.observation_space.shape[1]//self._num_frames,
                             self.observation_space.shape[2],
                             self.observation_space.shape[3]), dtype=np.uint8)

        img_views = []
        vis_obs_dict = self.env.visual_dict
        for i,camera_name in enumerate(self.camera_names):

            rgb_key = 'rgb:'+camera_name+':'+str(self.cfg.img_size)+'x'+str(self.cfg.img_size)+':2d'
            depth_key = 'd:'+camera_name+':'+str(self.cfg.img_size)+'x'+str(self.cfg.img_size)+':2d'
            if rgb_key not in vis_obs_dict or depth_key not in vis_obs_dict:
                rgb_key = 'rgb:'+camera_name+':240x424:2d'
                depth_key = 'd:'+camera_name+':240x424:2d'

            assert(rgb_key in vis_obs_dict and depth_key in vis_obs_dict)  

            rgb_img = vis_obs_dict[rgb_key].squeeze().transpose(2,0,1) # cxhxw
            rgb_img = rgb_img[:,
                              self.cfg.top_crops[i]:self.cfg.top_crops[i]+self.cfg.img_size,
                              self.cfg.left_crops[i]:self.cfg.left_crops[i]+self.cfg.img_size]
            
            if self.cfg.real_robot:
                depth_img = np.array(np.clip(vis_obs_dict[depth_key],0,255), dtype=np.uint8) # 1xhxw
            else:
                depth_img = vis_obs_dict[depth_key] # 1xhxw
            depth_img = depth_img[:,
                                  self.cfg.top_crops[i]:self.cfg.top_crops[i]+self.cfg.img_size,
                                  self.cfg.left_crops[i]:self.cfg.left_crops[i]+self.cfg.img_size]
            img_views.append(np.concatenate((rgb_img, depth_img), axis=0))
        return np.stack(img_views, axis=0)     

    def _stacked_obs(self):
        assert len(self._frames) == self._num_frames
        return np.concatenate(list(self._frames), axis=1)

    def reset(self):

        if (self.cfg.real_robot and (self.cfg.task.startswith('franka-FrankaPlanarPush') or 
                                 self.cfg.task.startswith('franka-FrankaBinPush')) 
            and len(self._frames)> 0):
            # Move to pre-reset
            print('moving to preset')
            if self.cfg.task.startswith('franka-FrankaPlanarPush'):
                preset_eef = np.array([0.5, 0.4, 1.25,1.0,np.arcsin(-0.74)])
            elif self.cfg.task.startswith('franka-FrankaBinPush'):
                preset_eef = np.array([0.5,0.3,1.25])
            else:
                raise NotImplementedError()
            preset_eef[:3] = 2*(((preset_eef[:3]-self.env.unwrapped.pos_limits['eef_low'][:3])/(np.abs(self.env.unwrapped.pos_limits['eef_high'][:3]-self.env.unwrapped.pos_limits['eef_low'][:3])+1e-8))-0.5)
            preset_step = 0
            while(preset_step < 50):
                self.step(preset_eef)
                preset_step += 1
            print('at preset')

            if self.cfg.task.startswith('franka-FrankaBinPush'):
                time.sleep(5) # Wait for ball to stop moving

        obs = self.env.reset()
        obs, rwd, done, env_info = self.env.unwrapped.forward(update_exteroception=True)
        self._state_obs = self._get_state_obs(obs)
        obs = self._get_pixel_obs()
        for _ in range(self._num_frames):
            self._frames.append(obs)
        return self._stacked_obs()

    def set_env_state(self, env_state):
        self.env.unwrapped.set_env_state(env_state)
        env_obs = self.env.get_obs()

        self._state_obs = self._get_state_obs(env_obs)
        obs = self._get_pixel_obs()    
        for _ in range(self._num_frames):
            self._frames.append(obs)
        return self._stacked_obs()            

    def get_base_env_action(self, action):
        assert(len(action.shape)==1)
        if self.franka_task == FrankaTask.BinPick:
            assert(action.shape[0] == 6)
            aug_action = np.zeros(7, dtype=action.dtype)
            aug_action[:3] = action[:3]
            aug_action[3] = 1.0+0.1*np.random.normal()
            aug_action[4] = -1.0+0.1*np.random.normal()
            aug_action[5] = np.arctan2(action[4], action[3])
            aug_action[5] = 2*(((aug_action[5] - self.env.pos_limits['eef_low'][5]) / (self.env.pos_limits['eef_high'][5] - self.env.pos_limits['eef_low'][5])) - 0.5)
            aug_action[6] = 2*(int(action[5]>0.0)-0.5)
        elif self.franka_task == FrankaTask.BinReorient:
            assert(action.shape[0] == 15)
            aug_action = np.zeros(16, dtype=action.dtype)
            aug_action[:3] = action[:3]
            aug_action[3] = 1.0+0.05*np.random.normal()
            aug_action[4] = -1.0+0.05*np.random.normal()
            aug_action[5] = np.arctan2(action[4], action[3])
            aug_action[5] = 2*(((aug_action[5] - self.env.pos_limits['eef_low'][5]) / (self.env.pos_limits['eef_high'][5] - self.env.pos_limits['eef_low'][5])) - 0.5)
            aug_action[6:] = action[5:]
        else:
            aug_action = np.zeros(7, dtype=action.dtype)
            aug_action[:3] = action[:3]
            # Note that unset dimensions of aug_action will be clipped to a constant value
            # as specified in env/arms/franka/__init__.py
            if self.franka_task == FrankaTask.PlanarPush:
                assert(action.shape[0] == 5)
                aug_action[5] = np.arctan2(action[4], action[3])
                aug_action[5] = 2*(((aug_action[5] - self.env.pos_limits['eef_low'][5]) / (self.env.pos_limits['eef_high'][5] - self.env.pos_limits['eef_low'][5])) - 0.5)
            else:
                assert(action.shape[0] == 3)
        return aug_action

    def get_trace_dict(self, action):
        aug_action = self.get_base_env_action(action)
        t, cur_obs = self.env.obsdict2obsvec(self.env.obs_dict, self.env.obs_keys)
        cur_env_info = self.env.get_env_infos()
        trace_dict = dict(time=self.env.time,
                          observations=cur_obs,
                          actions=aug_action.copy(),
                          rewards=cur_env_info['rwd_'+self.env.rwd_mode],
                          env_infos=cur_env_info,
                          done=bool(cur_env_info['done']))
        return trace_dict

    def step(self, action):
        reward = 0

        aug_action = self.get_base_env_action(action)
        
        for _ in range(self.cfg.action_repeat):
            obs, r, _, info = self.env.unwrapped.step(aug_action, update_exteroception=True)
            reward += r

        self._state_obs = self._get_state_obs(obs)
        obs = self._get_pixel_obs()
        self._frames.append(obs)
        info["success"] = info["solved"]
        if not self.cfg.dense_reward:
            reward = float(info["success"]) - self.cfg.torque_penalty*np.linalg.norm(self.env.unwrapped.sim.data.qfrc_actuator.copy()[:7])
        if self.cfg.real_robot:
            reward = 0.0#-1.0
        return self._stacked_obs(), reward, False, info

    def render(self, mode="rgb_array", width=None, height=None, camera_id=None):
        if self.cfg.real_robot:
            return self._frames[-1][0,:3].transpose(1,2,0)
        else:
            img = self.env.env.sim.renderer.render_offscreen(width=width, 
                                                             height=height, 
                                                             depth=False, 
                                                             camera_id=self.camera_names[0], 
                                                             device_id=self.env.env.device_id)
            return img

    def observation_spec(self):
        return self.observation_space

    def action_spec(self):
        return self.action_space

    def __getattr__(self, name):
        return getattr(self._env, name)

    def base_env(self):
        return self.env

    def post_process_task(self, obs, states, eval_mode=True):
        from modemv2.utils.move_utils import update_grasps, check_grasp_success, check_reorient_success
        assert(self.cfg.real_robot)
        logging_dir = git.Repo(hydra.utils.get_original_cwd(), search_parent_directories=True).working_tree_dir + "/modemv2"
        success = False
        rewards = None
        retry_episode = False
        if self.cfg.task.startswith('franka-FrankaBinPick'):
            _, latest_img, success, _, _, grasped = check_grasp_success(env=self.env, obs=None, force_img=eval_mode or (self.consec_train_fails>=self.RESET_PI_THRESH-1))
            
            if (success and not grasped) or (not success and grasped):
                print('Result unclear, retry episode')
                retry_episode = True 
            
            if success:
                rewards = recompute_real_rwd(self.cfg, states)
                self.consec_train_fails = 0
            else:
                if not eval_mode:
                    self.consec_train_fails += 1
                
                if eval_mode or self.consec_train_fails >= self.RESET_PI_THRESH:
                    assert(latest_img is not None)
                    grasp_centers = []
                    real_obj_pos = np.zeros(3)
   
                    reset_success = False                
                    while(not reset_success):
                            
                        while(len(grasp_centers) <= 0):
                            #grasp_centers, filtered_boxes, img_masked = update_grasps(img=latest_img, 
                            #                                                        out_dir=logging_dir+'/debug')
                            grasp_centers, filtered_boxes, img_masked = update_grasps(img=latest_img,
                                                                                    out_dir=logging_dir+'/debug',
                                                                                    min_pixels=100,
                                                                                    luv_thresh=True,
                                                                                    limit_yaw=True)                                                                                     
                            if len(grasp_centers) <= 0:
                                input('Block not detected, enter to continue')
                                print('Taking new block image')
                                retry_episode = True
                                _, latest_img, _, _, _, _ = check_grasp_success(env=self.env, obs=None, force_img=True)
                        
                        real_obj_pos = np.array([grasp_centers[-1][0],grasp_centers[-1][1], 0.91])
                        real_obj_yaw = grasp_centers[-1][2]
                        print('Real obj pos {}'.format(real_obj_pos))

                        print('Executing reset policy ({})'.format('eval' if eval_mode else 'train'))
                        self.reset_pi.set_real_obj_pos(real_obj_pos)
                        self.reset_pi.set_real_tar_pos(np.array([0.5, 0.0, 1.1]))
                        self.reset_pi.set_real_yaw(real_obj_yaw)
                        self.env.examine_policy_new(policy=self.reset_pi,
                                                    horizon=self.env.spec.max_episode_steps,
                                                    num_episodes=1,
                                                    frame_size=(640,480),
                                                    mode='evaluation',
                                                    output_dir=None,
                                                    filename=None,
                                                    camera_name=None,
                                                    render='none')  
                        _, latest_img, reset_success, _, _, _ = check_grasp_success(self.env, obs=None, force_img=True)
                        grasp_centers = []
                    print('Object has been reset')
                    if not eval_mode:
                        self.consec_train_fails = 0

        elif self.cfg.task.startswith('franka-FrankaPlanarPush') or self.cfg.task.startswith('franka-FrankaBinPush'):
            rewards = recompute_real_rwd(self.cfg, states, obs, self.col_thresh)
            success = torch.sum(rewards).item() >= 4.999
        elif self.cfg.task.startswith('franka-FrankaBinReorient'):
            output_dir = Path(logging_dir) / "logs" / self.cfg.task / self.cfg.exp_name / str(self.cfg.seed)
            output_dir = str(output_dir)
            reorient_success, reset_img = check_reorient_success(self.env.unwrapped, obs=None,
                                                                 out_dir=output_dir, 
                                                                 drop_goal_x=self.cfg.reorient_drop_goal_x,
                                                                 drop_goal_y=self.cfg.reorient_drop_goal_y,
                                                                 success_thresh=self.cfg.depth_success_thresh, 
                                                                 pickup_height=self.cfg.reorient_pickup_height,
                                                                 knock_height=self.cfg.reorient_knock_height)
            success = reorient_success
            rewards = recompute_real_rwd(self.cfg, states)            
        else:
            raise NotImplementedError()
        
        return success, rewards, retry_episode

def recompute_real_rwd(cfg, states, obs=None, col_thresh=None):
    assert(states.shape[1] == 25 or 
        (cfg.task == 'franka-FrankaBinPickRealRP05_v2d' and states.shape[1] == 23) or
        (cfg.task.startswith('franka-FrankaBinReorientReal') and states.shape[1] == 41))
    rewards = torch.zeros(
        (cfg.episode_length,), dtype=torch.float32, device=states.device
    )#-1.0
    if cfg.task.startswith('franka-FrankaBinPick'):
        for i in reversed(range(cfg.episode_length)):
            if states[i+1, -5] > 1.0:
                rewards[i] = 1.0#0.0
            else:
                break
    elif cfg.task.startswith('franka-FrankaPlanarPush') or cfg.task.startswith('franka-FrankaBinPush'):
        assert(len(obs.shape)==4)
        assert(obs.shape[0]==cfg.episode_length+1)
        assert(obs.shape[1]==3)
        assert(obs.shape[2]==cfg.img_size and obs.shape[3]==cfg.img_size)
        imgs = obs.transpose(0,2,3,1)
        for t in range(1,obs.shape[0]):
            img_success, luv_angle = col_thresh.detect_success(imgs[t])
            if img_success:
                rewards[t-1] = 1.0
        print('Ep reward: {}'.format(torch.sum(rewards).item()))
    elif cfg.task.startswith('franka-FrankaBinReorient'):
        rewards[-5:] = 1.0
        for i in reversed(range(cfg.episode_length)):
            #if states[i+1, -5] > 1.0 and states[i+1, 12] > 1.45:            

            if (states[i+1, 8] > 0.5 and states[i+1, 8] < 1.0 and
                states[i+1, 12] > 0.75 and states[i+1, 12] < 1.25 and
                states[i+1, 15] > 0.5 and states[i+1, 15] < 1.0 and
                states[i+1, -5] > 1.05 and states[i+1, -5] < 1.1):
                rewards[i] = 1.0
            elif states[i+1, -5] <= 1.0:
                break    
            #if (states[i+1, 8] < 1.0 and 
            #    states[i+1, 12] < 1.0 and 
            #    states[i+1, 15] < 1.0 and 
            #   states[i+1, -5] > 1.0):
            #    rewards[i] = 1.0
            #elif states[i+1, -5] <= 1.0:
            #    break            
    else:
        raise NotImplementedError()

    return rewards

def make_franka_env(cfg):
    env_id = cfg.task.split("-", 1)[-1] + "-v0"
    reward_mode = 'dense' if cfg.dense_reward else 'sparse'
    env = gym.make(env_id, **{'reward_mode': reward_mode, 'is_hardware': cfg.real_robot, 'torque_scale': cfg.torque_scale})
    env = FrankaWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=cfg.episode_length)

    if cfg.real_robot and (cfg.task.startswith('franka-FrankaPlanarPush') or cfg.task.startswith('franka-FrankaBinPush')):
        env.reset()

    env.reset()
    cfg.state_dim = env.state.shape[0]
    return env
