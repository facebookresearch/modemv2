# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

warnings.filterwarnings("ignore")
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
import torch
import numpy as np
import gym

gym.logger.set_level(40)
import time
from pathlib import Path
from cfg_parse import parse_cfg
from env import make_env, set_seed
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, get_demos, ReplayBuffer, do_policy_rollout, trace2episodes
from termcolor import colored
from copy import deepcopy
import logger
import hydra
from robohive.logger.grouped_datasets import Trace

torch.backends.cudnn.benchmark = True
import algorithm.helper as h
import git

def evaluate(env, agent, cfg, step, env_step, video, traj_plot=None, q_plot=None, policy_rollout=False):
    """Evaluate a trained agent and optionally save a video."""

    episode_states = []
    episode_actions = []
    episode_rewards = []
    episode_successes = []
    episode_start_states = []

    joint_configs = []
    joint_vels = []
    joint_accels = []
    joint_forces = []
    joint_jerks = []
    cart_pos = []
    cart_vels = []
    cart_accels = []
    cart_jerks = []
    velocimeter_id = env.unwrapped.sim.model.sensor_adr[env.unwrapped.sim.model.sensor_name2id('palm_velocimeter')]
    accelerometer_id = env.unwrapped.sim.model.sensor_adr[env.unwrapped.sim.model.sensor_name2id('palm_accelerometer')]
    contact_forces = []

    ep_q_stats = []
    ep_q_success = []
    ep_count = 0
    while ep_count < cfg.eval_episodes:
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        episode_start_states.append(env.unwrapped.get_env_state())
        states = [torch.tensor(env.state, dtype=torch.float32, device=agent.device)]
        obs_unstacked = [obs[0,-4:-1]]
        actions = []
        q_stats = []
        if video:
            if cfg.real_robot:
                video.init(env, enabled=(ep_count < 5))
            else:
                video.init(env, enabled=(ep_count == 0))
        success_count = 0.0

        start_time = time.time()
        while not done:              
            action, q_stat = agent.plan(obs, env.state, 
                                        eval_mode=True, 
                                        step=(0 if policy_rollout else step), 
                                        t=t)   
            
            obs, reward, done, info = env.step(action.cpu().numpy())
            success_count += float(info["success"])
            joint_configs.append(env.unwrapped.sim.data.qpos.copy())
            joint_vels.append(env.unwrapped.sim.data.qvel.copy())
            joint_accels.append(env.unwrapped.sim.data.qacc.copy())
            joint_forces.append(env.unwrapped.sim.data.qfrc_actuator.copy())
            cart_pos.append(env.unwrapped.sim.data.site_xpos[env.unwrapped.grasp_sid].copy())
            cart_vels.append(env.unwrapped.sim.data.sensordata[velocimeter_id:velocimeter_id+3])
            cart_accels.append(env.unwrapped.sim.data.sensordata[accelerometer_id:accelerometer_id+3])

            if t > 0:
                joint_jerks.append(joint_accels[-1]-joint_accels[-2])
                cart_jerks.append(cart_accels[-1]-cart_accels[-2])

            if cfg.task.startswith('franka-FrankaBinReorient'):
                contact_force = (env.unwrapped.sim.data.get_sensor('touch_sensor_tf')+
                                 env.unwrapped.sim.data.get_sensor('touch_sensor_ff')+
                                 env.unwrapped.sim.data.get_sensor('touch_sensor_pf'))
            else:
                contact_force = env.unwrapped.sim.data.get_sensor('touch_sensor_left')+env.unwrapped.sim.data.get_sensor('touch_sensor_right')
            if contact_force > 1e-5:
                contact_forces.append(contact_force)

            states.append(torch.tensor(env.state, dtype=torch.float32, device=agent.device))
            obs_unstacked.append(obs[0,-4:-1])
            actions.append(env.base_env().last_eef_cmd)
            ep_reward += reward
            if q_stat is not None:
                q_stats.append(q_stat)
            if video:
                video.record(env)
            t += 1
            
        ep_success = success_count>=5
        if cfg.real_robot:
            print('Episode length: {}'.format(time.time()-start_time))
            states = torch.stack(states, dim=0)
            obs_unstacked = np.stack(obs_unstacked, axis=0)
            task_success, new_rewards, retry_episode = env.post_process_task(obs_unstacked, states, eval_mode=True)
            
            if retry_episode:
                print('Episode result unclear, retry')
                continue
            
            if task_success:
                ep_reward = torch.sum(new_rewards).item()

            ep_success = 1 if task_success else 0

        episode_states.append(np.stack([state.cpu().numpy() for state in states],axis=0))
        episode_actions.append(np.stack(actions, axis=0))
        episode_rewards.append(ep_reward)
        episode_successes.append(ep_success)
        if len(q_stats) > 0:
            ep_q_stats.append(q_stats)
            ep_q_success.append(ep_success)
        if video:
            video.save(env_step)
        ep_count += 1

    assert(len(episode_states)) == cfg.eval_episodes
    assert(len(episode_actions)) == cfg.eval_episodes
    assert(len(episode_rewards)) == cfg.eval_episodes
    assert(len(episode_successes)) == cfg.eval_episodes

    ep_uncertainty_weight = 0.0
    uncertainty_count = 0
    if len(ep_q_stats) > 0 and q_plot is not None:
        print('LOGGING Q VALS')
        q_plot.log_q_vals(env_step, ep_q_stats, ep_q_success)
        for i in range(len(ep_q_stats)):
            for j in range(len(ep_q_stats[i])):
                ep_uncertainty_weight += ep_q_stats[i][j]['uncertainty_weight']
                uncertainty_count += 1
        ep_uncertainty_weight = ep_uncertainty_weight/uncertainty_count
    if traj_plot:
        if not cfg.real_robot:
            rollout_states, rollout_actions = do_policy_rollout(cfg, episode_start_states, agent)
        else:
            rollout_states, rollout_actions = episode_states, episode_actions
        traj_plot.save_traj(rollout_states, rollout_actions, 
                            episode_states, episode_actions, 
                            env_step)
    safety_eval = {
                'joint_configs': joint_configs,
                'joint_vels': joint_vels,
                'joint_accels': joint_accels,
                'joint_forces': joint_forces,
                'joint_jerks': joint_jerks,
                'cart_pos': cart_pos,
                'cart_vels': cart_vels,
                'cart_accels': cart_accels,
                'cart_jerks': cart_jerks,
                'contact_forces': contact_forces
                }        
    return np.nanmean(episode_rewards), np.nanmean(episode_successes), safety_eval

def load_checkpt_fp(cfg, checkpt_dir, bc_dir):
    model_fp = None
    step = 0

    if checkpt_dir is not None:
        while True:
            if os.path.exists(checkpt_dir / f'{str(step)}.pt'):
                model_fp = checkpt_dir / f'{str(step)}.pt'
            else:
                step -= cfg.save_freq
                step = max(0, step)
                break
            step += cfg.save_freq
        
    if model_fp is None:
        assert(step==0)

    if model_fp is None and cfg.get('bc_model_fp', None) is not None:
        if os.path.exists(cfg.bc_model_fp):
            model_fp = cfg.bc_model_fp
        elif os.path.exists(cfg.bc_model_fp+str(cfg.seed)+'.pt'):
            model_fp = cfg.bc_model_fp+str(cfg.seed)+'.pt'
        else:
            assert False, 'Could not find bc model file path'   

    bc_save_step = 0
    if model_fp is None:
        max_bc_steps = 2 * cfg.demos * cfg.episode_length
        bc_save_int = max(int(0.1*max_bc_steps), 10000)
        bc_save_step = bc_save_int
        # Check if there are existing bc models
        while True:
            if os.path.exists(bc_dir / f'bc_{str(bc_save_step)}.pt'):
                model_fp = bc_dir / f'bc_{str(bc_save_step)}.pt'
            else:
                bc_save_step -= bc_save_int
                bc_save_step = max(0, bc_save_step)
                break
            bc_save_step += bc_save_int  

    return model_fp, step, bc_save_step if bc_save_step > 0 else None 

@hydra.main(config_name="config", config_path="cfgs")
def train(cfg: dict):
    """Training script for online TD-MPC."""
    assert torch.cuda.is_available()
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    
    repo_path = git.Repo(hydra.utils.get_original_cwd(), search_parent_directories=True).working_tree_dir
    work_dir = Path(repo_path) / "modemv2" / "logs" / cfg.task / cfg.exp_name / str(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), work_dir)

    episode_dir = repo_path + f"/modemv2/episodes/{cfg.task}/{cfg.exp_name}/{str(cfg.seed)}" 
    if (not os.path.isdir(episode_dir)):
        os.makedirs(episode_dir)

    env, agent = make_env(cfg), TDMPC(cfg)
    demo_buffer = ReplayBuffer(deepcopy(cfg)) if cfg.get("demos", 0) > 0 else None
    buffer = ReplayBuffer(cfg)
    L = logger.Logger(work_dir, cfg)
    print(agent.model)

    model_fp, start_step, bc_start_step = load_checkpt_fp(cfg, L._model_dir, L._model_dir)
    skip_batch_train = (model_fp is not None) and (start_step >= cfg.seed_steps)

    assert(start_step == 0 or bc_start_step is None)
    if model_fp is not None:
        print('Loading agent '+str(model_fp))
        agent_data = torch.load(model_fp)
        if isinstance(agent_data, dict):
            agent.load(agent_data)
        else:
            agent = agent_data    
            agent.cfg = cfg  
        agent.model.eval()
        agent.model_target.eval()      


    # Load past episodes
    for i in range(start_step//cfg.episode_length):
        trace_fn = 'rollout'+f'{(i):010d}.pickle'
        trace_path = episode_dir+'/'+trace_fn  
        if os.path.isfile(trace_path):
            paths = Trace.load(trace_path)
            paths_episodes = trace2episodes(cfg=cfg,
                                            env=env,
                                            trace=paths,
                                            exclude_fails=False,
                                            is_demo=False)
            assert(len(paths_episodes)==1)
            print('Loaded training episode {} reward {}'.format(i,torch.sum(paths_episodes[0].reward).item()))
            buffer += paths_episodes[0]       
            print('Loaded episode {} of {}'.format(i+1,start_step//cfg.episode_length))
        else:
            print(
                colored(
                    'Failed to find episode at {}'.format(trace_path),
                    "yellow",
                    attrs=["bold"],
                )
            )            
    print('Loaded {} rollouts'.format(len(buffer)//cfg.episode_length))

    # Load demonstrations
    if cfg.get("demos", 0) > 0:

        valid_demo_buffer = None
        if cfg.bc_only:
            valid_demo_buffer = buffer

        demos = get_demos(cfg, env)

        for i,episode in enumerate(demos):
            if valid_demo_buffer is not None and i % 10 == 0:
                valid_demo_buffer += episode
            else:
                demo_buffer += episode
        print(colored(f"Loaded {cfg.demos} demonstrations", "yellow", attrs=["bold"]))
        print(colored("Phase 1: policy pretraining", "red", attrs=["bold"]))
        if model_fp is None or (bc_start_step is not None):
            if bc_start_step is None:
                bc_start_step = 0
            agent.init_bc(demo_buffer if cfg.get("demo_schedule", 0) != 0 else buffer, L, bc_start_step, valid_demo_buffer)
            agent.post_bc_load()
        print(colored("\nPhase 2: seeding", "green", attrs=["bold"]))

    if not cfg.vanilla_modem:
        agent.freeze_bc()

    if model_fp is not None: 
        if start_step > cfg.seed_steps:
            agent.unfreeze_online()

    if cfg.bc_only:
        L.save_model(agent, 0)
        print(colored(f'Model has been checkpointed', 'yellow', attrs=['bold']))
                        
        #eval_rew, eval_succ, _ = evaluate(env, agent, cfg, 0, 0, L.video, L.traj_plot, L.q_plot, policy_rollout=True)
        eval_rew, eval_succ, _ = evaluate(env, agent, cfg, 0, 0, L.video,  policy_rollout=True)
        common_metrics = {"env_step": 0, "episode_reward": eval_rew, "episode_success": eval_succ}
        L.log(common_metrics, category="eval")
        print('Eval reward: {}, Eval success: {}'.format(eval_rew, eval_succ))
        exit()

    # Run training
    start_time = time.time()
    start_step = start_step // cfg.action_repeat
    episode_idx = start_step // cfg.episode_length
    for step in range(start_step, start_step+cfg.train_steps + cfg.episode_length, cfg.episode_length):

        # Collect trajectory
        retry_episode = True
        while retry_episode:
            retry_episode = False
            obs = env.reset()
            episode = Episode(cfg, obs, env.state)
        
            trace = Trace('rollout'+f'{(step//cfg.episode_length):010d}')
            trace.create_group('Trial0')

            t = 0
            ep_q_std = []
            ep_uncertainty_weight = 0.0
            if cfg.real_robot:
                L.video.init(env, enabled=True)
            success_count = 0.0
            ep_start = time.time()
            while not episode.done:
                action, q_stats = agent.plan(obs, env.state, step=step, t=t) 
                if cfg.save_episodes:
                    trace.append_datums(group_key='Trial0', dataset_key_val=env.get_trace_dict(action.cpu().numpy()))
                obs, reward, done, info = env.step(action.cpu().numpy())
                episode += (obs, env.state, action, reward, done)
                success_count += float(info["success"])
                if cfg.real_robot:
                    L.video.record(env)
                if q_stats is not None:
                    ep_q_std.append(q_stats['model_std_topk'])
                    ep_uncertainty_weight += q_stats['uncertainty_weight']
                t+=1
            if cfg.real_robot:    
                print('Train ep duration {}'.format(time.time()-ep_start))

            if cfg.save_episodes:
                trace.append_datums(group_key='Trial0', dataset_key_val=env.get_trace_dict(action.cpu().numpy()))
            assert (
                len(episode) == cfg.episode_length
            ), f"Episode length {len(episode)} != {cfg.episode_length}"
            
            if cfg.real_robot:
                print('Checking success')

                task_success, new_rewards, retry_episode = env.post_process_task(episode.obs[:,0,-4:-1].cpu().numpy(), episode.state, eval_mode=False)
                if retry_episode:
                    print('Episode result unclear, retrying')
                    continue
                info['success'] = int(task_success)
                if task_success:
                    episode.reward = new_rewards
                    episode.cumulative_reward = torch.sum(episode.reward).item()
                    print('Ep reward {}'.format(episode.cumulative_reward))
                L.video.save(step*cfg.action_repeat, key="videos/train_video")
            else:
                task_success = (success_count >= 5)
                
            if len(ep_q_std) > 0:
                if task_success:
                    agent.succ_q_std.append(torch.tensor(ep_q_std, dtype=torch.float32, device=agent.device))
                else:
                    agent.fail_q_std.append(torch.tensor(ep_q_std, dtype=torch.float32, device=agent.device))

            if cfg.save_episodes:
                for _ in range(cfg.episode_length+1):
                    trace.append_datums(group_key='Trial0', dataset_key_val={'success':task_success})
                trace_fn = trace.name + '.pickle'
                trace_path = episode_dir+'/'+trace_fn
                print('Saving {}'.format(trace_path))
                trace.stack()
                trace.save(trace_name=trace_path, verify_length=True, f_res=np.float64)

        buffer += episode


        # Update model
        train_metrics = {}
        if ((step >= cfg.seed_steps and len(buffer) >= cfg.seed_steps) or cfg.seed_train):
            if step == cfg.seed_steps and not cfg.seed_train:
                print(
                    colored(
                        "Seeding complete, pretraining model...",
                        "yellow",
                        attrs=["bold"],
                    )
                )
                if skip_batch_train:
                    num_updates = 0
                else:
                    num_updates = cfg.seed_steps
            else:
                num_updates = cfg.episode_length
            for i in range(num_updates):
                train_metrics.update(agent.update(buffer, step + i, demo_buffer, train_pi=(step >cfg.seed_steps)))
            if step == cfg.seed_steps:
                agent.unfreeze_online()
                print(colored("Phase 3: interactive learning", "blue", attrs=["bold"]))

        # Log training episode
        episode_idx += 1
        env_step = int(step * cfg.action_repeat)
        common_metrics = {
            "episode": episode_idx,
            "step": step,
            "env_step": env_step,
            "total_time": time.time() - start_time,
            "episode_reward": episode.cumulative_reward,
            "episode_success": info.get("success", 0),
            'uncertainty_weight': ep_uncertainty_weight / cfg.episode_length
        }
        train_metrics.update(common_metrics)
        L.log(train_metrics, category="train")

        if cfg.save_model and env_step % cfg.save_freq == 0 and (start_step==0 or step > start_step):# and env_step > 0:
            L.save_model(agent, env_step)
            print(colored(f"Model has been checkpointed", "yellow", attrs=["bold"]))

        if not cfg.limit_mix_sched:
            agent.max_mix_prob = h.linear_schedule(cfg.mix_schedule, step)
        # Evaluate agent periodically
        if env_step % cfg.eval_freq == 0:
            
            eval_rew, eval_succ, safety_eval = evaluate(env, agent, cfg, step, env_step, L.video)

            for key in safety_eval.keys():
                eval_dir = work_dir/key
                
                if not os.path.exists(eval_dir):
                    os.makedirs(eval_dir)

                if len(safety_eval[key]) == 0:
                    safety_eval_val = np.array([])
                    safety_eval_mean = 0.0
                    safety_eval_max = 0.0
                    safety_eval_std = 0.0
                elif isinstance(safety_eval[key][0], float):
                    safety_eval_val = np.array(safety_eval[key])
                    safety_eval_mean = safety_eval_val.mean()
                    safety_eval_max = np.abs(safety_eval_val).max()
                    safety_eval_std = safety_eval_val.std()
                elif isinstance(safety_eval[key][0], np.ndarray):
                    safety_eval_val = np.stack(safety_eval[key])
                    safety_eval_max = np.abs(safety_eval_val).max()
                    safety_eval_val_norm = np.linalg.norm(safety_eval_val, axis=1)
                    safety_eval_mean = safety_eval_val_norm.mean()
                    safety_eval_std = safety_eval_val_norm.std()
                common_metrics.update({
                                        f'{str(key)}_mean': safety_eval_mean,
                                        f'{str(key)}_max': safety_eval_max,
                                        f'{str(key)}_std': safety_eval_std
                })
                eval_fp = eval_dir / f"{str(key)}{str(step)}.pickle"
                torch.save(safety_eval_val, eval_fp)
                
            print('Ep Reward {}, Ep Succ {}'.format(eval_rew, eval_succ))
            
            if cfg.limit_mix_sched:
                if eval_succ >= agent.max_eval_success:
                    agent.max_eval_success = eval_succ

                    sched_model_act_prob = h.linear_schedule(cfg.mix_schedule, step)
                    if agent.max_mix_prob < sched_model_act_prob:
                        agent.max_mix_prob += agent.mix_prob_inc

            common_metrics.update(
                {"episode_reward": eval_rew, "episode_success": eval_succ}
            )
            L.log(common_metrics, category="eval")


    L.finish()
    print("\nTraining completed successfully")


if __name__ == "__main__":
    train()
