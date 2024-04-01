# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import algorithm.helper as h
from tqdm import tqdm
from algorithm.helper import gather_paths_parallel
from collections import deque

class AC(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, cfg.action_dim)
        self._Q1, self._Q2 = h.q(cfg), h.q(cfg)
        self.pi_optim = torch.optim.Adam(self._pi.parameters(), lr=self.cfg.lr)

    def track_q_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        for m in [self._Q1, self._Q2]:
            h.set_requires_grad(m, enable)

    def track_pi_grad(self, enable=True):
        h.set_requires_grad(self._pi, enable)            

    def pi(self, z, std=0):
        """Samples an action from the learned policy (pi)."""
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(mu) * std
            return h.TruncatedNormal(mu, std).sample(clip=0.3)
        return mu

    def Q(self, z, a):
        """Predict state-action value (Q)."""
        x = torch.cat([z, a], dim=-1)
        return self._Q1(x), self._Q2(x)        
    

class TOLD(nn.Module):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._dynamics = h.mlp(
            cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, cfg.latent_dim
        )
        self._reward = h.mlp(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, 1)
        
        self._encoder = nn.ModuleList(h.enc(cfg))
        self._state_encoder = nn.ModuleList(h.state_enc(cfg))
        self._learn_encoder = nn.ModuleList(h.enc(cfg))
        self._learn_state_encoder = nn.ModuleList(h.state_enc(cfg))

        self._acs = nn.ModuleList([AC(cfg) for _ in range(cfg.ensemble_size)])

        self.apply(h.orthogonal_init)
        for ac in self._acs:
            for m in [ac._Q1, ac._Q2]:
                m[-1].weight.data.fill_(0)
                m[-1].bias.data.fill_(0)
        self._reward[-1].weight.data.fill_(0)
        self._reward[-1].bias.data.fill_(0)

    def track_encoder_grad(self, enable):
        for m in self._encoder:
            h.set_requires_grad(m, enable)
        for m in self._state_encoder:
            h.set_requires_grad(m, enable)

    def track_learn_encoder_grad(self, enable):
        for m in self._learn_encoder:
            h.set_requires_grad(m, enable)
        for m in self._learn_state_encoder:
            h.set_requires_grad(m, enable)        

    def h(self, obs, state, compute_learned=True):
        """Encodes an observation into its latent representation (h)."""
        x = self._state_encoder[0](state)
        if self.cfg.img_size > 0:
            for i, enc in enumerate(self._encoder):
                x = enc(obs[:,i]) + x
        x = self._state_encoder[1](x)

        if compute_learned:
            x_learn = self._learn_state_encoder[0](state)
            if self.cfg.img_size > 0:
                for i, enc in enumerate(self._learn_encoder):
                    x_learn = enc(obs[:,i]) + x_learn
            x_learn = self._learn_state_encoder[1](x_learn)            
        else:
            x_learn = None

        return x, x_learn

    def next(self, z, a):
        """Predicts next latent state (d) and single-step reward (R)."""
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x), self._reward(x)

    
class TDMPC:
    """
    Implementation of TD-MPC learning + inference.
    Adapted from https://github.com/nicklashansen/tdmpc"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda")
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        self.model = TOLD(cfg).cuda()
        self.model_target = deepcopy(self.model)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.bc_optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.aug = h.RandomShiftsAug(cfg)
        self.model.eval()
        self.model_target.eval()
        self.batch_size = cfg.batch_size
        
        self.succ_q_std = deque([], maxlen=30)
        self.fail_q_std = deque([], maxlen=30)
        
        self.max_eval_success = 0.0
        self.max_mix_prob = 0.0
        self.mix_prob_inc = 0.05

        print(
            "Total parameters: {:,}".format(
                sum(p.numel() for p in self.model.parameters())
            )
        )

    def state_dict(self):
        """Retrieve state dict of TOLD model, including slow-moving target network."""
        return {
            "model": self.model.state_dict(),
            "model_target": self.model_target.state_dict(),
        }

    def save(self, fp):
        """Save state dict of TOLD model to filepath."""
        state_dict = self.state_dict()
        torch.save(state_dict, fp)

    def load(self, fp):
        """Load a saved state dict from filepath (or dictionary) into current agent."""
        d = fp if isinstance(fp, dict) else torch.load(fp)
        self.model.load_state_dict(d["model"])
        self.model_target.load_state_dict(d["model_target"])

    @torch.no_grad()
    def act(self, obs, state, noise=None):
        """Sample action from current policy."""
        if not self.cfg.vanilla_modem:
            z, _ = self.model.h(obs, state, compute_learned=False)
            return self.model._acs[0].pi(z, self.cfg.min_std if noise is None else noise)
        else:
            assert(self.cfg.ensemble_size == 2)
            z, z_learned = self.model.h(obs, state, compute_learned=True)
            return self.model._acs[1].pi(z_learned, self.cfg.min_std if noise is None else noise)
    # actions_bc: (# traj, action dim)
    # actions_learned: (horizon, # traj, action dim)
    @torch.no_grad()
    def estimate_value(self, z_bc, z_learned, actions_bc, actions_learned, horizon):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1

        if actions_bc is not None:
            # Compute q val for bc
            q_bc = torch.min(*self.model._acs[0].Q(z_bc, actions_bc))
        else:
            q_bc = None


        if actions_learned is not None:
            for t in range(horizon):
                z_learned, reward = self.model.next(z_learned, actions_learned[t])
                G += discount * reward
                discount *= self.cfg.discount

            # Get learned actions
            end_actions_learned = []
            for i in range(1,len(self.model._acs)):
                end_actions_learned.append(self.model._acs[i].pi(z_learned, self.cfg.min_std))
            
            assert(len(end_actions_learned)+1 == len(self.model._acs))
            q_learned = []
            for i in range(1,len(self.model._acs)):
                q1_max, q2_max = None, None
                for j in range(0,len(end_actions_learned)):
                    if i == j+1 and self.cfg.ensemble_size > 2:
                        continue
                    q1_val, q2_val = self.model._acs[i].Q(z_learned, end_actions_learned[j])
                    if q1_max is None:
                        q1_max = q1_val
                    else:
                        q1_max = torch.max(q1_max, q1_val)
                    if q2_max is None:
                        q2_max = q2_val
                    else:
                        q2_max = torch.max(q2_max, q2_val)
                q_learned.append(q1_max)
                q_learned.append(q2_max)
            q_learned = torch.stack(q_learned, dim=0)
            G = G + discount*q_learned
            G_min = torch.min(G, dim=0).values
            G_mean = torch.mean(G, dim=0)        
            G_std = torch.std(G, dim=0)
        else:
            G_min, G_mean, G_std = None, None, None

        return q_bc, G_min, G_mean, G_std

    @torch.no_grad()
    def compute_elite_actions(self, actions, value, num_elites):
        assert(len(value.shape)==1)
        elite_idxs = torch.topk(
            value, num_elites, dim=0
        ).indices

        if len(actions.shape)== 2:
            elite_value, elite_actions = value[elite_idxs], actions[elite_idxs]
        elif len(actions.shape) == 3:
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

        # Update parameters
        max_value = elite_value.max(0)[0]
        score = torch.exp(self.cfg.temperature * (elite_value - max_value))
        score /= score.sum(0)

        return elite_actions, score

    @torch.no_grad()
    def plan(self, obs, state, eval_mode=False, step=None, t=0):
        """
        Plan next action using TD-MPC inference.
        obs: raw input observation.
        eval_mode: uniform sampling and action noise is disabled during evaluation.
        step: current time step. determines e.g. planning horizon.
        t: current time step.
        """
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )

        # Seed steps
        if step < self.cfg.seed_steps:# and not eval_mode:
            return self.act(obs, state).squeeze(0), None

        horizon = int(
            min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step))
        )

        # Encode observation only once
        z_bc, z_learned = self.model.h(obs, state, compute_learned=True)
        
        model_act_prob = h.linear_schedule(self.cfg.mix_schedule, step)
        if self.cfg.real_robot:
            model_act_prob = min(model_act_prob, self.max_mix_prob)
            
        use_model = True
        if np.random.rand() > model_act_prob:
            use_model = False

        if use_model:
            num_learn_traj = int(self.cfg.num_samples*self.cfg.mixture_coef / (self.cfg.ensemble_size-1))*(self.cfg.ensemble_size-1)
            num_learn_elites = int(self.cfg.num_elites*self.cfg.mixture_coef) 
        else:
            num_learn_traj = int(self.cfg.num_samples*(1.0-self.cfg.mixture_coef) / (self.cfg.ensemble_size-1))*(self.cfg.ensemble_size-1)
            num_learn_elites = int(self.cfg.num_elites*(1.0-self.cfg.mixture_coef))
        
        num_bc_traj = self.cfg.num_samples-num_learn_traj
        num_bc_elites = self.cfg.num_elites-num_learn_elites
        traj_per_ac = int(num_learn_traj / (self.cfg.ensemble_size-1))
        assert(num_bc_traj+traj_per_ac*(self.cfg.ensemble_size-1) == self.cfg.num_samples)
        
        if num_bc_traj > 0:
            if self.cfg.ignore_bc:
                actions_bc = 2*torch.randn(num_bc_traj,
                                           self.cfg.action_dim,
                                           device=self.device)
            else:
                actions_bc = self.model._acs[0].pi(z_bc.repeat(num_bc_traj,1), 
                                                self.cfg.min_std)
        else:
            actions_bc = None

        if num_learn_traj > 0 :   
            actions_learned = torch.empty((horizon, num_learn_traj, self.cfg.action_dim), dtype=torch.float32, device=self.device)
            for i in range(1,len(self.model._acs)):
                z = z_learned.repeat(traj_per_ac,1)
                for t in range(horizon):
                    actions_learned[t, (i-1)*traj_per_ac:i*traj_per_ac] = self.model._acs[i].pi(z, self.cfg.min_std)
                    z, _ = self.model.next(z, actions_learned[t, (i-1)*traj_per_ac:i*traj_per_ac])
        else:
            actions_learned = None

        mean_bc = None
        std_bc = self.cfg.min_std * torch.ones(self.cfg.action_dim, device=self.device)
        mean_learned = None
        std_learned = self.cfg.min_std * torch.ones(horizon, self.cfg.action_dim, device=self.device)

        z_bc = z_bc.repeat(num_bc_traj, 1)
        z_learned = z_learned.repeat(num_learn_traj,1)
        


        if self.cfg.vanilla_modem:
            assert(num_learn_traj == self.cfg.num_samples)
            assert(use_model)
            assert(self.cfg.val_min_w > 0.999)
            assert(self.cfg.val_mean_w < 0.001)
            assert(self.cfg.val_std_w < 0.001)
            std_learned = 2 * torch.ones(horizon, self.cfg.action_dim, device=self.device)

            zero_samples = int(0.95*self.cfg.num_samples)
            if t==0 and hasattr(self, "_prev_mean"):
                actions_learned[:-1, :zero_samples, :] = self._prev_mean[1:].unsqueeze(1)
                actions_learned[-1, :zero_samples, :] = 0.0
            else:
                actions_learned[:, :zero_samples, :] = 0.0
            actions_learned[:, :zero_samples, :] += std_learned.unsqueeze(1) * torch.randn(
                                                                horizon,
                                                                zero_samples,
                                                                self.cfg.action_dim,
                                                                device=std_learned.device,
                                                            )
            actions_learned = torch.clamp(actions_learned, -1, 1)
    

        # Iterate
        for _ in range(self.cfg.iterations):
            if mean_bc is not None:
                actions_bc = mean_bc.unsqueeze(0) +  std_bc.unsqueeze(0)*torch.randn(num_bc_traj, self.cfg.action_dim, device=self.device)
                actions_bc = torch.clamp(actions_bc, -1, 1)
            if mean_learned is not None:
                actions_learned = mean_learned.unsqueeze(1) + std_learned.unsqueeze(1)*torch.randn(horizon,num_learn_traj,self.cfg.action_dim, device=self.device)
                actions_learned = torch.clamp(actions_learned, -1, 1)
            
            
            value_bc, G_min, G_mean, G_std = self.estimate_value(z_bc=z_bc,
                                                                                z_learned=z_learned,
                                                                                actions_bc=actions_bc,
                                                                                actions_learned=actions_learned,
                                                                                horizon=horizon)

            if value_bc is not None:
                value_bc = value_bc.nan_to_num(-self.cfg.episode_length).squeeze(1)
                elite_actions_bc, score_bc = self.compute_elite_actions(actions_bc, value_bc, num_bc_elites)
                assert(len(score_bc.shape)==1 and score_bc.shape[0] == num_bc_elites)
                _mean_bc = torch.sum(score_bc.unsqueeze(1) * elite_actions_bc, dim=0) / (
                    score_bc.sum(0) + 1e-9
                )
                _std_bc = torch.sqrt(
                    torch.sum(
                        score_bc.unsqueeze(1) * (elite_actions_bc - _mean_bc) ** 2,
                        dim=0,
                    )
                    / (score_bc.sum(0) + 1e-9)
                )
                _std_bc = _std_bc.clamp_(0.0, 2)
                if mean_bc is not None:
                    mean_bc = self.cfg.momentum * mean_bc + (1 - self.cfg.momentum) * _mean_bc
                else:
                    mean_bc = _mean_bc
                std_bc = _std_bc

            if G_min is not None:
                value_learn = self.cfg.val_min_w*G_min + self.cfg.val_mean_w*G_mean + self.cfg.val_std_w*G_std
                value_learn = value_learn.nan_to_num(-self.cfg.episode_length).squeeze(1)
                elite_actions_learn, score_learn = self.compute_elite_actions(actions_learned, value_learn, num_learn_elites)
                assert(len(score_learn.shape) == 1 and score_learn.shape[0] == num_learn_elites)
                _mean_learned = torch.sum(score_learn.unsqueeze(1) * elite_actions_learn, dim=1) / (
                    score_learn.sum(0) + 1e-9
                )
                _std_learned = torch.sqrt(
                    torch.sum(
                        score_learn.unsqueeze(1) * (elite_actions_learn - _mean_learned.unsqueeze(1)) ** 2,
                        dim=1,
                    )
                    / (score_learn.sum(0) + 1e-9)
                )

                if mean_learned is not None:
                    mean_learned = self.cfg.momentum * mean_learned + (1 - self.cfg.momentum) * _mean_learned
                else:
                    mean_learned = _mean_learned
                std_learned = _std_learned

                if self.cfg.vanilla_modem:
                    self._prev_mean = mean_learned

        if not use_model:
            all_actions = actions_bc
            all_value = value_bc
        else:
            all_actions = actions_learned[0]
            all_value = value_learn
        
        elite_actions, score = self.compute_elite_actions(all_actions, all_value, self.cfg.num_elites)

        out_mean = torch.sum(score.unsqueeze(1) * elite_actions, dim=0) / (
            score.sum(0) + 1e-9
        )
        out_std = torch.sqrt(
            torch.sum(
                score.unsqueeze(1) * (elite_actions - out_mean) ** 2,
                dim=0,
            )
            / (score.sum(0) + 1e-9)
        )
        out_std = out_std.clamp_(self.std, 2)

        # Outputs
        score = score.cpu().numpy()
        action = elite_actions[np.random.choice(np.arange(score.shape[0]), p=score)]

        if not eval_mode:
            action += out_std * torch.randn(self.cfg.action_dim, device=self.device)

        return action.clamp_(-1, 1), None #q_stats

    def eval_batch(self, buffer):
        obs, _, action, _, state, _, _, _ = buffer.sample()
        z, _ = self.model.h(self.aug(obs), state, compute_learned=False)
        a = self.model._acs[0].pi(z)
        return h.mse(a, action[0], reduce=True)
    
    def init_bc(self, train_buffer, log, start_step=0, valid_buffer=None):
        """Initialize policy using a behavior cloning objective (iterations: 2x #samples)."""
        self.model.train()
        end_step = 2 * self.cfg.demos * self.cfg.episode_length
        bc_save_int = max(int(0.1*end_step), 5000)
        mse_total = 0.0
        for i in tqdm(
            range(start_step, end_step), "Pretraining policy"
        ):
            self.bc_optim.zero_grad(set_to_none=True)
            mse = self.eval_batch(train_buffer)
            mse.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.grad_clip_norm,
                error_if_nonfinite=False,
            )
            self.bc_optim.step()
            mse_total += mse.item()

            if i > 0 and i % bc_save_int == 0:
                log.save_model(self, 'bc_'+str(i))

            if self.cfg.bc_only and i % 50 == 0 and i > 0:
                
                valid_mse = 0.0
                if valid_buffer is not None:
                    valid_mse = self.eval_batch(valid_buffer)
                    valid_mse = valid_mse.item()
                
                    train_metrics = {
                    'env_step': i,
                    'bc_mse': mse_total / 50,
                    'valid_mse': valid_mse}
                log.log(train_metrics, category='train')
                mse_total = 0.0		

        self.model.eval()

    def post_bc_load(self):
        self.model._learn_encoder.load_state_dict(self.model._encoder.state_dict())
        self.model._learn_state_encoder.load_state_dict(self.model._state_encoder.state_dict())

        self.model_target._encoder.load_state_dict(self.model._encoder.state_dict())
        self.model_target._state_encoder.load_state_dict(self.model._state_encoder.state_dict())

        self.model_target._learn_encoder.load_state_dict(self.model._encoder.state_dict())
        self.model_target._learn_state_encoder.load_state_dict(self.model._state_encoder.state_dict())

        for i in range(0,len(self.model._acs)):
            if i > 0:
                self.model._acs[i]._pi.load_state_dict(self.model._acs[0]._pi.state_dict())
            self.model_target._acs[i]._pi.load_state_dict(self.model._acs[0]._pi.state_dict())

    def freeze_bc(self):
        self.model.track_encoder_grad(False)
        self.model.track_learn_encoder_grad(False)

        for i in range(len(self.model._acs)):
            self.model._acs[i].track_pi_grad(False)          
        
        self.model_target.track_encoder_grad(False)
        self.model_target.track_learn_encoder_grad(False)
        
        for i in range(len(self.model_target._acs)):
            self.model_target._acs[i].track_pi_grad(False)

    def unfreeze_online(self):
  
        self.model.track_learn_encoder_grad(True)
        for i in range(1,len(self.model._acs)):
            self.model._acs[i].track_pi_grad(True)          

        self.model_target.track_learn_encoder_grad(True)
        for i in range(1,len(self.model_target._acs)):
            self.model_target._acs[i].track_pi_grad(True)     

    def update_pi(self, zs):
        """Update policy using a sequence of latent states."""
        total_pi_loss = 0.0
        for i in range(1, len(self.model._acs)):
            self.model._acs[i].pi_optim.zero_grad(set_to_none=True)
            self.model._acs[i].track_q_grad(False)

            # Loss is a weighted sum of Q-values
            pi_loss = 0
            for t, z in enumerate(zs):
                a = self.model._acs[i].pi(z, self.cfg.min_std)
                Q = torch.min(*self.model._acs[i].Q(z, a))
                pi_loss += -Q.mean() * (self.cfg.rho**t)

            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model._acs[i]._pi.parameters(),
                self.cfg.grad_clip_norm,
                error_if_nonfinite=False,
            )
            self.model._acs[i].pi_optim.step()
            self.model._acs[i].track_q_grad(True)
            total_pi_loss += pi_loss.item()
        return total_pi_loss/(len(self.model._acs)-1)

    @torch.no_grad()
    def _td_target(self, next_obs, next_state, reward):
        """Compute the TD-target from a reward and the observation at the following time step."""
        next_z_bc, next_z_learn = self.model.h(next_obs, next_state, compute_learned=True)
        td_targets = []
        for i in range(len(self.model._acs)):
            if i == 0:
                td_target = reward + self.cfg.discount * torch.min(
                    *self.model_target._acs[i].Q(next_z_bc, self.model._acs[i].pi(next_z_bc, self.cfg.min_std))
                )
            else:
                td_target = reward + self.cfg.discount * torch.min(
                    *self.model_target._acs[i].Q(next_z_learn, self.model._acs[i].pi(next_z_learn, self.cfg.min_std))
                )
            td_targets.append(td_target)
        return td_targets

    def update(self, replay_buffer, step=int(1e6), demo_buffer=None, train_pi=False):
        """Main update function. Corresponds to one iteration of the TOLD model learning."""
        # Update oversampling ratio
        self.demo_batch_size = int(
            h.linear_schedule(self.cfg.demo_schedule, step) * self.batch_size
        )

        if demo_buffer is None:
            self.demo_batch_size = 0
        else:
            demo_buffer.cfg.batch_size = self.demo_batch_size
        replay_buffer.cfg.batch_size = self.batch_size - self.demo_batch_size

        # Sample from interaction dataset
        (
            obs,
            next_obses,
            action,
            reward,
            state,
            next_states,
            idxs,
            weights,
        ) = replay_buffer.sample()

        # Sample from demonstration dataset
        if self.demo_batch_size > 0:
            (
                demo_obs,
                demo_next_obses,
                demo_action,
                demo_reward,
                demo_state,
                demo_next_states,
                demo_idxs,
                demo_weights,
            ) = demo_buffer.sample()
            obs, next_obses, action, reward, state, next_states, idxs, weights = (
                torch.cat([obs, demo_obs]),
                torch.cat([next_obses, demo_next_obses], dim=1),
                torch.cat([action, demo_action], dim=1),
                torch.cat([reward, demo_reward], dim=1),
                torch.cat([state, demo_state]),
                torch.cat([next_states, demo_next_states], dim=1),
                torch.cat([idxs, demo_idxs]),
                torch.cat([weights, demo_weights]),
            )

        # Prepare for update
        self.optim.zero_grad(set_to_none=True)
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        # Representation
        z_bc, z_learn = self.model.h(self.aug(obs), state, compute_learned=True)
        zs = [z_learn.detach()]

        consistency_loss, reward_loss, value_loss, priority_loss = 0, 0, 0, 0
        for t in range(self.cfg.horizon):

            # Predictions
            Qs = []
            for i in range(len(self.model._acs)):
                if i == 0:
                    Q1, Q2 = self.model._acs[i].Q(z_bc, action[t])        
                else:
                    Q1, Q2 = self.model._acs[i].Q(z_learn, action[t])
                Qs.append((Q1,Q2))
 
            z_learn, reward_pred = self.model.next(z_learn, action[t])
            with torch.no_grad():
                next_obs = self.aug(next_obses[t])
                next_state = next_states[t]
                z_bc, next_z = self.model_target.h(next_obs, next_state)
                td_targets = self._td_target(next_obs, next_state, reward[t])
            zs.append(z_learn.detach())

            # Losses
            rho = self.cfg.rho**t
            consistency_loss += rho * torch.mean(h.mse(z_learn, next_z), dim=1, keepdim=True)
            reward_loss += rho * h.mse(reward_pred, reward[t])
            for i in range(len(self.model._acs)):
                value_loss += rho * (h.mse(Qs[i][0], td_targets[i]) + h.mse(Qs[i][1], td_targets[i]))
                priority_loss += rho * (h.l1(Qs[i][0], td_targets[i]) + h.l1(Qs[i][1], td_targets[i]))

        # Optimize model
        total_loss = (
            self.cfg.consistency_coef * consistency_loss.clamp(max=1e4)
            + self.cfg.reward_coef * reward_loss.clamp(max=1e4)
            + self.cfg.value_coef * value_loss.clamp(max=1e4)
        )
        weighted_loss = (total_loss.squeeze(1) * weights).mean()
        weighted_loss.register_hook(lambda grad: grad * (1 / self.cfg.horizon))
        weighted_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False
        )
        self.optim.step()

        # Update priorities for both buffers
        if self.cfg.per:
            priorities = priority_loss.clamp(max=1e4).detach()
            replay_buffer.update_priorities(
                idxs[: self.cfg.batch_size], priorities[: self.cfg.batch_size]
            )
            if demo_buffer is not None:
                demo_buffer.update_priorities(demo_idxs, priorities[self.cfg.batch_size :])

        if train_pi:
            pi_loss = self.update_pi(zs)
        else:
            pi_loss = 0.0

        if step % self.cfg.update_freq == 0:
            h.soft_update_params(self.model, self.model_target, self.cfg.tau)

        self.model.eval()
        return {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "weighted_loss": float(weighted_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "demo_batch_size": self.demo_batch_size,
        }
