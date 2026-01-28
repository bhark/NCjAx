#!/usr/bin/env python3

'''
This is a rough (read: slightly messy and too compact) 
example of training the NCA substrate using Deep Q Learning. 
It is provided as reference only.
'''

import jax, jax.numpy as jnp, numpy as np, optax, gymnasium as gym, random
from collections import namedtuple
from tqdm import tqdm
from NCjAx import Config, NCA

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class _SumTree:
    def __init__(self, capacity: int):
        self.cap = 1 << (capacity - 1).bit_length()
        self.tree = np.zeros(2 * self.cap, dtype=np.float32)
    def total(self) -> float: return float(self.tree[1])
    def update(self, idx: int, value: float):
        i = idx + self.cap; self.tree[i] = value; i //= 2
        while i >= 1:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]; i //= 2
    def get(self, prefix_sum: float) -> int:
        i = 1
        while i < self.cap:
            left = 2 * i
            if self.tree[left] >= prefix_sum: i = left
            else: prefix_sum -= self.tree[left]; i = left + 1
        return i - self.cap
    def get_value(self, idx: int) -> float: return float(self.tree[idx + self.cap])

class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=200_000, eps=1e-5):
        self.capacity, self.alpha, self.beta_start, self.beta_frames, self.eps = capacity, alpha, beta_start, beta_frames, eps
        self.storage, self.next, self.size = [None]*capacity, 0, 0
        self.tree, self.max_priority, self.frame = _SumTree(capacity), 1.0, 0
    def __len__(self): return self.size
    def beta_by_frame(self):
        t = min(1.0, self.frame / float(self.beta_frames))
        return self.beta_start + t * (1.0 - self.beta_start)
    def push(self, transition: Transition):
        self.storage[self.next] = transition
        self.tree.update(self.next, (self.max_priority + self.eps) ** self.alpha)
        self.next = (self.next + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def sample(self, batch_size):
        assert self.size > 0
        self.frame += batch_size
        total = self.tree.total()
        idxs, transitions = [], []
        for _ in range(batch_size):
            u = random.random() * max(total, 1e-12)
            idx = self.tree.get(u)
            if idx >= self.size: idx = random.randrange(self.size)
            idxs.append(idx); transitions.append(self.storage[idx])
        batch = Transition(*zip(*transitions))
        ps = np.array([self.tree.get_value(i) for i in idxs], dtype=np.float32)
        probs = ps / (total + 1e-12)
        beta = self.beta_by_frame()
        weights = (probs * self.size) ** (-beta); weights /= weights.max() + 1e-12
        return batch, np.array(idxs, dtype=np.int32), weights.astype(np.float32)
    def update_priorities(self, idxs, td_errors):
        prios = (np.abs(td_errors) + self.eps) ** self.alpha
        for i, p in zip(idxs, prios): self.tree.update(int(i), float(p))
        self.max_priority = max(self.max_priority, float(prios.max(initial=self.max_priority)))

class NCAdQN:
    def __init__(self, obs_dim=4, action_dim=2, grid_size=12, hidden_channels=6, conv_features=20, k_steps=35,
                 lr=1e-3, gamma=0.98, epsilon_start=1.0, epsilon_end=0.0, epsilon_decay=0.9995,
                 target_update_freq=100, seed=42):
        self.key = jax.random.PRNGKey(seed)
        self.config = Config(grid_size=grid_size, hidden_channels=hidden_channels, num_input_nodes=obs_dim,
                             num_output_nodes=action_dim, perception='learned3x3', conv_features=conv_features,
                             fire_rate=0.5, k_default=k_steps, hidden=30, dtype=jnp.float32)
        self.nca = NCA(self.config)
        k1, k2, k3, k4 = jax.random.split(self.key, 4)
        raw_params = self.nca.init_params(k1)
        self.params, _ = self.nca.pretrain(raw_params, k4, steps=6000)
        self.target_params, self.key = self.params, k3
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
        self.opt_state = self.optimizer.init(self.params)
        self.gamma, self.epsilon, self.epsilon_end, self.epsilon_decay = gamma, epsilon_start, epsilon_end, epsilon_decay
        self.target_update_freq, self.update_counter = target_update_freq, 0
        self._get_q_values = jax.jit(self._get_q_values_impl)
        self._update_step = jax.jit(self._update_step_impl)
    def _get_q_values_impl(self, params, obs, key):
        k1, k2 = jax.random.split(key)
        state = self.nca.init_state(k1)
        q_values, _ = self.nca.process(state, params, k2, obs, mode='set')
        return q_values * 100
    def get_q_values(self, obs, params=None):
        if params is None: params = self.params
        obs_norm = obs / np.array([2.4, 10.0, 0.21, 10.0])
        self.key, subkey = jax.random.split(self.key)
        return np.array(self._get_q_values(params, jnp.array(obs_norm, dtype=jnp.float32), subkey))
    def select_action(self, obs, training=True):
        if training and random.random() < self.epsilon: return random.randint(0, 1)
        return int(np.argmax(self.get_q_values(obs)))
    def _update_step_impl(self, params, opt_state, batch_states, batch_actions, batch_rewards, batch_next_states,
                          batch_dones, target_params, key, importance_weights):
        def loss_fn(params, key):
            n = batch_states.shape[0]
            total_loss = 0.0; total_td_error = 0.0; td_vec = jnp.zeros((n,), dtype=jnp.float32)
            for i in range(n):
                k1, k2, key = jax.random.split(key, 3)
                st = self.nca.init_state(k1)
                qv, st_after = self.nca.process(st, params, k1, batch_states[i], mode='set')
                pen = self.nca.get_overflow_penalty(st_after)
                q = qv[batch_actions[i]]
                stt = self.nca.init_state(k2)
                nxt, _ = self.nca.process(stt, target_params, k2, batch_next_states[i], mode='set')
                target = batch_rewards[i] + self.gamma * jnp.max(nxt) * (1 - batch_dones[i])
                td = q - target
                per = jnp.where(jnp.abs(td) <= 1.0, 0.5 * td**2, jnp.abs(td) - 0.5)
                w = importance_weights[i]
                total_loss += per * w + pen
                total_td_error += jnp.abs(td)
                td_vec = td_vec.at[i].set(jnp.abs(td))
            avg_loss = total_loss / jnp.sum(importance_weights)
            avg_td = total_td_error / n
            return avg_loss, (avg_loss, avg_td, td_vec)
        (loss, (avg_loss, avg_td, td_vec)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, avg_loss, avg_td, td_vec
    def update(self, batch, importance_weights=None):
        obs_norm = np.array([2.4, 10.0, 0.21, 10.0])
        bs = jnp.array(batch.state, dtype=jnp.float32) / obs_norm
        ba = jnp.array(batch.action, dtype=jnp.int32)
        br = jnp.array(batch.reward, dtype=jnp.float32)
        bns = jnp.array(batch.next_state, dtype=jnp.float32) / obs_norm
        bd = jnp.array(batch.done, dtype=jnp.float32)
        iw = jnp.ones(len(br), dtype=jnp.float32) if importance_weights is None else jnp.array(importance_weights, dtype=jnp.float32)
        self.key, subkey = jax.random.split(self.key)
        self.params, self.opt_state, loss, td_error, td_vec = self._update_step(self.params, self.opt_state, bs, ba, br, bns, bd, self.target_params, subkey, iw)
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0: self.target_params = self.params
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss), np.array(td_vec, dtype=np.float32)

def train(num_episodes=3000, batch_size=64, buffer_size=30000, min_buffer_size=2500, seed=42):
    env = gym.make('CartPole-v1'); env.action_space.seed(seed)
    agent = NCAdQN(seed=seed)
    buffer = PrioritizedReplayBuffer(buffer_size, alpha=0.6, beta_start=0.4, beta_frames=200_000)
    episode_rewards = []
    obs, _ = env.reset(seed=seed)
    while len(buffer) < min_buffer_size:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(Transition(obs, action, reward, next_obs, done))
        obs = next_obs if not done else env.reset(seed=seed)[0]
    pbar = tqdm(range(num_episodes), desc="training")
    for ep in pbar:
        obs, _ = env.reset(seed=seed + ep)
        done, steps, ep_reward = False, 0, 0
        while not done and steps < 500:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward; steps += 1
            buffer.push(Transition(obs, action, reward, next_obs, done))
            if len(buffer) >= min_buffer_size and steps % 4 == 0:
                for _ in range(2):
                    batch, idxs, weights = buffer.sample(batch_size)
                    loss, td = agent.update(batch, weights)
                    buffer.update_priorities(idxs, td)
            obs = next_obs
        episode_rewards.append(ep_reward)
        avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        pbar.set_postfix(r=f"{ep_reward:.0f}", avg=f"{avg:.1f}", eps=f"{agent.epsilon:.3f}")
    env.close()
    return agent, episode_rewards

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="nca-dqn + per on cartpole")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(num_episodes=args.episodes, seed=args.seed)
