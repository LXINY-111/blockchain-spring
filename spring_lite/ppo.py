from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from model import ActorCritic


@dataclass
class RolloutBuffer:
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def __len__(self):
        return len(self.states)


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = torch.device(device)

        self.net = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def select_action(self, state) -> Tuple[int, float, float]:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, _, value = self.net.get_action(state_t)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
        )

    def deterministic_action(self, state) -> Tuple[int, float]:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, confidence, _ = self.net.act_deterministic(state_t)

        return int(action.item()), float(confidence.item())

    def _discounted_returns(self, rewards, dones):
        returns = []
        running_return = 0.0

        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                running_return = 0.0
            running_return = reward + self.gamma * running_return
            returns.append(running_return)

        returns.reverse()
        return np.asarray(returns, dtype=np.float32)

    def update(self, buffer: RolloutBuffer):
        if len(buffer) == 0:
            return {}

        states = torch.tensor(np.asarray(buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32, device=self.device)

        returns_np = self._discounted_returns(buffer.rewards, buffer.dones)
        returns = torch.tensor(returns_np, dtype=torch.float32, device=self.device)

        old_values = torch.tensor(buffer.values, dtype=torch.float32, device=self.device)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        last_loss = {}
        for _ in range(self.ppo_epochs):
            log_probs, entropy, values = self.net.evaluate_actions(states, actions)

            ratio = torch.exp(log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages

            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns)
            entropy_loss = -entropy.mean()

            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
            self.optimizer.step()

            last_loss = {
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.mean().item()),
            }

        return last_loss

    def save(self, path, extra=None):
        payload = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "model_state_dict": self.net.state_dict(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    def load(self, path):
        payload = torch.load(path, map_location=self.device)
        self.net.load_state_dict(payload["model_state_dict"])
        return payload