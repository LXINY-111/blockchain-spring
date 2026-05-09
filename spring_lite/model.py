import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor):
        logits = self.actor(state)
        value = self.critic(state).squeeze(-1)
        return logits, value

    def get_action(self, state: torch.Tensor):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values

    @torch.no_grad()
    def act_deterministic(self, state: torch.Tensor):
        logits, value = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1)
        confidence = torch.max(probs, dim=-1).values
        return action, confidence, value