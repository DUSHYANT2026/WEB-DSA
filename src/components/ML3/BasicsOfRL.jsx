import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function BasicsOfRL() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "🤖 RL Fundamentals",
      id: "fundamentals",
      description: "Core concepts that define the reinforcement learning paradigm.",
      keyPoints: [
        "Agent-Environment interaction loop",
        "Markov Decision Processes (MDPs)",
        "Rewards and discounting",
        "Policies, value functions, and models"
      ],
      detailedExplanation: [
        "Key components of RL:",
        "- Agent: The learner and decision maker",
        "- Environment: Everything the agent interacts with",
        "- State: Representation of current situation",
        "- Action: Choices available to the agent",
        "- Reward: Immediate feedback signal",
        "",
        "MDP Formalism:",
        "- States (S): Set of possible situations",
        "- Actions (A): Set of possible moves",
        "- Transition function (P): P(s'|s,a)",
        "- Reward function (R): R(s,a,s')",
        "- Discount factor (γ): Future reward weighting",
        "",
        "Learning objectives:",
        "- Policy (π): Strategy for choosing actions",
        "- Value function (V/Q): Expected return",
        "- Model: Agent's representation of environment"
      ],
      code: {
        python: `# Basic RL Framework
import numpy as np

class Environment:
    def __init__(self):
        self.state = None
        self.action_space = ['up', 'down', 'left', 'right']
        
    def reset(self):
        self.state = (0, 0)  # Start position
        return self.state
        
    def step(self, action):
        x, y = self.state
        if action == 'up' and y < 10:
            y += 1
        elif action == 'down' and y > 0:
            y -= 1
        elif action == 'left' and x > 0:
            x -= 1
        elif action == 'right' and x < 10:
            x += 1
            
        self.state = (x, y)
        reward = 1 if (x, y) == (10, 10) else -0.1
        done = (x, y) == (10, 10)
        return self.state, reward, done, {}

class Agent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.q_table = {}
        
    def get_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon or state not in self.q_table:
            return np.random.choice(self.action_space)
        return max(self.q_table[state].items(), key=lambda x: x[1])[0]
        
    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.action_space}
        # Simple Q-learning update
        max_next_q = max(self.q_table[next_state].values()) if next_state in self.q_table else 0
        self.q_table[state][action] += 0.1 * (reward + 0.9 * max_next_q - self.q_table[state][action])

# Training loop
env = Environment()
agent = Agent(env.action_space)

for episode in range(1000):
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state`,
        complexity: "Q-learning: O(|S|×|A|) space, O(1) per update"
      }
    },
    {
      title: "⚙️ MDPs & Bellman Equations",
      id: "mdp",
      description: "Mathematical framework for modeling sequential decision problems.",
      keyPoints: [
        "Markov property: Future depends only on present",
        "Bellman equations: Recursive relationships",
        "Optimality principle: Decomposing problems",
        "Dynamic programming solutions"
      ],
      detailedExplanation: [
        "Markov Decision Process components:",
        "- S: Set of states with Markov property",
        "- A: Set of actions available",
        "- P: Transition probability matrix",
        "- R: Reward function",
        "- γ: Discount factor (0 ≤ γ ≤ 1)",
        "",
        "Bellman Equations:",
        "- Value function: V^π(s) = E[Σ γ^t r_t | s_0=s]",
        "- Q-function: Q^π(s,a) = E[Σ γ^t r_t | s_0=s, a_0=a]",
        "- Optimality equations:",
        "  V*(s) = max_a [R(s,a) + γΣ P(s'|s,a)V*(s')]",
        "  Q*(s,a) = R(s,a) + γΣ P(s'|s,a)max_a' Q*(s',a')",
        "",
        "Solution methods:",
        "- Value iteration: Repeated application of Bellman optimality",
        "- Policy iteration: Alternate policy evaluation and improvement",
        "- Linear programming: Formulate as optimization problem"
      ],
      code: {
        python: `# Solving MDPs with Dynamic Programming
import numpy as np

def value_iteration(mdp, theta=1e-6):
    """Value iteration algorithm"""
    V = {s: 0 for s in mdp.states}
    while True:
        delta = 0
        for s in mdp.states:
            v = V[s]
            # Bellman optimality update
            V[s] = max(
                sum(p*(mdp.reward(s,a,s') + mdp.gamma*V[s'])
                for a in mdp.actions(s)
                for (s', p) in mdp.transitions(s,a).items()
            )
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    
    # Extract optimal policy
    policy = {}
    for s in mdp.states:
        policy[s] = max(
            mdp.actions(s),
            key=lambda a: sum(
                p*(mdp.reward(s,a,s') + mdp.gamma*V[s'])
                for (s', p) in mdp.transitions(s,a).items()
            )
        )
    return policy, V

# Example GridWorld MDP
class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.gamma = 0.9
        self.states = [(i,j) for i in range(size) for j in range(size)]
        self.actions = ['up', 'down', 'left', 'right']
        
    def transitions(self, s, a):
        i, j = s
        transitions = {}
        
        # Deterministic transitions
        if a == 'up' and i > 0:
            transitions[(i-1, j)] = 1.0
        elif a == 'down' and i < self.size-1:
            transitions[(i+1, j)] = 1.0
        elif a == 'left' and j > 0:
            transitions[(i, j-1)] = 1.0
        elif a == 'right' and j < self.size-1:
            transitions[(i, j+1)] = 1.0
        else:
            transitions[s] = 1.0  # Hit wall, stay
            
        return transitions
        
    def reward(self, s, a, s_prime):
        # Reward +1 for reaching goal at (size-1, size-1)
        return 1.0 if s_prime == (self.size-1, self.size-1) else 0.0

# Solve the MDP
mdp = GridWorld()
optimal_policy, optimal_values = value_iteration(mdp)`,
        complexity: "Value iteration: O(|S|²|A|) per iteration"
      }
    },
    {
      title: "🔄 Q-Learning & TD Methods",
      id: "qlearning",
      description: "Model-free algorithms for learning from interaction.",
      keyPoints: [
        "Temporal Difference learning",
        "Q-learning: Off-policy TD control",
        "SARSA: On-policy TD control",
        "Exploration vs exploitation tradeoff"
      ],
      detailedExplanation: [
        "Temporal Difference Learning:",
        "- Learn directly from experience without model",
        "- Update estimates based on other estimates (bootstrapping)",
        "- Combines ideas from Monte Carlo and dynamic programming",
        "",
        "Q-Learning:",
        "- Off-policy: Learns optimal policy while following exploratory policy",
        "- Update rule: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]",
        "- Converges to optimal Q-function under certain conditions",
        "",
        "SARSA:",
        "- On-policy: Learns the policy being followed",
        "- Update rule: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]",
        "- Generally more conservative than Q-learning",
        "",
        "Exploration Strategies:",
        "- ε-greedy: Random action with probability ε",
        "- Boltzmann exploration: Action selection based on Q-values",
        "- Optimistic initialization: Encourage exploration of all states",
        "- Upper Confidence Bound (UCB): Balance exploration/exploitation"
      ],
      code: {
        python: `# Q-Learning Implementation
import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return self.actions[np.argmax(self.q_table[state])]
        
    def learn(self, state, action, reward, next_state, done):
        action_idx = self.actions.index(action)
        current_q = self.q_table[state][action_idx]
        
        # Q-learning update
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
            
        self.q_table[state][action_idx] += self.alpha * (target - current_q)

# Example usage with Gym environment
import gym

env = gym.make('FrozenLake-v1')
agent = QLearningAgent(list(range(env.action_space.n)))

num_episodes = 10000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state

# SARSA Implementation
class SARSAAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return self.actions[np.argmax(self.q_table[state])]
        
    def learn(self, state, action, reward, next_state, next_action, done):
        action_idx = self.actions.index(action)
        current_q = self.q_table[state][action_idx]
        
        # SARSA update
        if done:
            target = reward
        else:
            next_action_idx = self.actions.index(next_action)
            target = reward + self.gamma * self.q_table[next_state][next_action_idx]
            
        self.q_table[state][action_idx] += self.alpha * (target - current_q)`,
        complexity: "Q-learning/SARSA: O(1) per update, O(|S||A|) space"
      }
    },
    {
      title: "🧠 Policy Gradient Methods",
      id: "policygradient",
      description: "Directly optimizing policies using gradient ascent.",
      keyPoints: [
        "Policy parameterization (e.g., neural networks)",
        "Score function estimator",
        "REINFORCE algorithm",
        "Advantage estimation techniques"
      ],
      detailedExplanation: [
        "Policy Gradient Theorem:",
        "- ∇J(θ) ∝ E[∇log π(a|s) Q^π(s,a)]",
        "- Enables gradient-based optimization of policies",
        "- Works with continuous action spaces",
        "",
        "REINFORCE Algorithm:",
        "- Monte Carlo policy gradient",
        "- Uses complete episode returns",
        "- High variance but unbiased",
        "- Requires careful learning rate tuning",
        "",
        "Advantage Estimation:",
        "- Baseline subtraction reduces variance",
        "- Generalized Advantage Estimation (GAE)",
        "- Actor-Critic methods: Learn value function as baseline",
        "",
        "Modern Extensions:",
        "- Proximal Policy Optimization (PPO)",
        "- Trust Region Policy Optimization (TRPO)",
        "- Soft Actor-Critic (SAC)",
        "- Deterministic Policy Gradients (DDPG)"
      ],
      code: {
        python: `# Policy Gradient with REINFORCE
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

class REINFORCE:
    def __init__(self, state_size, action_size, lr=1e-2, gamma=0.99):
        self.policy = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.saved_log_probs = []
        self.rewards = []
        
    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
        
    def update_policy(self):
        R = 0
        policy_loss = []
        returns = []
        
        # Calculate discounted returns
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        del self.rewards[:]
        del self.saved_log_probs[:]

# Example usage with CartPole
import gym
env = gym.make('CartPole-v1')
agent = REINFORCE(env.observation_space.shape[0], env.action_space.n)

num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        agent.rewards.append(reward)
        
    agent.update_policy()

# Actor-Critic Implementation
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super().__init__()
        # Shared feature layer
        self.fc1 = nn.Linear(state_size, hidden_size)
        
        # Actor (policy) head
        self.fc_actor = nn.Linear(hidden_size, action_size)
        
        # Critic (value) head
        self.fc_critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        policy = torch.softmax(self.fc_actor(x), dim=-1)
        value = self.fc_critic(x)
        return policy, value.squeeze()`,
        complexity: "REINFORCE: O(T) per episode, Actor-Critic: O(1) per step"
      }
    },
    {
      title: "🎮 Deep RL & Applications",
      id: "deeprl",
      description: "Combining deep learning with reinforcement learning for complex problems.",
      keyPoints: [
        "Deep Q-Networks (DQN)",
        "Policy gradient with neural networks",
        "Experience replay and target networks",
        "Applications in games, robotics, and more"
      ],
      detailedExplanation: [
        "Deep Q-Networks:",
        "- Q-learning with neural network function approximation",
        "- Experience replay: Break temporal correlations",
        "- Target networks: Stabilize learning",
        "- Double DQN: Reduce overestimation bias",
        "",
        "Advanced Architectures:",
        "- Dueling Networks: Separate value and advantage streams",
        "- Prioritized Experience Replay: Important transitions",
        "- Rainbow: Combining multiple improvements",
        "",
        "Applications:",
        "- Game playing (AlphaGo, Atari, StarCraft)",
        "- Robotics control and manipulation",
        "- Autonomous vehicles",
        "- Resource management",
        "- Personalized recommendations",
        "",
        "Challenges:",
        "- Sample efficiency",
        "- Stability of training",
        "- Credit assignment",
        "- Exploration in high-dimensional spaces"
      ],
      code: {
        python: `# Deep Q-Network Implementation
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))
        
        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        # Compute loss and update
        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Proximal Policy Optimization (PPO)
class PPONetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, action_size)
        self.critic = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.actor(x), dim=-1), self.critic(x)

def ppo_update(policy, optimizer, samples, clip_ratio=0.2):
    states, actions, old_log_probs, returns, advantages = samples
    
    # Calculate new policy
    new_probs, new_values = policy(states)
    dist = Categorical(new_probs)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()
    
    # Probability ratio
    ratio = (new_log_probs - old_log_probs).exp()
    
    # Clipped objective
    clipped_ratio = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio)
    policy_loss = -torch.min(ratio*advantages, clipped_ratio*advantages).mean()
    
    # Value loss
    value_loss = 0.5 * (returns - new_values.squeeze()).pow(2).mean()
    
    # Total loss
    loss = policy_loss + 0.5*value_loss - 0.01*entropy
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()`,
        complexity: "DQN: O(batch_size) per update, PPO: O(batch_size) per epoch"
      }
    }
  ];

  return (
    <div style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '2rem',
      background: 'linear-gradient(to bottom right, #f0f9ff, #e0f2fe)',
      borderRadius: '20px',
      boxShadow: '0 10px 30px rgba(0,0,0,0.1)'
    }}>
      <h1 style={{
        fontSize: '3.5rem',
        fontWeight: '800',
        textAlign: 'center',
        background: 'linear-gradient(to right, #0369a1, #0ea5e9)',
        WebkitBackgroundClip: 'text',
        backgroundClip: 'text',
        color: 'transparent',
        marginBottom: '3rem'
      }}>
        Basics of Reinforcement Learning
      </h1>

      <div style={{
        backgroundColor: 'rgba(14, 165, 233, 0.1)',
        padding: '2rem',
        borderRadius: '12px',
        marginBottom: '3rem',
        borderLeft: '4px solid #0ea5e9'
      }}>
        <h2 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#0369a1',
          marginBottom: '1rem'
        }}>Reinforcement Learning → Basics</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Reinforcement learning is a computational approach to learning from interaction.
          This section covers the fundamental concepts and algorithms that enable agents
          to learn optimal behavior through trial-and-error interactions with environments.
        </p>
      </div>

      {content.map((section) => (
        <div
          key={section.id}
          style={{
            marginBottom: '3rem',
            padding: '2rem',
            backgroundColor: 'white',
            borderRadius: '16px',
            boxShadow: '0 5px 15px rgba(0,0,0,0.05)',
            transition: 'all 0.3s ease',
            border: '1px solid #bae6fd',
            ':hover': {
              boxShadow: '0 8px 25px rgba(0,0,0,0.1)',
              transform: 'translateY(-2px)'
            }
          }}
        >
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '1.5rem'
          }}>
            <h2 style={{
              fontSize: '2rem',
              fontWeight: '700',
              color: '#0369a1'
            }}>{section.title}</h2>
            <button
              onClick={() => toggleSection(section.id)}
              style={{
                background: 'linear-gradient(to right, #0369a1, #0ea5e9)',
                color: 'white',
                padding: '0.75rem 1.5rem',
                borderRadius: '8px',
                border: 'none',
                cursor: 'pointer',
                fontWeight: '600',
                transition: 'all 0.2s ease',
                ':hover': {
                  transform: 'scale(1.05)',
                  boxShadow: '0 5px 15px rgba(3, 105, 161, 0.4)'
                }
              }}
            >
              {visibleSection === section.id ? "Collapse Section" : "Expand Section"}
            </button>
          </div>

          {visibleSection === section.id && (
            <div style={{ display: 'grid', gap: '2rem' }}>
              <div style={{
                backgroundColor: '#f0f9ff',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#0369a1',
                  marginBottom: '1rem'
                }}>Core Concepts</h3>
                <p style={{
                  color: '#374151',
                  fontSize: '1.1rem',
                  lineHeight: '1.6',
                  marginBottom: '1rem'
                }}>
                  {section.description}
                </p>
                <ul style={{
                  listStyleType: 'disc',
                  paddingLeft: '1.5rem',
                  display: 'grid',
                  gap: '0.5rem'
                }}>
                  {section.keyPoints.map((point, index) => (
                    <li key={index} style={{
                      color: '#374151',
                      fontSize: '1.1rem'
                    }}>{point}</li>
                  ))}
                </ul>
              </div>

              <div style={{
                backgroundColor: '#e0f2fe',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#0369a1',
                  marginBottom: '1rem'
                }}>Technical Details</h3>
                <div style={{ display: 'grid', gap: '1rem' }}>
                  {section.detailedExplanation.map((paragraph, index) => (
                    <p key={index} style={{
                      color: '#374151',
                      fontSize: '1.1rem',
                      lineHeight: '1.6',
                      margin: paragraph === '' ? '0.5rem 0' : '0'
                    }}>
                      {paragraph}
                    </p>
                  ))}
                </div>
              </div>

              <div style={{
                backgroundColor: '#bae6fd',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#0369a1',
                  marginBottom: '1rem'
                }}>Implementation</h3>
                <div style={{
                  borderRadius: '8px',
                  overflow: 'hidden',
                  border: '2px solid #7dd3fc'
                }}>
                  <SyntaxHighlighter
                    language="python"
                    style={tomorrow}
                    customStyle={{
                      padding: "1.5rem",
                      fontSize: "0.95rem",
                      background: "#f9f9f9",
                      borderRadius: "0.5rem",
                    }}
                  >
                    {section.code.python}
                  </SyntaxHighlighter>
                </div>
              </div>
            </div>
          )}
        </div>
      ))}

      {/* Comparison Table */}
      <div style={{
        marginTop: '3rem',
        padding: '2rem',
        backgroundColor: 'white',
        borderRadius: '16px',
        boxShadow: '0 5px 15px rgba(0,0,0,0.05)',
        border: '1px solid #bae6fd'
      }}>
        <h2 style={{
          fontSize: '2rem',
          fontWeight: '700',
          color: '#0369a1',
          marginBottom: '2rem'
        }}>RL Algorithm Comparison</h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            textAlign: 'left'
          }}>
            <thead style={{
              backgroundColor: '#0369a1',
              color: 'white'
            }}>
              <tr>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Algorithm</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Type</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Strengths</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Use Cases</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Value Iteration", "Model-based", "Guaranteed optimality, simple", "Small discrete MDPs"],
                ["Q-Learning", "Model-free, off-policy", "Flexible, learns optimal policy", "Discrete action spaces"],
                ["SARSA", "Model-free, on-policy", "Safer learning, conservative", "Risk-sensitive domains"],
                ["REINFORCE", "Policy gradient", "Handles continuous actions", "Robotics, control"],
                ["PPO", "Policy gradient", "Stable, good performance", "Complex environments"]
              ].map((row, index) => (
                <tr key={index} style={{
                  backgroundColor: index % 2 === 0 ? '#f0f9ff' : 'white',
                  borderBottom: '1px solid #e2e8f0'
                }}>
                  {row.map((cell, cellIndex) => (
                    <td key={cellIndex} style={{
                      padding: '1rem',
                      color: '#334155'
                    }}>
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Key Takeaways */}
      <div style={{
        marginTop: '3rem',
        padding: '2rem',
        backgroundColor: '#f0f9ff',
        borderRadius: '16px',
        boxShadow: '0 5px 15px rgba(0,0,0,0.05)',
        border: '1px solid #bae6fd'
      }}>
        <h3 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#0369a1',
          marginBottom: '1.5rem'
        }}>RL Practitioner's Guide</h3>
        <div style={{ display: 'grid', gap: '1.5rem' }}>
          <div style={{
            backgroundColor: 'white',
            padding: '1.5rem',
            borderRadius: '12px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
          }}>
            <h4 style={{
              fontSize: '1.3rem',
              fontWeight: '600',
              color: '#0369a1',
              marginBottom: '0.75rem'
            }}>Algorithm Selection</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Discrete actions & small state space → Q-learning/SARSA
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Continuous actions → Policy gradient methods
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                High-dimensional states (images) → Deep RL
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Sample efficiency critical → Model-based RL
              </li>
            </ul>
          </div>
          
          <div style={{
            backgroundColor: 'white',
            padding: '1.5rem',
            borderRadius: '12px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
          }}>
            <h4 style={{
              fontSize: '1.3rem',
              fontWeight: '600',
              color: '#0369a1',
              marginBottom: '0.75rem'
            }}>Implementation Tips</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Hyperparameter Tuning:</strong> Learning rate most critical<br/>
              <strong>Exploration:</strong> Start high ε/τ, decay over time<br/>
              <strong>Stability:</strong> Use target networks in deep RL<br/>
              <strong>Debugging:</strong> Monitor average returns, Q-values
            </p>
          </div>

          <div style={{
            backgroundColor: 'white',
            padding: '1.5rem',
            borderRadius: '12px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
          }}>
            <h4 style={{
              fontSize: '1.3rem',
              fontWeight: '600',
              color: '#0369a1',
              marginBottom: '0.75rem'
            }}>Advanced Topics</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Hierarchical RL:</strong> Temporal abstraction<br/>
              <strong>Multi-Agent RL:</strong> Competitive/cooperative agents<br/>
              <strong>Inverse RL:</strong> Learn rewards from demonstrations<br/>
              <strong>Meta-RL:</strong> Learn to learn across tasks
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default BasicsOfRL;