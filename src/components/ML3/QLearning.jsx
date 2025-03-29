import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function QLearning() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "🧠 Q-Learning Fundamentals",
      id: "fundamentals",
      description: "A model-free reinforcement learning algorithm that learns the value of actions in particular states.",
      keyPoints: [
        "Off-policy temporal difference learning",
        "Q-table stores state-action values",
        "Bellman equation for value updates",
        "Exploration vs exploitation tradeoff"
      ],
      detailedExplanation: [
        "Core Concepts:",
        "- Q(s,a): Expected future reward for taking action a in state s",
        "- γ (gamma): Discount factor for future rewards",
        "- α (alpha): Learning rate for updates",
        "- ε (epsilon): Exploration rate",
        "",
        "Algorithm Steps:",
        "1. Initialize Q-table with zeros or random values",
        "2. Observe current state s",
        "3. Choose action a (using ε-greedy policy)",
        "4. Take action, observe reward r and new state s'",
        "5. Update Q(s,a) using Bellman equation",
        "6. Repeat until convergence or episode completion",
        "",
        "Key Properties:",
        "- Guaranteed to converge to optimal policy (given sufficient exploration)",
        "- Doesn't require environment model",
        "- Can handle stochastic environments",
        "- Tabular method (limited by state space size)"
      ],
      code: {
        python: `# Q-Learning Implementation
import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.alpha = 0.1  # learning rate
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
    
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size-1)  # explore
        return np.argmax(self.q_table[state])  # exploit
    
    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        
        # Bellman equation update
        new_q = current_q + self.alpha * (
            reward + self.gamma * max_next_q * (1 - done) - current_q
        )
        self.q_table[state, action] = new_q
        
        # Decay exploration rate
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Example usage
env = GridWorld()  # hypothetical environment
agent = QLearningAgent(env.state_size, env.action_size)

for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state`,
        complexity: "Time: O(n_episodes * n_steps), Space: O(n_states * n_actions)"
      }
    },
    {
      title: "⚙️ Deep Q-Networks (DQN)",
      id: "dqn",
      description: "Extension of Q-learning that uses neural networks to approximate the Q-function for large state spaces.",
      keyPoints: [
        "Q-function approximation with neural networks",
        "Experience replay for stability",
        "Target network to reduce correlation",
        "Handles high-dimensional state spaces"
      ],
      detailedExplanation: [
        "Key Innovations:",
        "- Replaces Q-table with neural network Q(s,a;θ)",
        "- Experience replay: Stores transitions (s,a,r,s') in memory",
        "- Target network: Separate network for stable Q-targets",
        "- Frame stacking for temporal information",
        "",
        "Training Process:",
        "1. Store experiences in replay buffer",
        "2. Sample random minibatch from buffer",
        "3. Compute target Q-values using target network",
        "4. Update main network via gradient descent",
        "5. Periodically update target network",
        "",
        "Advanced Variants:",
        "- Double DQN: Reduces overestimation bias",
        "- Dueling DQN: Separates value and advantage streams",
        "- Prioritized Experience Replay: Important transitions sampled more often",
        "- Rainbow: Combines multiple improvements",
        "",
        "Applications:",
        "- Atari game playing (original DQN application)",
        "- Robotics control",
        "- Autonomous systems",
        "- Resource management"
      ],
      code: {
        python: `# Deep Q-Network Implementation
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        model = tf.keras.Sequential([
            layers.Dense(24, input_dim=self.state_shape, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])
        
        targets = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay`,
        complexity: "Time: O(n_episodes * n_steps * training_time), Space: O(replay_buffer_size + model_size)"
      }
    },
    {
      title: "🔄 Policy Gradient Methods",
      id: "policy-gradients",
      description: "Alternative approach that directly optimizes the policy rather than learning value functions.",
      keyPoints: [
        "Directly parameterizes and optimizes policy",
        "Gradient ascent on expected reward",
        "Better for continuous action spaces",
        "Includes REINFORCE, Actor-Critic, PPO"
      ],
      detailedExplanation: [
        "Comparison with Q-Learning:",
        "- Q-learning: Learns value function, derives policy",
        "- Policy gradients: Learns policy directly",
        "- Generally higher variance but more flexible",
        "",
        "Key Algorithms:",
        "- REINFORCE: Monte Carlo policy gradient",
        "- Actor-Critic: Combines value and policy learning",
        "- A3C: Asynchronous advantage actor-critic",
        "- PPO: Proximal policy optimization (state-of-the-art)",
        "",
        "Advantages:",
        "- Naturally handles continuous action spaces",
        "- Can learn stochastic policies",
        "- Better convergence properties in some cases",
        "- More stable in certain environments",
        "",
        "Implementation Considerations:",
        "- Importance of baseline reduction",
        "- Trust region methods for stability",
        "- Parallel sampling for variance reduction",
        "- Entropy regularization for exploration"
      ],
      code: {
        python: `# REINFORCE Policy Gradient Implementation
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class PolicyGradientAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99  # discount factor
        self.learning_rate = 0.01
        self.states = []
        self.actions = []
        self.rewards = []
        self.model = self._build_model()
    
    def _build_model(self):
        model = tf.keras.Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='softmax')
        ])
        optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model
    
    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        probs = self.model.predict(state)[0]
        action = np.random.choice(self.action_size, p=probs)
        return action
    
    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def train(self):
        discounted_rewards = self._discount_rewards()
        
        states = np.vstack(self.states)
        actions = np.array(self.actions)
        
        # One-hot encode actions
        actions_one_hot = np.zeros([len(actions), self.action_size])
        actions_one_hot[np.arange(len(actions)), actions] = 1
        
        # Scale rewards
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        
        # Multiply actions by discounted rewards
        actions_one_hot *= discounted_rewards[:, None]
        
        # Train
        self.model.train_on_batch(states, actions_one_hot)
        
        # Reset episode memory
        self.states = []
        self.actions = []
        self.rewards = []
    
    def _discount_rewards(self):
        discounted = np.zeros_like(self.rewards)
        running_sum = 0
        for t in reversed(range(len(self.rewards))):
            running_sum = running_sum * self.gamma + self.rewards[t]
            discounted[t] = running_sum
        return discounted`,
        complexity: "Time: O(n_episodes * n_steps * training_time), Space: O(model_size + episode_memory)"
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
        Q-Learning and Reinforcement Learning
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
          color: '#0ea5e9',
          marginBottom: '1rem'
        }}>Reinforcement Learning → Q-Learning</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Q-Learning is a fundamental reinforcement learning algorithm that enables agents to learn optimal 
          policies through trial-and-error interactions with an environment. This section covers both 
          classical Q-Learning and its modern deep learning extensions.
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
              color: '#0ea5e9'
            }}>{section.title}</h2>
            <button
              onClick={() => toggleSection(section.id)}
              style={{
                background: 'linear-gradient(to right, #0ea5e9, #38bdf8)',
                color: 'white',
                padding: '0.75rem 1.5rem',
                borderRadius: '8px',
                border: 'none',
                cursor: 'pointer',
                fontWeight: '600',
                transition: 'all 0.2s ease',
                ':hover': {
                  transform: 'scale(1.05)',
                  boxShadow: '0 5px 15px rgba(14, 165, 233, 0.4)'
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
                  color: '#0ea5e9',
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
                  color: '#0ea5e9',
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
                  color: '#0ea5e9',
                  marginBottom: '1rem'
                }}>Implementation</h3>
                <p style={{
                  color: '#374151',
                  fontWeight: '600',
                  marginBottom: '1rem',
                  fontSize: '1.1rem'
                }}>{section.code.complexity}</p>
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
          color: '#0ea5e9',
          marginBottom: '2rem'
        }}>RL Algorithm Comparison</h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            textAlign: 'left'
          }}>
            <thead style={{
              backgroundColor: '#0ea5e9',
              color: 'white'
            }}>
              <tr>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Algorithm</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Type</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Strengths</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Limitations</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Q-Learning", "Value-based", "Simple, guaranteed convergence", "Discrete actions, small state spaces"],
                ["Deep Q-Network (DQN)", "Value-based", "Handles high-dim states", "Overestimation bias, sample inefficient"],
                ["Policy Gradients", "Policy-based", "Continuous actions, stochastic policies", "High variance, slow convergence"],
                ["Actor-Critic", "Hybrid", "Lower variance than pure policy gradients", "Complex to implement/tune"],
                ["PPO", "Policy-based", "Stable, good performance", "Many hyperparameters"]
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
          color: '#0ea5e9',
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
              color: '#0ea5e9',
              marginBottom: '0.75rem'
            }}>When to Use Q-Learning</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Discrete action spaces with limited possibilities
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Environments with small to medium state spaces
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Problems where exploration is straightforward
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                When you need interpretable value functions
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
              color: '#0ea5e9',
              marginBottom: '0.75rem'
            }}>Implementation Tips</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Hyperparameter Tuning:</strong> Start with α=0.1, γ=0.99, ε=1.0 (decay 0.995)<br/>
              <strong>Reward Shaping:</strong> Scale rewards to reasonable range (-1 to 1 works well)<br/>
              <strong>Exploration:</strong> Use ε-greedy or Boltzmann exploration<br/>
              <strong>Debugging:</strong> Monitor Q-value updates and reward progression
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
              color: '#0ea5e9',
              marginBottom: '0.75rem'
            }}>Advanced Applications</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Multi-Agent RL:</strong> Q-learning extensions for multiple agents<br/>
              <strong>Hierarchical RL:</strong> Combining Q-learning at different time scales<br/>
              <strong>Inverse RL:</strong> Learning reward functions from demonstrations<br/>
              <strong>Transfer Learning:</strong> Pre-trained Q-functions for new tasks
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default QLearning;