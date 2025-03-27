import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function MDP() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "📜 MDP Fundamentals",
      id: "fundamentals",
      description: "The mathematical framework for modeling decision making in stochastic environments.",
      keyPoints: [
        "States (S): Possible situations of the environment",
        "Actions (A): Available choices at each state",
        "Transition Model (P): Probability of state transitions",
        "Reward Function (R): Immediate payoff for actions",
        "Discount Factor (γ): Importance of future rewards"
      ],
      detailedExplanation: [
        "Formal Definition:",
        "- MDP is a 5-tuple: (S, A, P, R, γ)",
        "- P(s'|s,a): Probability of reaching s' from s taking action a",
        "- R(s,a,s'): Reward received after transition",
        "- γ ∈ [0,1]: Discount factor balancing immediate vs future rewards",
        "",
        "Key Properties:",
        "- Markov Property: Future depends only on current state",
        "- Fully Observable: Agent knows current state",
        "- Stationary: Dynamics don't change over time",
        "- Discrete vs Continuous: Time and space variants",
        "",
        "Applications in RL:",
        "- Basis for most reinforcement learning algorithms",
        "- Used in robotics, game AI, operations research",
        "- Foundation for POMDPs (partially observable MDPs)",
        "- Extension to continuous spaces (continuous MDPs)"
      ],
      code: {
        python: `# MDP Implementation Example
import numpy as np

class MDP:
    def __init__(self, states, actions, transitions, rewards, gamma=0.9):
        self.states = states
        self.actions = actions
        self.transitions = transitions  # Dict: (s,a) -> {s': prob}
        self.rewards = rewards          # Dict: (s,a,s') -> reward
        self.gamma = gamma              # Discount factor
        
    def get_possible_actions(self, state):
        """Return possible actions at a state"""
        return self.actions[state]
        
    def get_transition_prob(self, state, action, next_state):
        """Get P(next_state|state,action)"""
        return self.transitions.get((state, action, next_state), 0.0)
        
    def get_reward(self, state, action, next_state):
        """Get immediate reward R(s,a,s')"""
        return self.rewards.get((state, action, next_state), 0.0)
        
    def is_terminal(self, state):
        """Check if state is terminal"""
        return state in self.terminal_states

# Example: Simple GridWorld
states = ['s0', 's1', 's2', 's3', 'terminal']
actions = {
    's0': ['right', 'down'],
    's1': ['left', 'down'],
    's2': ['right', 'up'],
    's3': ['left', 'up']
}

# Transition probabilities
transitions = {
    ('s0', 'right', 's1'): 1.0,
    ('s0', 'down', 's2'): 1.0,
    # ... other transitions
}

# Rewards
rewards = {
    ('s1', 'left', 's0'): 0,
    ('s3', 'up', 's1'): 10,  # Goal state
    # ... other rewards
}

gridworld = MDP(states, actions, transitions, rewards)`,
        complexity: "Tabular representation: O(|S|²|A|) space complexity"
      }
    },
    {
      title: "🔄 Value Iteration",
      id: "value-iteration",
      description: "Dynamic programming algorithm for computing optimal policies in MDPs.",
      keyPoints: [
        "Value Function (V): Expected return from each state",
        "Bellman Equation: Recursive definition of optimal value",
        "Iterative updates: Convergence to optimal values",
        "Policy extraction: Deriving actions from values"
      ],
      detailedExplanation: [
        "Algorithm Steps:",
        "1. Initialize value function V₀(s) arbitrarily",
        "2. Repeat until convergence:",
        "   Vₖ₊₁(s) ← maxₐ[Σₛ' P(s'|s,a)[R(s,a,s') + γVₖ(s')]]",
        "3. Extract policy: π(s) = argmaxₐ Q(s,a)",
        "",
        "Convergence Properties:",
        "- Guaranteed to converge to optimal V*",
        "- Error decreases exponentially",
        "- Stopping criterion: max|Vₖ₊₁ - Vₖ| < ε",
        "",
        "Practical Considerations:",
        "- Suitable for small, discrete MDPs",
        "- Can be slow for large state spaces",
        "- Parallelization opportunities",
        "- Asynchronous variants available",
        "",
        "Example Applications:",
        "- Inventory management",
        "- Robot path planning",
        "- Game strategy optimization",
        "- Financial decision making"
      ],
      code: {
        python: `# Value Iteration Implementation
def value_iteration(mdp, epsilon=0.001):
    """Solve MDP using value iteration"""
    V = {s: 0 for s in mdp.states}  # Initialize values
    
    while True:
        delta = 0
        new_V = {}
        
        for s in mdp.states:
            if mdp.is_terminal(s):
                new_V[s] = 0
                continue
                
            # Compute Q-values for all actions
            Q = {}
            for a in mdp.get_possible_actions(s):
                Q[a] = 0
                for s_prime in mdp.states:
                    prob = mdp.get_transition_prob(s, a, s_prime)
                    reward = mdp.get_reward(s, a, s_prime)
                    Q[a] += prob * (reward + mdp.gamma * V[s_prime])
            
            # Update value and track change
            new_V[s] = max(Q.values())
            delta = max(delta, abs(new_V[s] - V[s]))
            
        V = new_V
        if delta < epsilon:
            break
    
    # Extract optimal policy
    policy = {}
    for s in mdp.states:
        if mdp.is_terminal(s):
            policy[s] = None
            continue
            
        Q = {}
        for a in mdp.get_possible_actions(s):
            Q[a] = 0
            for s_prime in mdp.states:
                prob = mdp.get_transition_prob(s, a, s_prime)
                reward = mdp.get_reward(s, a, s_prime)
                Q[a] += prob * (reward + mdp.gamma * V[s_prime])
        
        policy[s] = max(Q.items(), key=lambda x: x[1])[0]
    
    return V, policy

# Run on our GridWorld
optimal_values, optimal_policy = value_iteration(gridworld)`,
        complexity: "O(|S|²|A|) per iteration, O(log(1/ε)) iterations needed"
      }
    },
    {
      title: "🛠️ Policy Iteration",
      id: "policy-iteration",
      description: "Alternative dynamic programming approach that alternates between policy evaluation and improvement.",
      keyPoints: [
        "Policy Evaluation: Compute value of current policy",
        "Policy Improvement: Greedily update policy",
        "Monotonic Improvement: Each iteration better or equal",
        "Faster convergence: Often fewer iterations than value iteration"
      ],
      detailedExplanation: [
        "Algorithm Steps:",
        "1. Initialize random policy π₀",
        "2. Repeat until convergence:",
        "   a. Policy Evaluation: Compute V^πₖ",
        "   b. Policy Improvement: πₖ₊₁(s) = argmaxₐ Q^πₖ(s,a)",
        "",
        "Policy Evaluation:",
        "- Solves linear system of Bellman equations",
        "- Can use iterative method (similar to value iteration)",
        "- Faster convergence when policy changes little",
        "",
        "Advantages:",
        "- Typically converges in fewer iterations",
        "- More stable policy updates",
        "- Exact solution for small problems",
        "- Clear stopping criterion",
        "",
        "Variants:",
        "- Modified Policy Iteration",
        "- Prioritized sweeping",
        "- Asynchronous policy iteration",
        "- Approximate policy iteration"
      ],
      code: {
        python: `# Policy Iteration Implementation
def policy_iteration(mdp, epsilon=0.001):
    """Solve MDP using policy iteration"""
    # Initialize random policy
    policy = {s: np.random.choice(mdp.get_possible_actions(s)) 
              if not mdp.is_terminal(s) else None 
              for s in mdp.states}
    
    while True:
        # Policy Evaluation
        V = {s: 0 for s in mdp.states}
        while True:
            delta = 0
            new_V = {}
            
            for s in mdp.states:
                if mdp.is_terminal(s):
                    new_V[s] = 0
                    continue
                    
                a = policy[s]
                new_V[s] = 0
                for s_prime in mdp.states:
                    prob = mdp.get_transition_prob(s, a, s_prime)
                    reward = mdp.get_reward(s, a, s_prime)
                    new_V[s] += prob * (reward + mdp.gamma * V[s_prime])
                
                delta = max(delta, abs(new_V[s] - V[s]))
            
            V = new_V
            if delta < epsilon:
                break
        
        # Policy Improvement
        policy_stable = True
        
        for s in mdp.states:
            if mdp.is_terminal(s):
                continue
                
            old_action = policy[s]
            Q = {}
            
            for a in mdp.get_possible_actions(s):
                Q[a] = 0
                for s_prime in mdp.states:
                    prob = mdp.get_transition_prob(s, a, s_prime)
                    reward = mdp.get_reward(s, a, s_prime)
                    Q[a] += prob * (reward + mdp.gamma * V[s_prime])
            
            best_action = max(Q.items(), key=lambda x: x[1])[0]
            policy[s] = best_action
            
            if old_action != best_action:
                policy_stable = False
        
        if policy_stable:
            break
    
    return V, policy

# Run on our GridWorld
values, policy = policy_iteration(gridworld)`,
        complexity: "Policy Evaluation: O(|S|²) per iteration, Policy Improvement: O(|S|²|A|)"
      }
    },
    {
      title: "🎲 Monte Carlo Methods",
      id: "monte-carlo",
      description: "Model-free learning from complete episodes of experience.",
      keyPoints: [
        "Episode-based: Learn from complete trajectories",
        "Model-free: No knowledge of transition dynamics needed",
        "First-visit vs Every-visit: Different update rules",
        "Exploration vs Exploitation: ε-greedy policies"
      ],
      detailedExplanation: [
        "Key Characteristics:",
        "- Requires complete episodes (must reach terminal state)",
        "- Estimates value functions by sample averages",
        "- High variance but unbiased estimates",
        "- Can work with only actual experience (no model needed)",
        "",
        "Algorithm Variants:",
        "- First-visit MC: Only first occurrence in episode counts",
        "- Every-visit MC: All occurrences count",
        "- Exploring starts: Ensure all (s,a) pairs visited",
        "- ε-greedy policies: Balance exploration/exploitation",
        "",
        "Advantages:",
        "- Simple to implement",
        "- Works without environment model",
        "- Can focus on important states",
        "- Naturally handles episodic tasks",
        "",
        "Challenges:",
        "- High variance in estimates",
        "- Must wait until episode completion",
        "- Exploration can be inefficient",
        "- Doesn't bootstrap (like TD methods)"
      ],
      code: {
        python: `# Monte Carlo Control Implementation
def monte_carlo_control(mdp, num_episodes=10000, epsilon=0.1, gamma=0.99):
    """Monte Carlo with ε-greedy policy"""
    # Initialize
    Q = defaultdict(lambda: np.zeros(len(mdp.actions)))
    returns = defaultdict(list)
    policy = {s: np.random.choice(mdp.actions) for s in mdp.states}
    
    for _ in range(num_episodes):
        # Generate episode
        episode = []
        state = mdp.reset()
        
        while True:
            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.choice(mdp.actions)
            else:
                action = policy[state]
                
            next_state, reward, done = mdp.step(action)
            episode.append((state, action, reward))
            state = next_state
            
            if done:
                break
        
        # Process episode
        G = 0
        visited = set()
        
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            # First-visit MC
            if (state, action) not in visited:
                visited.add((state, action))
                returns[(state, action)].append(G)
                Q[state][action] = np.mean(returns[(state, action)])
                
                # Update policy
                best_action = np.argmax(Q[state])
                policy[state] = best_action
    
    return policy, Q

# Example usage (assuming an environment class)
env = GridWorldEnv()
policy, Q = monte_carlo_control(env)`,
        complexity: "O(episode_length) per episode, depends on mixing time"
      }
    },
    {
      title: "⏱️ Temporal Difference Learning",
      id: "td-learning",
      description: "Combining ideas from Monte Carlo and dynamic programming for model-free learning.",
      keyPoints: [
        "TD(0): Simplest temporal difference algorithm",
        "Bootstrapping: Update estimates using other estimates",
        "SARSA: On-policy TD control",
        "Q-Learning: Off-policy TD control"
      ],
      detailedExplanation: [
        "Key Concepts:",
        "- Learn from incomplete episodes",
        "- Update estimates based on other estimates (bootstrapping)",
        "- Lower variance than Monte Carlo",
        "- Can learn optimal policy while following exploratory policy",
        "",
        "Algorithm Variants:",
        "- TD(0): One-step lookahead",
        "- SARSA: On-policy, learns action-value function",
        "- Q-Learning: Off-policy, learns optimal action-values",
        "- Expected SARSA: Like Q-learning but on-policy",
        "",
        "Advantages:",
        "- Online learning (no need to wait for episode end)",
        "- Lower variance than Monte Carlo",
        "- Proven convergence to optimal policy",
        "- Works in non-episodic tasks",
        "",
        "Practical Considerations:",
        "- Step size (learning rate) is crucial",
        "- Exploration strategy affects learning",
        "- Function approximation for large state spaces",
        "- Eligibility traces for multi-step updates"
      ],
      code: {
        python: `# Q-Learning Implementation
def q_learning(env, num_episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """Q-Learning: Off-policy TD control"""
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, done, _ = env.step(action)
            
            # Q-learning update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][best_next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            
            state = next_state
    
    # Derive policy
    policy = {s: np.argmax(Q[s]) for s in Q.keys()}
    return policy, Q

# SARSA Implementation
def sarsa(env, num_episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1):
    """SARSA: On-policy TD control"""
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        # Choose initial action
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        while not done:
            next_state, reward, done, _ = env.step(action)
            
            # Choose next action (on-policy)
            if np.random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])
            
            # SARSA update
            td_target = reward + gamma * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            
            state, action = next_state, next_action
    
    # Derive policy
    policy = {s: np.argmax(Q[s]) for s in Q.keys()}
    return policy, Q`,
        complexity: "O(1) per step, independent of state space size"
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
        Markov Decision Processes
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
        }}>Reinforcement Learning → MDP</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Markov Decision Processes provide the mathematical foundation for reinforcement learning,
          modeling sequential decision-making problems where outcomes are partly random and partly
          under the control of a decision maker.
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
        }}>MDP Solution Methods Comparison</h2>
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
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Method</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Model Required</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Convergence</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Best Use Case</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Value Iteration", "Yes", "Optimal", "Small discrete MDPs"],
                ["Policy Iteration", "Yes", "Optimal", "Medium discrete MDPs"],
                ["Monte Carlo", "No", "Optimal with GLIE", "Episodic tasks"],
                ["Temporal Difference", "No", "Optimal", "Continuing tasks"],
                ["Q-Learning", "No", "Optimal", "Off-policy learning"]
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
        }}>MDP Practitioner's Guide</h3>
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
            }}>Method Selection</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Use dynamic programming when transition model is available
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Prefer TD methods for continuing, non-episodic tasks
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Monte Carlo works well when episodes are naturally defined
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Q-learning is ideal for off-policy learning requirements
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
            }}>Practical Considerations</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Exploration:</strong> Balance with ε-greedy or softmax policies<br/>
              <strong>Discount Factor:</strong> Lower for myopic, higher for far-sighted agents<br/>
              <strong>Function Approximation:</strong> Necessary for large state spaces<br/>
              <strong>Hyperparameters:</strong> Tune learning rates and exploration schedules
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
              <strong>POMDPs:</strong> Partially observable extensions<br/>
              <strong>Hierarchical MDPs:</strong> Temporal abstraction<br/>
              <strong>Inverse RL:</strong> Learn rewards from demonstrations<br/>
              <strong>Multi-agent MDPs:</strong> Game theoretic approaches
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MDP;