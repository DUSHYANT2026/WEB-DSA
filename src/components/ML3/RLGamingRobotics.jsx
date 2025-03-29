import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function RLGamingRobotics() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "🎮 Game AI Applications",
      id: "gaming",
      description: "How reinforcement learning is revolutionizing game artificial intelligence.",
      keyPoints: [
        "NPC behavior learning and adaptation",
        "Procedural content generation",
        "Game balancing and testing",
        "Player modeling and personalization"
      ],
      detailedExplanation: [
        "Key Applications in Gaming:",
        "- Autonomous agents that learn from experience (AlphaStar, OpenAI Five)",
        "- Dynamic difficulty adjustment based on player skill",
        "- Automated playtesting and bug detection",
        "- Real-time strategy game AI that adapts to opponents",
        "",
        "Technical Approaches:",
        "- Deep Q-Learning for action selection",
        "- Policy gradient methods for complex strategies",
        "- Self-play for competitive environments",
        "- Imitation learning from human demonstrations",
        "",
        "Notable Examples:",
        "- AlphaGo/AlphaZero: Mastering board games",
        "- OpenAI Five: Defeating human teams in Dota 2",
        "- DeepMind's Capture The Flag: 3D navigation and teamwork",
        "- MineRL: Learning to play Minecraft"
      ],
      code: {
        python: `# Game AI with RL using Stable Baselines3
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create a vectorized environment
env = make_vec_env('LunarLander-v2', n_envs=4)

# Initialize PPO agent
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99
)

# Train the agent
model.learn(total_timesteps=1_000_000)

# Save the trained model
model.save("ppo_lunarlander")

# Load and test the trained model
del model  # remove to demonstrate loading
model = PPO.load("ppo_lunarlander")

obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()`,
        complexity: "Training: O(n) per episode, Inference: O(1) per step"
      }
    },
    {
      title: "🤖 Robotics Applications",
      id: "robotics",
      description: "Reinforcement learning for autonomous robot control and decision making.",
      keyPoints: [
        "Locomotion and motion control",
        "Manipulation and grasping",
        "Navigation and path planning",
        "Multi-robot coordination"
      ],
      detailedExplanation: [
        "Key Applications in Robotics:",
        "- Legged robot locomotion (Boston Dynamics-inspired)",
        "- Robotic arm control for precise manipulation",
        "- Autonomous vehicle navigation",
        "- Drone flight control and obstacle avoidance",
        "",
        "Technical Approaches:",
        "- Model-based RL for sample efficiency",
        "- Hierarchical RL for complex tasks",
        "- Sim-to-real transfer learning",
        "- Multi-agent RL for coordination",
        "",
        "Notable Examples:",
        "- OpenAI's Rubik's Cube solving robot hand",
        "- DeepMind's robotic soccer players",
        "- Boston Dynamics' adaptive locomotion",
        "- NVIDIA's autonomous warehouse robots",
        "",
        "Implementation Challenges:",
        "- Sample efficiency for real-world training",
        "- Safety constraints during exploration",
        "- Reward function design",
        "- Simulator accuracy for sim-to-real transfer"
      ],
      code: {
        python: `# Robotics RL with PyBullet and Stable Baselines3
import pybullet_envs
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment (PyBullet's Minitaur)
env = gym.make('MinitaurBulletEnv-v0')

# Initialize Soft Actor-Critic (good for robotics)
model = SAC(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=1_000_000,
    learning_starts=10_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    ent_coef='auto'
)

# Train the agent
model.learn(total_timesteps=500_000)

# Evaluate the trained policy
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} ± {std_reward}")

# Sim-to-real considerations
def sim_to_real_transfer(obs):
    """Add noise/delay to simulate real-world conditions"""
    obs = obs + np.random.normal(0, 0.01, obs.shape)  # sensor noise
    return obs  # Could also add latency simulation`,
        complexity: "Training: O(n²) for complex dynamics, Inference: O(1) per step"
      }
    },
    {
      title: "🔄 Sim-to-Real Transfer",
      id: "sim2real",
      description: "Bridging the gap between simulation training and real-world deployment.",
      keyPoints: [
        "Domain randomization techniques",
        "Reality gap modeling",
        "Adaptive policy transfer",
        "System identification methods"
      ],
      detailedExplanation: [
        "Key Challenges:",
        "- Simulation inaccuracies (physics, sensors)",
        "- Partial observability in real world",
        "- Latency and control delays",
        "- Unmodeled dynamics and disturbances",
        "",
        "Technical Solutions:",
        "- Domain randomization for robustness",
        "- System identification for model calibration",
        "- Meta-learning for fast adaptation",
        "- Adversarial training to bridge reality gap",
        "",
        "Implementation Patterns:",
        "- Progressive neural networks for transfer",
        "- Ensemble of simulators with varied parameters",
        "- Real-world fine-tuning with safe exploration",
        "- Residual learning for correcting simulator errors",
        "",
        "Case Studies:",
        "- OpenAI's robotic hand transfer from simulation",
        "- NVIDIA's drone control transfer",
        "- Google's quadruped locomotion transfer",
        "- MIT's robotic manipulation transfer"
      ],
      code: {
        python: `# Sim-to-Real with Domain Randomization
import numpy as np
from domain_randomization import DomainRandomizedEnv

# Base environment
base_env = gym.make('RobotArmEnv-v0')

# Create domain randomized version
randomized_env = DomainRandomizedEnv(
    base_env,
    randomize_friction=True,  # Random friction coefficients
    randomize_mass=True,      # Random link masses
    randomize_damping=True,   # Random joint damping
    randomize_gravity=True,   # Random gravity vector
    randomize_sensor_noise=True,  # Add sensor noise
    friction_range=(0.5, 1.5),
    mass_range=(0.8, 1.2),
    gravity_range=(9.6, 10.4)
)

# Train with randomized environment
model = PPO('MlpPolicy', randomized_env, verbose=1)
model.learn(total_timesteps=1_000_000)

# Adaptive policy for real-world deployment
class AdaptivePolicy:
    def __init__(self, trained_model):
        self.model = trained_model
        self.observation_buffer = []
        
    def predict(self, obs):
        # Add real-world observations to buffer
        self.observation_buffer.append(obs)
        if len(self.observation_buffer) > 1000:
            self.adapt_to_reality()
        return self.model.predict(obs)
    
    def adapt_to_reality(self):
        # Implement online adaptation logic here
        # Could use the observation buffer to fine-tune
        pass`,
        complexity: "Domain randomization: O(n), Online adaptation: O(n²)"
      }
    },
    {
      title: "🤝 Multi-Agent Systems",
      id: "multiagent",
      description: "Reinforcement learning for cooperative and competitive multi-agent scenarios.",
      keyPoints: [
        "Cooperative multi-agent RL",
        "Competitive/Adversarial learning",
        "Communication protocols",
        "Hierarchical multi-agent control"
      ],
      detailedExplanation: [
        "Gaming Applications:",
        "- Team-based game AI (MOBAs, RTS games)",
        "- NPC crowd behavior simulation",
        "- Dynamic opponent modeling",
        "- Emergent strategy development",
        "",
        "Robotics Applications:",
        "- Swarm robotics coordination",
        "- Multi-robot manipulation tasks",
        "- Fleet learning for autonomous vehicles",
        "- Distributed sensor networks",
        "",
        "Technical Approaches:",
        "- Centralized training with decentralized execution",
        "- Learning communication protocols",
        "- Opponent modeling and meta-learning",
        "- Credit assignment in cooperative tasks",
        "",
        "Notable Examples:",
        "- DeepMind's AlphaStar (Starcraft II)",
        "- OpenAI's Hide and Seek multi-agent environment",
        "- Google's robotic grasping with multiple arms",
        "- NVIDIA's drone swarm navigation"
      ],
      code: {
        python: `# Multi-Agent RL with RLlib
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

# Configure multi-agent environment
config = {
    "env": "MultiAgentPong",
    "num_workers": 4,
    "framework": "torch",
    "multiagent": {
        "policies": {
            "player1": (None, obs_space, act_space, {"gamma": 0.99}),
            "player2": (None, obs_space, act_space, {"gamma": 0.99}),
        },
        "policy_mapping_fn": lambda agent_id: "player1" if agent_id.startswith("p1") else "player2",
    },
    "model": {
        "fcnet_hiddens": [256, 256],
    },
}

# Train both agents simultaneously
tune.run(
    PPOTrainer,
    config=config,
    stop={"training_iteration": 100},
    checkpoint_at_end=True
)

# Cooperative multi-robot example
class CooperativeRobots:
    def __init__(self, num_robots):
        self.robots = [PPOTrainer(config) for _ in range(num_robots)]
        self.shared_memory = SharedMemoryBuffer()
        
    def train_cooperatively(self):
        # Implement centralized learning with decentralized execution
        # Robots share experiences through the memory buffer
        # Each robot learns from collective experiences
        pass`,
        complexity: "Training: O(n²) for n agents, Inference: O(n) per step"
      }
    }
  ];

  return (
    <div style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '2rem',
      background: 'linear-gradient(to bottom right, #ecfdf5, #f0fdf4)',
      borderRadius: '20px',
      boxShadow: '0 10px 30px rgba(0,0,0,0.1)'
    }}>
      <h1 style={{
        fontSize: '3.5rem',
        fontWeight: '800',
        textAlign: 'center',
        background: 'linear-gradient(to right, #0d9488, #14b8a6)',
        WebkitBackgroundClip: 'text',
        backgroundClip: 'text',
        color: 'transparent',
        marginBottom: '3rem'
      }}>
        RL for Gaming & Robotics
      </h1>

      <div style={{
        backgroundColor: 'rgba(13, 148, 136, 0.1)',
        padding: '2rem',
        borderRadius: '12px',
        marginBottom: '3rem',
        borderLeft: '4px solid #0d9488'
      }}>
        <h2 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#0d9488',
          marginBottom: '1rem'
        }}>Reinforcement Learning → Applications (Game AI, Robotics)</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Reinforcement learning has become a transformative technology for game AI and robotics,
          enabling systems to learn complex behaviors through interaction. This section covers
          practical applications and cutting-edge techniques in these domains.
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
            border: '1px solid #99f6e4',
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
              color: '#0d9488'
            }}>{section.title}</h2>
            <button
              onClick={() => toggleSection(section.id)}
              style={{
                background: 'linear-gradient(to right, #0d9488, #14b8a6)',
                color: 'white',
                padding: '0.75rem 1.5rem',
                borderRadius: '8px',
                border: 'none',
                cursor: 'pointer',
                fontWeight: '600',
                transition: 'all 0.2s ease',
                ':hover': {
                  transform: 'scale(1.05)',
                  boxShadow: '0 5px 15px rgba(13, 148, 136, 0.4)'
                }
              }}
            >
              {visibleSection === section.id ? "Collapse Section" : "Expand Section"}
            </button>
          </div>

          {visibleSection === section.id && (
            <div style={{ display: 'grid', gap: '2rem' }}>
              <div style={{
                backgroundColor: '#ccfbf1',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#0d9488',
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
                backgroundColor: '#99f6e4',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#0d9488',
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
                backgroundColor: '#5eead4',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#0d9488',
                  marginBottom: '1rem'
                }}>Implementation Example</h3>
                <p style={{
                  color: '#374151',
                  fontWeight: '600',
                  marginBottom: '1rem',
                  fontSize: '1.1rem'
                }}>{section.code.complexity}</p>
                <div style={{
                  borderRadius: '8px',
                  overflow: 'hidden',
                  border: '2px solid #14b8a6'
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

      {/* Case Studies */}
      <div style={{
        marginTop: '3rem',
        padding: '2rem',
        backgroundColor: 'white',
        borderRadius: '16px',
        boxShadow: '0 5px 15px rgba(0,0,0,0.05)',
        border: '1px solid #99f6e4'
      }}>
        <h2 style={{
          fontSize: '2rem',
          fontWeight: '700',
          color: '#0d9488',
          marginBottom: '2rem'
        }}>Notable Case Studies</h2>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
          gap: '1.5rem'
        }}>
          {[
            {
              title: "AlphaStar (Starcraft II)",
              description: "Mastered real-time strategy game at grandmaster level",
              techniques: "Self-play, imitation learning, neural architecture search"
            },
            {
              title: "OpenAI Robotic Hand",
              description: "Solved Rubik's Cube with dexterous manipulation",
              techniques: "Domain randomization, automatic domain randomization"
            },
            {
              title: "DeepMind Capture The Flag",
              description: "First-person shooter agents demonstrating teamwork",
              techniques: "Population-based training, reward shaping"
            },
            {
              title: "NVIDIA Drone Racing",
              description: "Autonomous drones racing through complex courses",
              techniques: "Sim-to-real transfer, reinforcement learning"
            },
            {
              title: "Boston Dynamics Locomotion",
              description: "Adaptive walking/running in dynamic environments",
              techniques: "Model-based RL, hierarchical control"
            },
            {
              title: "Google Robot Soccer",
              description: "Multi-robot coordination for soccer gameplay",
              techniques: "Multi-agent RL, decentralized execution"
            }
          ].map((caseStudy, index) => (
            <div key={index} style={{
              backgroundColor: '#ecfdf5',
              padding: '1.5rem',
              borderRadius: '12px',
              borderLeft: '4px solid #0d9488'
            }}>
              <h3 style={{
                fontSize: '1.3rem',
                fontWeight: '700',
                color: '#0d9488',
                marginBottom: '0.5rem'
              }}>{caseStudy.title}</h3>
              <p style={{ color: '#374151', marginBottom: '0.75rem' }}>{caseStudy.description}</p>
              <div style={{
                backgroundColor: '#ccfbf1',
                padding: '0.75rem',
                borderRadius: '8px'
              }}>
                <p style={{
                  color: '#0d9488',
                  fontWeight: '600',
                  fontSize: '0.9rem',
                  margin: 0
                }}>Key Techniques: {caseStudy.techniques}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Key Takeaways */}
      <div style={{
        marginTop: '3rem',
        padding: '2rem',
        backgroundColor: '#ccfbf1',
        borderRadius: '16px',
        boxShadow: '0 5px 15px rgba(0,0,0,0.05)',
        border: '1px solid #99f6e4'
      }}>
        <h3 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#0d9488',
          marginBottom: '1.5rem'
        }}>Practical Insights</h3>
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
              color: '#0d9488',
              marginBottom: '0.75rem'
            }}>Implementation Best Practices</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Start with simpler environments before scaling complexity
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Use curriculum learning to progressively increase difficulty
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Implement comprehensive monitoring and logging
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Design reward functions carefully to avoid unintended behaviors
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
              color: '#0d9488',
              marginBottom: '0.75rem'
            }}>Challenges and Solutions</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Sample Efficiency:</strong> Use model-based RL or demonstrations<br/>
              <strong>Exploration:</strong> Implement intrinsic motivation or curiosity<br/>
              <strong>Safety:</strong> Constrained RL or safe exploration techniques<br/>
              <strong>Transfer:</strong> Domain randomization and adaptation methods
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
              color: '#0d9488',
              marginBottom: '0.75rem'
            }}>Emerging Trends</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Foundation Models:</strong> General-purpose RL policies<br/>
              <strong>Meta-Learning:</strong> Rapid adaptation to new tasks<br/>
              <strong>Neuro-Symbolic:</strong> Combining RL with symbolic reasoning<br/>
              <strong>Human-in-the-Loop:</strong> Interactive RL with human feedback
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default RLGamingRobotics;