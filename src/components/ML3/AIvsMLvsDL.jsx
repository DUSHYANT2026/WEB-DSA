import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";
import { useTheme } from "../../ThemeContext.jsx";

function AIvsMLvsDL() {
  const { darkMode } = useTheme();
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "🤖 Artificial Intelligence (AI)",
      id: "ai",
      description: "The broad discipline of creating intelligent machines capable of performing tasks that typically require human intelligence.",
      keyPoints: [
        "Encompasses all approaches to machine intelligence",
        "Includes both symbolic and sub-symbolic methods",
        "Goal: Create systems that can reason, learn, and act",
        "Applications: Robotics, NLP, expert systems, planning"
      ],
      detailedExplanation: [
        "AI Characteristics:",
        "- Reasoning and problem solving",
        "- Knowledge representation",
        "- Planning and decision making",
        "- Natural language processing",
        "- Perception (vision, speech)",
        "- Motion and manipulation",
        "",
        "Approaches:",
        "- Symbolic AI (rule-based systems)",
        "- Statistical methods",
        "- Computational intelligence",
        "- Machine learning (subset)",
        "",
        "Historical Milestones:",
        "- 1950: Turing Test proposed",
        "- 1956: Dartmouth Conference (AI founding)",
        "- 1997: Deep Blue beats chess champion",
        "- 2011: IBM Watson wins Jeopardy"
      ],
      code: {
        python: `# Simple Expert System (Rule-Based AI)
class MedicalDiagnosisSystem:
    def __init__(self):
        self.knowledge_base = {
            'flu': {'symptoms': ['fever', 'cough', 'fatigue']},
            'allergy': {'symptoms': ['sneezing', 'itchy eyes']},
            'migraine': {'symptoms': ['headache', 'nausea']}
        }
    
    def diagnose(self, symptoms):
        possible_conditions = []
        for condition, data in self.knowledge_base.items():
            if all(symptom in symptoms for symptom in data['symptoms']):
                possible_conditions.append(condition)
        return possible_conditions

# Usage
system = MedicalDiagnosisSystem()
print(system.diagnose(['fever', 'cough']))  # Output: ['flu']`,
        complexity: "Rule-based systems: O(n) where n is number of rules"
      }
    },
    {
      title: "📊 Machine Learning (ML)",
      id: "ml",
      description: "A subset of AI focused on developing algorithms that improve automatically through experience and data-driven pattern recognition.",
      keyPoints: [
        "Learns from data without explicit programming",
        "Three main types: Supervised, Unsupervised, Reinforcement",
        "Requires feature engineering",
        "Applications: Recommendation systems, fraud detection"
      ],
      detailedExplanation: [
        "ML Characteristics:",
        "- Data-driven pattern recognition",
        "- Improves with experience (data)",
        "- Generalizes from examples",
        "- Focuses on predictive accuracy",
        "",
        "Key Components:",
        "- Feature extraction/engineering",
        "- Model selection",
        "- Training process",
        "- Evaluation metrics",
        "",
        "Common Algorithms:",
        "- Linear Regression",
        "- Decision Trees",
        "- Support Vector Machines",
        "- Random Forests",
        "- k-Nearest Neighbors",
        "",
        "Workflow:",
        "1. Data collection and preprocessing",
        "2. Feature engineering",
        "3. Model training",
        "4. Evaluation",
        "5. Deployment"
      ],
      code: {
        python: `# Complete ML Pipeline Example
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
X, y = make_classification(n_samples=1000, n_features=20)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create pipeline
ml_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Train model
ml_pipeline.fit(X_train, y_train)

# Evaluate
predictions = ml_pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

# Feature importance
importances = ml_pipeline.named_steps['classifier'].feature_importances_
print("Feature importances:", importances)`,
        complexity: "Random Forest: O(m*n log n), where m=features, n=samples"
      }
    },
    {
      title: "🧠 Deep Learning (DL)",
      id: "dl",
      description: "A specialized subset of ML using hierarchical neural networks to model complex patterns in large datasets.",
      keyPoints: [
        "Uses artificial neural networks with multiple layers",
        "Automates feature extraction",
        "Excels with unstructured data (images, text, audio)",
        "Applications: Computer vision, speech recognition"
      ],
      detailedExplanation: [
        "DL Characteristics:",
        "- Multiple processing layers (deep architectures)",
        "- Automatic feature learning",
        "- Scalable with data and compute",
        "- State-of-the-art on many tasks",
        "",
        "Architectures:",
        "- Convolutional Neural Networks (CNNs)",
        "- Recurrent Neural Networks (RNNs)",
        "- Transformers",
        "- Autoencoders",
        "- Generative Adversarial Networks (GANs)",
        "",
        "Key Advances:",
        "- 2012: AlexNet breakthrough on ImageNet",
        "- 2014: GANs introduced",
        "- 2017: Transformer architecture",
        "- 2018: BERT for NLP",
        "- 2020: GPT-3 for generative tasks",
        "",
        "Implementation Considerations:",
        "- Requires large datasets",
        "- GPU/TPU acceleration essential",
        "- Hyperparameter tuning critical",
        "- Regularization techniques important"
      ],
      code: {
        python: `# Deep Learning with PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# Define neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Initialize model
model = NeuralNet(input_size=784, hidden_size=500, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# CNN Example
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x`,
        complexity: "Training: O(n × (k × d + m)), n=layers, k=kernel size, d=depth, m=parameters"
      }
    }
  ];

  return (
    <div style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '2rem',
      background: darkMode 
        ? 'linear-gradient(to bottom right, #1e293b, #0f172a)' 
        : 'linear-gradient(to bottom right, #f0f9ff, #f0fdf4)',
      borderRadius: '20px',
      boxShadow: darkMode ? '0 10px 30px rgba(0,0,0,0.3)' : '0 10px 30px rgba(0,0,0,0.1)',
      color: darkMode ? '#e2e8f0' : '#1e293b'
    }}>
      <h1 style={{
        fontSize: '3.5rem',
        fontWeight: '800',
        textAlign: 'center',
        background: 'linear-gradient(to right, #0ea5e9, #10b981)',
        WebkitBackgroundClip: 'text',
        backgroundClip: 'text',
        color: 'transparent',
        marginBottom: '3rem'
      }}>
        AI vs ML vs DL: Understanding the Differences
      </h1>

      <div style={{
        backgroundColor: darkMode ? 'rgba(14, 165, 233, 0.2)' : 'rgba(14, 165, 233, 0.1)',
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
        }}>Introduction to Machine Learning → Differences</h2>
        <p style={{
          color: darkMode ? '#e2e8f0' : '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL) are often used 
          interchangeably but represent distinct concepts with hierarchical relationships. 
          This section clarifies their differences and relationships.
        </p>
      </div>

      {content.map((section) => (
        <div
          key={section.id}
          style={{
            marginBottom: '3rem',
            padding: '2rem',
            backgroundColor: darkMode ? '#1e293b' : 'white',
            borderRadius: '16px',
            boxShadow: darkMode ? '0 5px 15px rgba(0,0,0,0.3)' : '0 5px 15px rgba(0,0,0,0.05)',
            transition: 'all 0.3s ease',
            border: darkMode ? '1px solid #334155' : '1px solid #e0f2fe',
            ':hover': {
              boxShadow: darkMode ? '0 8px 25px rgba(0,0,0,0.4)' : '0 8px 25px rgba(0,0,0,0.1)',
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
                background: 'linear-gradient(to right, #0ea5e9, #10b981)',
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
                backgroundColor: darkMode ? '#1e3a8a' : '#ecfdf5',
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
                  color: darkMode ? '#e2e8f0' : '#374151',
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
                      color: darkMode ? '#e2e8f0' : '#374151',
                      fontSize: '1.1rem'
                    }}>{point}</li>
                  ))}
                </ul>
              </div>

              <div style={{
                backgroundColor: darkMode ? '#164e63' : '#ecfeff',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#0ea5e9',
                  marginBottom: '1rem'
                }}>Technical Deep Dive</h3>
                <div style={{ display: 'grid', gap: '1rem' }}>
                  {section.detailedExplanation.map((paragraph, index) => (
                    <p key={index} style={{
                      color: darkMode ? '#e2e8f0' : '#374151',
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
                backgroundColor: darkMode ? '#064e3b' : '#f0fdfa',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#0ea5e9',
                  marginBottom: '1rem'
                }}>Implementation Example</h3>
                <p style={{
                  color: darkMode ? '#e2e8f0' : '#374151',
                  fontWeight: '600',
                  marginBottom: '1rem',
                  fontSize: '1.1rem'
                }}>{section.code.complexity}</p>
                <div style={{
                  borderRadius: '8px',
                  overflow: 'hidden',
                  border: darkMode ? '2px solid #0c4a6e' : '2px solid #a5f3fc'
                }}>
                  <SyntaxHighlighter
                    language="python"
                    style={tomorrow}
                    customStyle={{
                      padding: "1.5rem",
                      fontSize: "0.95rem",
                      background: darkMode ? "#1e293b" : "#f9f9f9",
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

      {/* Hierarchical Relationship */}
      <div style={{
        marginTop: '3rem',
        padding: '2rem',
        backgroundColor: darkMode ? '#1e293b' : 'white',
        borderRadius: '16px',
        boxShadow: darkMode ? '0 5px 15px rgba(0,0,0,0.3)' : '0 5px 15px rgba(0,0,0,0.05)',
        border: darkMode ? '1px solid #334155' : '1px solid #e0f2fe'
      }}>
        <h2 style={{
          fontSize: '2rem',
          fontWeight: '700',
          color: '#0ea5e9',
          marginBottom: '2rem'
        }}>Hierarchical Relationship</h2>
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          flexDirection: 'column',
          gap: '1rem'
        }}>
          <div style={{
            backgroundColor: '#0ea5e9',
            color: 'white',
            padding: '1.5rem 3rem',
            borderRadius: '8px',
            fontSize: '1.5rem',
            fontWeight: '700',
            textAlign: 'center',
            width: '300px'
          }}>
            Artificial Intelligence
          </div>
          <div style={{ fontSize: '2rem', color: '#0ea5e9' }}>↓</div>
          <div style={{
            backgroundColor: '#10b981',
            color: 'white',
            padding: '1.25rem 2.5rem',
            borderRadius: '8px',
            fontSize: '1.3rem',
            fontWeight: '700',
            textAlign: 'center',
            width: '250px'
          }}>
            Machine Learning
          </div>
          <div style={{ fontSize: '2rem', color: '#10b981' }}>↓</div>
          <div style={{
            backgroundColor: '#059669',
            color: 'white',
            padding: '1rem 2rem',
            borderRadius: '8px',
            fontSize: '1.1rem',
            fontWeight: '700',
            textAlign: 'center',
            width: '200px'
          }}>
            Deep Learning
          </div>
        </div>
        <p style={{
          color: darkMode ? '#e2e8f0' : '#374151',
          fontSize: '1.1rem',
          textAlign: 'center',
          marginTop: '2rem',
          lineHeight: '1.6'
        }}>
          Deep Learning is a specialized subset of Machine Learning, which itself is a subset of Artificial Intelligence.<br/>
          This hierarchy represents increasing specialization and technical complexity.
        </p>
      </div>

      {/* Comparative Analysis */}
      <div style={{
        marginTop: '3rem',
        padding: '2rem',
        backgroundColor: darkMode ? '#1e293b' : 'white',
        borderRadius: '16px',
        boxShadow: darkMode ? '0 5px 15px rgba(0,0,0,0.3)' : '0 5px 15px rgba(0,0,0,0.05)',
        border: darkMode ? '1px solid #334155' : '1px solid #e0f2fe'
      }}>
        <h2 style={{
          fontSize: '2rem',
          fontWeight: '700',
          color: '#0ea5e9',
          marginBottom: '2rem'
        }}>Comparative Analysis</h2>
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
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Characteristic</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>AI</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>ML</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>DL</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Scope", "Broadest (All intelligent systems)", "Subset of AI", "Subset of ML"],
                ["Data Dependency", "Rules-based or Data-driven", "Requires structured data", "Requires big data"],
                ["Hardware Needs", "Basic computing", "Medium resources", "GPUs/TPUs required"],
                ["Interpretability", "High (Rule-based)", "Moderate", "Low (Black box)"],
                ["Development Approach", "Symbolic logic + Learning", "Statistical learning", "Neural architectures"],
                ["Example Systems", "Expert systems, Chatbots", "Spam filters, Recommendation engines", "Self-driving cars, GPT models"]
              ].map((row, index) => (
                <tr key={index} style={{
                  backgroundColor: index % 2 === 0 
                    ? (darkMode ? '#334155' : '#f0fdf4') 
                    : (darkMode ? '#1e293b' : 'white'),
                  borderBottom: darkMode ? '1px solid #334155' : '1px solid #e2e8f0'
                }}>
                  {row.map((cell, cellIndex) => (
                    <td key={cellIndex} style={{
                      padding: '1rem',
                      color: darkMode ? '#e2e8f0' : '#334155'
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
        backgroundColor: darkMode ? '#1e3a8a' : '#ecfdf5',
        borderRadius: '16px',
        boxShadow: darkMode ? '0 5px 15px rgba(0,0,0,0.3)' : '0 5px 15px rgba(0,0,0,0.05)',
        border: darkMode ? '1px solid #1e40af' : '1px solid #d1fae5'
      }}>
        <h3 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#0ea5e9',
          marginBottom: '1.5rem'
        }}>Practical Implications</h3>
        <div style={{ display: 'grid', gap: '1.5rem' }}>
          <div style={{
            backgroundColor: darkMode ? '#1e293b' : 'white',
            padding: '1.5rem',
            borderRadius: '12px',
            boxShadow: darkMode ? '0 2px 8px rgba(0,0,0,0.3)' : '0 2px 8px rgba(0,0,0,0.05)'
          }}>
            <h4 style={{
              fontSize: '1.3rem',
              fontWeight: '600',
              color: '#0ea5e9',
              marginBottom: '0.75rem'
            }}>When to Use Each Approach</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ 
                color: darkMode ? '#e2e8f0' : '#374151', 
                fontSize: '1.1rem' 
              }}>
                <strong>AI:</strong> When explicit rules can solve the problem
              </li>
              <li style={{ 
                color: darkMode ? '#e2e8f0' : '#374151', 
                fontSize: '1.1rem' 
              }}>
                <strong>ML:</strong> When patterns exist in structured data
              </li>
              <li style={{ 
                color: darkMode ? '#e2e8f0' : '#374151', 
                fontSize: '1.1rem' 
              }}>
                <strong>DL:</strong> When dealing with unstructured data or complex patterns
              </li>
              <li style={{ 
                color: darkMode ? '#e2e8f0' : '#374151', 
                fontSize: '1.1rem' 
              }}>
                <strong>Hybrid:</strong> Often combine approaches for best results
              </li>
            </ul>
          </div>
          
          <div style={{
            backgroundColor: darkMode ? '#1e293b' : 'white',
            padding: '1.5rem',
            borderRadius: '12px',
            boxShadow: darkMode ? '0 2px 8px rgba(0,0,0,0.3)' : '0 2px 8px rgba(0,0,0,0.05)'
          }}>
            <h4 style={{
              fontSize: '1.3rem',
              fontWeight: '600',
              color: '#0ea5e9',
              marginBottom: '0.75rem'
            }}>Technology Evolution</h4>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
              gap: '1rem'
            }}>
              {[
                ["1950s-70s", "Symbolic AI and Expert Systems"],
                ["1980s-2000s", "Machine Learning foundations"],
                ["2010s", "Deep Learning revolution"],
                ["2020s", "Large Language Models"]
              ].map(([era, description], index) => (
                <div key={index} style={{
                  backgroundColor: darkMode ? '#334155' : '#f0fdf4',
                  padding: '1rem',
                  borderRadius: '8px',
                  borderLeft: '4px solid #10b981'
                }}>
                  <div style={{
                    fontWeight: '700',
                    color: '#059669',
                    marginBottom: '0.5rem'
                  }}>{era}</div>
                  <div style={{ 
                    color: darkMode ? '#e2e8f0' : '#374151' 
                  }}>{description}</div>
                </div>
              ))}
            </div>
          </div>

          <div style={{
            backgroundColor: darkMode ? '#1e293b' : 'white',
            padding: '1.5rem',
            borderRadius: '12px',
            boxShadow: darkMode ? '0 2px 8px rgba(0,0,0,0.3)' : '0 2px 8px rgba(0,0,0,0.05)'
          }}>
            <h4 style={{
              fontSize: '1.3rem',
              fontWeight: '600',
              color: '#0ea5e9',
              marginBottom: '0.75rem'
            }}>Application Spectrum</h4>
            <p style={{
              color: darkMode ? '#e2e8f0' : '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>AI:</strong> Comprehensive problem-solving across domains<br/>
              <strong>ML:</strong> Pattern recognition in structured data<br/>
              <strong>DL:</strong> Complex feature detection in unstructured data<br/>
              <br/>
              Most real-world systems combine elements of all three approaches.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AIvsMLvsDL;