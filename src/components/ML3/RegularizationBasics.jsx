import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function RegularizationBasics() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "🛡️ L1 Regularization (Lasso)",
      id: "l1",
      description: "Adds absolute value of magnitude of coefficients as penalty term to the loss function.",
      keyPoints: [
        "Penalty term: λ∑|w| where w are model weights",
        "Produces sparse models (some weights become exactly zero)",
        "Useful for feature selection",
        "Robust to outliers"
      ],
      detailedExplanation: [
        "Mathematical Formulation:",
        "- Loss = Original Loss + λ∑|w|",
        "- λ controls regularization strength",
        "- Non-differentiable at zero (requires special handling)",
        "",
        "When to Use:",
        "- When you suspect many features are irrelevant",
        "- For models where interpretability is important",
        "- When working with high-dimensional data",
        "",
        "Implementation Considerations:",
        "- Requires subgradient methods or proximal operators",
        "- Coordinate descent works particularly well",
        "- Feature scaling is crucial",
        "- λ should be tuned via cross-validation"
      ],
      code: {
        python: `# L1 Regularization in Python
from sklearn.linear_model import Lasso
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 10)  # 100 samples, 10 features
y = 2*X[:,0] + 0.5*X[:,1] - X[:,2] + np.random.randn(100)  # Only 3 relevant features

# Lasso regression with different alpha (λ) values
lasso = Lasso(alpha=0.1)  # alpha is λ in sklearn
lasso.fit(X, y)

# Examine coefficients
print("Coefficients:", lasso.coef_)

# Cross-validated Lasso
from sklearn.linear_model import LassoCV
lasso_cv = LassoCV(cv=5).fit(X, y)
print("Optimal alpha:", lasso_cv.alpha_)
print("CV-selected coefficients:", lasso_cv.coef_)

# Implementing L1 manually with PyTorch
import torch
import torch.nn as nn

class L1RegularizedLinear(nn.Module):
    def __init__(self, input_dim, output_dim, l1_lambda=0.01):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.l1_lambda = l1_lambda
        
    def forward(self, x):
        return self.linear(x)
        
    def l1_loss(self):
        return self.l1_lambda * torch.sum(torch.abs(self.linear.weight))

model = L1RegularizedLinear(10, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop would include:
# loss = criterion(outputs, y) + model.l1_loss()`,
        complexity: "L1 adds O(d) computation where d is number of features"
      }
    },
    {
      title: "🛡️ L2 Regularization (Ridge)",
      id: "l2",
      description: "Adds squared magnitude of coefficients as penalty term to the loss function.",
      keyPoints: [
        "Penalty term: λ∑w² where w are model weights",
        "Shrinks coefficients but doesn't set them to zero",
        "Works well when features are correlated",
        "Has closed-form solution"
      ],
      detailedExplanation: [
        "Mathematical Formulation:",
        "- Loss = Original Loss + λ∑w²",
        "- λ controls regularization strength",
        "- Differentiable everywhere",
        "",
        "When to Use:",
        "- When you want to keep all features in the model",
        "- For dealing with multicollinearity",
        "- When features are all potentially relevant",
        "",
        "Implementation Considerations:",
        "- Has analytical solution: w = (XᵀX + λI)⁻¹Xᵀy",
        "- Numerically more stable than ordinary least squares",
        "- Works well with gradient descent",
        "- λ should be tuned via cross-validation",
        "",
        "Geometric Interpretation:",
        "- Constrains weights to lie within a hypersphere",
        "- Prevents any single weight from growing too large",
        "- Results in more distributed feature importance"
      ],
      code: {
        python: `# L2 Regularization in Python
from sklearn.linear_model import Ridge
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 10)
y = 2*X[:,0] + 0.5*X[:,1] - X[:,2] + np.random.randn(100)

# Ridge regression
ridge = Ridge(alpha=1.0)  # alpha is λ in sklearn
ridge.fit(X, y)

# Examine coefficients
print("Coefficients:", ridge.coef_)

# Cross-validated Ridge
from sklearn.linear_model import RidgeCV
ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5).fit(X, y)
print("Optimal alpha:", ridge_cv.alpha_)

# Implementing L2 manually with TensorFlow
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(10,),
    tf.keras.regularizers.l2(0.01)  # L2 regularization
])

model.compile(optimizer='sgd', loss='mse')
history = model.fit(X, y, epochs=100)

# Alternatively, explicit L2 loss
def l2_loss(model, lambda_=0.01):
    return lambda_ * tf.reduce_sum([tf.reduce_sum(w**2) for w in model.trainable_variables])

# Training loop would include:
# loss = mse_loss(y_true, y_pred) + l2_loss(model)`,
        complexity: "L2 adds O(d) computation where d is number of features"
      }
    },
    {
      title: "⚖️ Elastic Net",
      id: "elastic",
      description: "Combines L1 and L2 regularization to get benefits of both approaches.",
      keyPoints: [
        "Penalty term: λ₁∑|w| + λ₂∑w²",
        "Good compromise between L1 and L2",
        "Useful when there are multiple correlated features",
        "Can select groups of correlated features"
      ],
      detailedExplanation: [
        "Mathematical Formulation:",
        "- Loss = Original Loss + λ₁∑|w| + λ₂∑w²",
        "- λ₁ controls L1 strength, λ₂ controls L2 strength",
        "- Convex combination when λ₁ + λ₂ = 1",
        "",
        "When to Use:",
        "- When you have many correlated features",
        "- When you want some feature selection but not complete sparsity",
        "- For datasets where both L1 and L2 provide partial benefits",
        "",
        "Implementation Considerations:",
        "- Requires tuning two hyperparameters (can use ratio)",
        "- More computationally intensive than pure L1 or L2",
        "- sklearn uses l1_ratio = λ₁/(λ₁ + λ₂)",
        "- Works well with coordinate descent",
        "",
        "Practical Tips:",
        "- Start with l1_ratio around 0.5",
        "- Scale features before regularization",
        "- Use warm starts for hyperparameter tuning",
        "- Can help with very high-dimensional data"
      ],
      code: {
        python: `# Elastic Net in Python
from sklearn.linear_model import ElasticNet
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 10)
y = 2*X[:,0] + 0.5*X[:,1] - X[:,2] + np.random.randn(100)

# Elastic Net regression
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # alpha=λ₁+λ₂, l1_ratio=λ₁/(λ₁+λ₂)
elastic.fit(X, y)

# Examine coefficients
print("Coefficients:", elastic.coef_)

# Cross-validated Elastic Net
from sklearn.linear_model import ElasticNetCV
elastic_cv = ElasticNetCV(l1_ratio=[.1, .5, .9], cv=5).fit(X, y)
print("Optimal l1_ratio:", elastic_cv.l1_ratio_)
print("Optimal alpha:", elastic_cv.alpha_)

# Implementing Elastic Net manually
def elastic_net_loss(y_true, y_pred, model, l1_ratio=0.5, alpha=0.1):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    l1_loss = tf.reduce_sum([tf.reduce_sum(tf.abs(w)) for w in model.trainable_variables])
    l2_loss = tf.reduce_sum([tf.reduce_sum(tf.square(w)) for w in model.trainable_variables])
    return mse_loss + alpha * (l1_ratio * l1_loss + (1 - l1_ratio) * l2_loss)

# Usage in training:
# loss = elastic_net_loss(y_true, y_pred, model)`,
        complexity: "Elastic Net adds O(d) computation like L1/L2"
      }
    },
    {
      title: "🎯 Dropout",
      id: "dropout",
      description: "Randomly drops units from the neural network during training to prevent co-adaptation.",
      keyPoints: [
        "Randomly sets activations to zero during training",
        "Approximate way of training many thinned networks",
        "Works like an ensemble method",
        "Scale activations by 1/(1-p) at test time"
      ],
      detailedExplanation: [
        "How Dropout Works:",
        "- Each unit is dropped with probability p during training",
        "- Typically p=0.5 for hidden layers, p=0.2 for input layers",
        "- At test time, weights are scaled by 1-p",
        "- Can be viewed as model averaging",
        "",
        "When to Use:",
        "- For large neural networks with many parameters",
        "- When you observe overfitting in training",
        "- As a replacement for L2 regularization in deep learning",
        "- Particularly effective in computer vision",
        "",
        "Implementation Considerations:",
        "- Usually implemented as a layer in deep learning frameworks",
        "- Can be combined with other regularization techniques",
        "- Different dropout rates per layer often work best",
        "- Batch normalization changes dropout dynamics",
        "",
        "Advanced Variants:",
        "- Concrete Dropout: learns dropout rates automatically",
        "-Spatial Dropout: for convolutional networks",
        "- Weight Dropout: drops weights instead of activations",
        "- Alpha Dropout: for self-normalizing networks"
      ],
      code: {
        python: `# Dropout in Python
import tensorflow as tf
from tensorflow.keras.layers import Dropout
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(1000, 20)
y = (X[:,0] > 0).astype(int)  # Binary classification

# Model with Dropout
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    Dropout(0.5),  # 50% dropout
    tf.keras.layers.Dense(64, activation='relu'),
    Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X, y, epochs=10, validation_split=0.2)

# Implementing Dropout manually in PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, training=False):
        x = F.relu(self.fc1(x))
        if training:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if training:
            x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))

# During training:
# outputs = model(inputs, training=True)
# During evaluation:
# outputs = model(inputs, training=False)`,
        complexity: "Dropout adds minimal overhead during training (just masking)"
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
        Regularization Techniques
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
        }}>Model Evaluation and Optimization → Regularization Techniques</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Regularization methods prevent overfitting by adding constraints or penalties to model parameters,
          leading to better generalization on unseen data. These techniques are essential for building robust
          machine learning models.
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
                backgroundColor: '#e0f2fe',
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
                backgroundColor: '#f0f9ff',
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
                backgroundColor: '#ecfdf5',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#0369a1',
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
                  border: '2px solid #bae6fd'
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
        }}>Regularization Techniques Comparison</h2>
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
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Technique</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Best For</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Pros</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Cons</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["L1 (Lasso)", "Feature selection, sparse models", "Automatic feature selection, robust to outliers", "Can be unstable with correlated features"],
                ["L2 (Ridge)", "Correlated features, small datasets", "Stable solutions, works well with gradient descent", "Keeps all features (no sparsity)"],
                ["Elastic Net", "High-dimensional correlated data", "Combines L1/L2 benefits, selects groups of features", "Two parameters to tune"],
                ["Dropout", "Large neural networks", "Effective regularization, works like ensemble", "Increases training time"]
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
        backgroundColor: '#ecfdf5',
        borderRadius: '16px',
        boxShadow: '0 5px 15px rgba(0,0,0,0.05)',
        border: '1px solid #a7f3d0'
      }}>
        <h3 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#0369a1',
          marginBottom: '1.5rem'
        }}>Regularization Best Practices</h3>
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
            }}>Choosing the Right Technique</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Use L1 when you need feature selection or interpretability
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Prefer L2 for small datasets or correlated features
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Elastic Net offers a good compromise between L1 and L2
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Dropout is particularly effective for large neural networks
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
              <strong>Feature Scaling:</strong> Always standardize features before L1/L2 regularization<br/>
              <strong>Hyperparameter Tuning:</strong> Use cross-validation to find optimal λ values<br/>
              <strong>Early Stopping:</strong> Can be viewed as implicit regularization<br/>
              <strong>Combination:</strong> Often beneficial to combine multiple techniques
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
            }}>Advanced Considerations</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Adaptive Regularization:</strong> Layer-wise or parameter-wise λ values<br/>
              <strong>Structured Sparsity:</strong> Group lasso for structured feature selection<br/>
              <strong>Bayesian Approaches:</strong> Regularization through priors<br/>
              <strong>Curriculum Learning:</strong> Gradually increasing regularization strength
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default RegularizationBasics;