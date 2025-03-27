import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function Calculus() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "📈 Differentiation and Partial Derivatives",
      id: "differentiation",
      description: "Fundamental tools for analyzing how functions change, essential for optimization in ML.",
      keyPoints: [
        "Derivatives measure instantaneous rate of change",
        "Partial derivatives for multivariate functions",
        "Gradient: Vector of partial derivatives",
        "Jacobian and Hessian matrices for higher-order derivatives"
      ],
      detailedExplanation: [
        "Key concepts in ML:",
        "- Gradient descent optimization relies on first derivatives",
        "- Second derivatives (Hessian) inform optimization curvature",
        "- Automatic differentiation enables backpropagation in neural networks",
        "",
        "Important applications:",
        "- Training neural networks via backpropagation",
        "- Optimization of loss functions",
        "- Sensitivity analysis of model parameters",
        "- Physics-informed machine learning",
        "",
        "Implementation considerations:",
        "- Numerical vs symbolic differentiation",
        "- Forward-mode vs reverse-mode autodiff",
        "- Gradient checking for verification",
        "- Handling non-differentiable functions"
      ],
      code: {
        python: `# Calculus in Machine Learning
import numpy as np
import torch

# Automatic differentiation example
x = torch.tensor(2.0, requires_grad=True)
y = x**3 + 2*x + 1
y.backward()
print(x.grad)  # dy/dx = 3x² + 2 → 14

# Partial derivatives
def f(x1, x2):
    return 3*x1**2 + 2*x1*x2 + x2**2

# Compute gradient numerically
def gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(*x_plus) - f(*x_minus)) / (2*eps)
    return grad

x = np.array([1.0, 2.0])
print(gradient(f, x))  # [10., 6.]

# Hessian matrix
def hessian(f, x, eps=1e-5):
    n = x.shape[0]
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x1 = x.copy()
            x1[i] += eps
            x1[j] += eps
            x2 = x.copy()
            x2[i] += eps
            x2[j] -= eps
            x3 = x.copy()
            x3[i] -= eps
            x3[j] += eps
            x4 = x.copy()
            x4[i] -= eps
            x4[j] -= eps
            hess[i,j] = (f(*x1)-f(*x2)-f(*x3)+f(*x4))/(4*eps*eps)
    return hess

print(hessian(f, x))`,
        complexity: "Gradient: O(n), Hessian: O(n²), Autodiff: O(1) per operation"
      }
    },
    {
      title: "⛓️ Chain Rule and Gradient Descent",
      id: "chain-rule",
      description: "The backbone of training neural networks through backpropagation.",
      keyPoints: [
        "Chain rule: Derivatives of composite functions",
        "Backpropagation: Efficient application of chain rule",
        "Stochastic gradient descent variants",
        "Learning rate and optimization strategies"
      ],
      detailedExplanation: [
        "How it powers ML:",
        "- Enables training of deep neural networks",
        "- Efficient computation of gradients through computational graphs",
        "- Forms basis for all modern deep learning frameworks",
        "",
        "Key components:",
        "- Forward pass: Compute loss function",
        "- Backward pass: Propagate errors backward",
        "- Parameter updates: Adjust weights using gradients",
        "",
        "Advanced topics:",
        "- Momentum and adaptive learning rates (Adam, RMSprop)",
        "- Second-order optimization methods",
        "- Gradient clipping for stability",
        "- Vanishing/exploding gradients in deep networks",
        "",
        "Practical considerations:",
        "- Batch size selection",
        "- Learning rate scheduling",
        "- Early stopping criteria",
        "- Gradient checking implementations"
      ],
      code: {
        python: `# Implementing Gradient Descent
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(X, W1, W2):
    h = sigmoid(X @ W1)
    y_hat = sigmoid(h @ W2)
    return y_hat, h

def backward(X, y, y_hat, h, W2):
    dL_dy = y_hat - y
    dL_dW2 = h.T @ dL_dy
    dL_dh = dL_dy @ W2.T
    dL_dW1 = X.T @ (dL_dh * h * (1 - h))
    return dL_dW1, dL_dW2

# Initialize parameters
X = np.random.randn(100, 3)  # 100 samples, 3 features
y = np.random.randint(0, 2, 100)  # Binary targets
W1 = np.random.randn(3, 4)  # First layer weights
W2 = np.random.randn(4, 1)  # Second layer weights
lr = 0.1

# Training loop
for epoch in range(1000):
    # Forward pass
    y_hat, h = forward(X, W1, W2)
    
    # Compute loss
    loss = -np.mean(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
    
    # Backward pass
    dW1, dW2 = backward(X, y, y_hat, h, W2)
    
    # Update weights
    W1 -= lr * dW1
    W2 -= lr * dW2
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Using PyTorch autograd
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(3, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.BCELoss()

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1,1)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()`,
        complexity: "Backpropagation: O(n) where n is number of operations in computational graph"
      }
    },
    {
      title: "➗ Taylor Series Expansion",
      id: "taylor",
      description: "Approximating complex functions with polynomials, useful for optimization and analysis.",
      keyPoints: [
        "Taylor series: Polynomial approximation around a point",
        "First-order approximation (linearization)",
        "Second-order approximation (quadratic)",
        "Applications in optimization and uncertainty"
      ],
      detailedExplanation: [
        "ML applications:",
        "- Understanding optimization surfaces",
        "- Newton's optimization method uses second-order expansion",
        "- Approximating non-linear activation functions",
        "- Analyzing model behavior around operating points",
        "",
        "Key concepts:",
        "- Maclaurin series (expansion around zero)",
        "- Remainder term and approximation error",
        "- Convergence conditions and radius",
        "- Multivariate Taylor expansion",
        "",
        "Practical uses:",
        "- Deriving optimization algorithms",
        "- Approximate inference methods",
        "- Sensitivity analysis",
        "- Explaining model predictions locally",
        "",
        "Advanced topics:",
        "- Taylor expansions in infinite-dimensional spaces",
        "- Applications in differential equations for ML",
        "- Taylor-mode automatic differentiation",
        "- Higher-order optimization methods"
      ],
      code: {
        python: `# Taylor Series in ML
import numpy as np
import matplotlib.pyplot as plt

def taylor_exp(x, n_terms=5):
    """Taylor series for e^x around 0"""
    result = 0
    for n in range(n_terms):
        result += x**n / np.math.factorial(n)
    return result

# Compare approximations
x = np.linspace(-2, 2, 100)
plt.plot(x, np.exp(x), label='Actual')
for n in [1, 2, 3, 5]:
    plt.plot(x, [taylor_exp(xi, n) for xi in x], label=f'{n} terms')
plt.legend()
plt.title('Taylor Series Approximation of e^x')
plt.show()

# Quadratic approximation for optimization
def quadratic_approx(f, x0, delta=1e-4):
    """Second-order Taylor approximation"""
    f0 = f(x0)
    grad = (f(x0+delta) - f(x0-delta)) / (2*delta)
    hess = (f(x0+delta) - 2*f0 + f(x0-delta)) / delta**2
    return lambda x: f0 + grad*(x-x0) + 0.5*hess*(x-x0)**2

# Example function
def f(x):
    return np.sin(x) + 0.1*x**2

# Find minimum using quadratic approximation
x0 = 1.0
q = quadratic_approx(f, x0)
minimum = x0 - q.__closure__[1].cell_contents/q.__closure__[2].cell_contents

# Multivariate Taylor expansion
def quadratic_approx_multi(f, x0, eps=1e-5):
    n = len(x0)
    grad = np.zeros(n)
    hess = np.zeros((n,n))
    
    # Gradient
    for i in range(n):
        x_plus = x0.copy()
        x_minus = x0.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2*eps)
    
    # Hessian
    for i in range(n):
        for j in range(n):
            x1 = x0.copy()
            x1[i] += eps
            x1[j] += eps
            x2 = x0.copy()
            x2[i] += eps
            x2[j] -= eps
            x3 = x0.copy()
            x3[i] -= eps
            x3[j] += eps
            x4 = x0.copy()
            x4[i] -= eps
            x4[j] -= eps
            hess[i,j] = (f(x1)-f(x2)-f(x3)+f(x4))/(4*eps*eps)
    
    f0 = f(x0)
    return lambda x: f0 + grad @ (x-x0) + 0.5*(x-x0) @ hess @ (x-x0)`,
        complexity: "Single-variable: O(n), Multivariate: O(n²) for Hessian"
      }
    }
  ];

  return (
    <div style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '2rem',
      background: 'linear-gradient(to bottom right, #fff7ed, #ffedd5)',
      borderRadius: '20px',
      boxShadow: '0 10px 30px rgba(0,0,0,0.1)'
    }}>
      <h1 style={{
        fontSize: '3.5rem',
        fontWeight: '800',
        textAlign: 'center',
        background: 'linear-gradient(to right, #ea580c, #d97706)',
        WebkitBackgroundClip: 'text',
        backgroundClip: 'text',
        color: 'transparent',
        marginBottom: '3rem'
      }}>
        Calculus for Machine Learning
      </h1>

      <div style={{
        backgroundColor: 'rgba(234, 88, 12, 0.1)',
        padding: '2rem',
        borderRadius: '12px',
        marginBottom: '3rem',
        borderLeft: '4px solid #ea580c'
      }}>
        <h2 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#ea580c',
          marginBottom: '1rem'
        }}>Mathematics for ML → Calculus</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Calculus provides the mathematical foundation for optimization and learning in machine learning. 
          This section covers the essential concepts with direct applications to ML models, including 
          differentiation, gradient descent, and function approximation.
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
            border: '1px solid #fed7aa',
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
              color: '#ea580c'
            }}>{section.title}</h2>
            <button
              onClick={() => toggleSection(section.id)}
              style={{
                background: 'linear-gradient(to right, #ea580c, #d97706)',
                color: 'white',
                padding: '0.75rem 1.5rem',
                borderRadius: '8px',
                border: 'none',
                cursor: 'pointer',
                fontWeight: '600',
                transition: 'all 0.2s ease',
                ':hover': {
                  transform: 'scale(1.05)',
                  boxShadow: '0 5px 15px rgba(234, 88, 12, 0.4)'
                }
              }}
            >
              {visibleSection === section.id ? "Collapse Section" : "Expand Section"}
            </button>
          </div>

          {visibleSection === section.id && (
            <div style={{ display: 'grid', gap: '2rem' }}>
              <div style={{
                backgroundColor: '#fffbeb',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#ea580c',
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
                backgroundColor: '#fef3c7',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#ea580c',
                  marginBottom: '1rem'
                }}>ML Applications</h3>
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
                backgroundColor: '#fef3c7',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#ea580c',
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
                  border: '2px solid #fdba74'
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
        border: '1px solid #fed7aa'
      }}>
        <h2 style={{
          fontSize: '2rem',
          fontWeight: '700',
          color: '#ea580c',
          marginBottom: '2rem'
        }}>Calculus Concepts in ML</h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            textAlign: 'left'
          }}>
            <thead style={{
              backgroundColor: '#ea580c',
              color: 'white'
            }}>
              <tr>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Concept</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>ML Application</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Example Use Case</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Key Libraries</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Differentiation", "Gradient computation", "Neural network training", "PyTorch, TensorFlow"],
                ["Chain Rule", "Backpropagation", "Deep learning", "Autograd, JAX"],
                ["Taylor Series", "Function approximation", "Optimization methods", "SciPy, NumPy"],
                ["Partial Derivatives", "Multivariate optimization", "Hyperparameter tuning", "Optuna, Scikit-learn"]
              ].map((row, index) => (
                <tr key={index} style={{
                  backgroundColor: index % 2 === 0 ? '#fff7ed' : 'white',
                  borderBottom: '1px solid #f3f4f6'
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
        backgroundColor: '#fffbeb',
        borderRadius: '16px',
        boxShadow: '0 5px 15px rgba(0,0,0,0.05)',
        border: '1px solid #fef3c7'
      }}>
        <h3 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#ea580c',
          marginBottom: '1.5rem'
        }}>ML Practitioner's Perspective</h3>
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
              color: '#ea580c',
              marginBottom: '0.75rem'
            }}>Essential Calculus for ML</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Gradients power all optimization-based learning algorithms
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                The chain rule enables efficient training of deep networks
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Taylor expansions help understand model behavior locally
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Partial derivatives handle multi-dimensional parameter spaces
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
              color: '#ea580c',
              marginBottom: '0.75rem'
            }}>Implementation Considerations</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Automatic Differentiation:</strong> Prefer over symbolic/numeric methods<br/>
              <strong>Gradient Checking:</strong> Validate implementations during development<br/>
              <strong>Numerical Stability:</strong> Handle vanishing/exploding gradients<br/>
              <strong>Second-Order Methods:</strong> Useful for small, critical models
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
              color: '#ea580c',
              marginBottom: '0.75rem'
            }}>Advanced Applications</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Neural ODEs:</strong> Continuous-depth models<br/>
              <strong>Physics-Informed ML:</strong> Incorporating domain knowledge<br/>
              <strong>Meta-Learning:</strong> Learning optimization processes<br/>
              <strong>Differentiable Programming:</strong> End-to-end differentiable systems
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Calculus;