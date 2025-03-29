import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function BiasVarianceTradeoff() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "🎯 Understanding Bias and Variance",
      id: "concepts",
      description: "Fundamental concepts that determine model performance and generalization.",
      keyPoints: [
        "Bias: Error from overly simplistic assumptions",
        "Variance: Error from sensitivity to small fluctuations",
        "Irreducible error: Noise inherent in the data",
        "Total error = Bias² + Variance + Irreducible error"
      ],
      detailedExplanation: [
        "Bias (Underfitting):",
        "- High bias models are too simple for the data",
        "- Consistently miss relevant patterns",
        "- Examples: Linear regression for complex data",
        "- Symptoms: High training and test error",
        "",
        "Variance (Overfitting):",
        "- High variance models are too complex",
        "- Capture noise as if it were signal",
        "- Examples: Deep trees with no pruning",
        "- Symptoms: Low training error but high test error",
        "",
        "Visualizing the Tradeoff:",
        "- Simple models → high bias, low variance",
        "- Complex models → low bias, high variance",
        "- Goal: Find the sweet spot in the middle",
        "- Changes with model complexity and training size"
      ],
      code: {
        python: `# Visualizing Bias-Variance Tradeoff
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 0.5 * X[:,0] + np.random.randn(100) * 0.1  # Linear relationship with noise

# Create models of varying complexity
degrees = [1, 4, 15]
plt.figure(figsize=(14, 5))

for i, degree in enumerate(degrees):
    ax = plt.subplot(1, len(degrees), i + 1)
    
    # Polynomial regression
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X, y)
    X_test = np.linspace(0, 1, 100)
    y_pred = model.predict(X_test[:, np.newaxis])
    
    # Plot
    plt.scatter(X, y, s=10, label='Data')
    plt.plot(X_test, y_pred, color='r', label='Model')
    plt.title(f'Degree {degree}\nMSE: {mean_squared_error(y, model.predict(X)):.4f}')
    plt.ylim(-0.5, 1.5)
    plt.legend()

plt.suptitle('Bias-Variance Tradeoff Illustrated', y=1.02)
plt.tight_layout()
plt.show()`,
        complexity: "Analysis: O(n) for basic models, O(n²) for complex relationships"
      }
    },
    {
      title: "⚖️ The Tradeoff in Practice",
      id: "tradeoff",
      description: "How bias and variance manifest in real-world machine learning models.",
      keyPoints: [
        "Model complexity affects bias and variance inversely",
        "Training set size impacts variance more than bias",
        "Regularization balances bias and variance",
        "Different algorithms have different bias-variance profiles"
      ],
      detailedExplanation: [
        "Model Complexity Relationship:",
        "- Increasing complexity decreases bias but increases variance",
        "- There's an optimal complexity for each problem",
        "- Can be visualized with validation curves",
        "",
        "Training Data Considerations:",
        "- More data reduces variance without affecting bias",
        "- Small datasets are prone to high variance",
        "- Data quality affects irreducible error",
        "",
        "Algorithm Characteristics:",
        "- Linear models: High bias, low variance",
        "- Decision trees: Low bias, high variance",
        "- SVM with RBF kernel: Can tune bias-variance with γ",
        "- Neural networks: Can range based on architecture",
        "",
        "Practical Implications:",
        "- Simple problems need simpler models",
        "- Complex problems may require complex (but regularized) models",
        "- More data allows using more complex models",
        "- Domain knowledge helps choose appropriate bias"
      ],
      code: {
        python: `# Managing Bias-Variance Tradeoff
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Generate more complex synthetic data
X = np.random.rand(500, 1)
y = np.sin(2 * np.pi * X[:,0]) + np.random.randn(500) * 0.2

# Learning curves to diagnose bias-variance
def plot_learning_curve(estimator, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)
    
    plt.figure()
    plt.title(title)
    plt.plot(train_sizes, train_scores_mean, 'o-', label='Training error')
    plt.plot(train_sizes, test_scores_mean, 'o-', label='Validation error')
    plt.xlabel('Training examples')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid()

# High bias model (too simple)
plot_learning_curve(LinearRegression(), "High Bias Model")

# High variance model (too complex)
plot_learning_curve(RandomForestRegressor(n_estimators=200, max_depth=None), 
                   "High Variance Model")

# Well-balanced model
plot_learning_curve(RandomForestRegressor(n_estimators=100, max_depth=3), 
                   "Balanced Model")`,
        complexity: "Learning curves: O(k*n) where k is number of training sizes"
      }
    },
    {
      title: "🛠️ Techniques for Balancing",
      id: "techniques",
      description: "Practical methods to manage bias and variance in machine learning models.",
      keyPoints: [
        "Regularization (L1/L2) reduces variance",
        "Ensemble methods (bagging reduces variance, boosting reduces bias)",
        "Cross-validation for proper evaluation",
        "Feature engineering to address bias"
      ],
      detailedExplanation: [
        "Reducing Variance:",
        "- Regularization (L1/L2/ElasticNet)",
        "- Pruning decision trees",
        "- Dropout in neural networks",
        "- Early stopping",
        "",
        "Reducing Bias:",
        "- More complex models",
        "- Feature engineering",
        "- Boosting algorithms",
        "- Removing regularization",
        "",
        "Specialized Techniques:",
        "- Bagging (e.g., Random Forests) for variance reduction",
        "- Boosting (e.g., XGBoost) for bias reduction",
        "- Stacking for optimal balance",
        "- Dimensionality reduction for high-dimensional data",
        "",
        "Evaluation Methods:",
        "- Train-test splits to detect overfitting",
        "- Learning curves to diagnose issues",
        "- Validation curves to tune hyperparameters",
        "- Nested cross-validation for reliable estimates"
      ],
      code: {
        python: `# Techniques to Balance Bias and Variance
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# Generate data
X = np.random.rand(200, 10)
y = X[:,0] + 0.5 * X[:,1]**2 + np.random.randn(200) * 0.1

# 1. Regularization examples
ridge = Ridge(alpha=1.0)  # L2 regularization
lasso = Lasso(alpha=0.1)  # L1 regularization

print("Ridge CV MSE:", -cross_val_score(ridge, X, y, cv=5, 
                                      scoring='neg_mean_squared_error').mean())
print("Lasso CV MSE:", -cross_val_score(lasso, X, y, cv=5, 
                                      scoring='neg_mean_squared_error').mean())

# 2. Ensemble methods
# Bagging to reduce variance
bagging = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=5),
                          n_estimators=50)
# Boosting to reduce bias
boosting = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                   learning_rate=0.1)

print("Bagging CV MSE:", -cross_val_score(bagging, X, y, cv=5,
                                        scoring='neg_mean_squared_error').mean())
print("Boosting CV MSE:", -cross_val_score(boosting, X, y, cv=5,
                                         scoring='neg_mean_squared_error').mean())

# 3. Hyperparameter tuning with validation curve
from sklearn.model_selection import validation_curve

param_range = np.logspace(-3, 3, 10)
train_scores, test_scores = validation_curve(
    Ridge(), X, y, param_name="alpha", param_range=param_range,
    cv=5, scoring="neg_mean_squared_error")

# Plot validation curve
plt.figure()
plt.semilogx(param_range, -train_scores.mean(axis=1), 'o-', label='Training error')
plt.semilogx(param_range, -test_scores.mean(axis=1), 'o-', label='Validation error')
plt.xlabel('Regularization strength (alpha)')
plt.ylabel('MSE')
plt.title('Validation Curve for Ridge Regression')
plt.legend()
plt.grid()
plt.show()`,
        complexity: "Regularization: O(n), Ensembles: O(m*n) where m is number of estimators"
      }
    }
  ];

  return (
    <div style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '2rem',
      background: 'linear-gradient(to bottom right, #eff6ff, #e0f2fe)',
      borderRadius: '20px',
      boxShadow: '0 10px 30px rgba(0,0,0,0.1)'
    }}>
      <h1 style={{
        fontSize: '3.5rem',
        fontWeight: '800',
        textAlign: 'center',
        background: 'linear-gradient(to right, #1d4ed8, #3b82f6)',
        WebkitBackgroundClip: 'text',
        backgroundClip: 'text',
        color: 'transparent',
        marginBottom: '3rem'
      }}>
        Bias-Variance Tradeoff
      </h1>

      <div style={{
        backgroundColor: 'rgba(29, 78, 216, 0.1)',
        padding: '2rem',
        borderRadius: '12px',
        marginBottom: '3rem',
        borderLeft: '4px solid #1d4ed8'
      }}>
        <h2 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#1d4ed8',
          marginBottom: '1rem'
        }}>Model Evaluation and Optimization</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          The bias-variance tradeoff is a fundamental concept that helps explain model behavior
          and guides the selection and tuning of machine learning algorithms. Understanding this
          tradeoff is crucial for building models that generalize well to unseen data.
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
            border: '1px solid #bfdbfe',
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
              color: '#1d4ed8'
            }}>{section.title}</h2>
            <button
              onClick={() => toggleSection(section.id)}
              style={{
                background: 'linear-gradient(to right, #1d4ed8, #3b82f6)',
                color: 'white',
                padding: '0.75rem 1.5rem',
                borderRadius: '8px',
                border: 'none',
                cursor: 'pointer',
                fontWeight: '600',
                transition: 'all 0.2s ease',
                ':hover': {
                  transform: 'scale(1.05)',
                  boxShadow: '0 5px 15px rgba(29, 78, 216, 0.4)'
                }
              }}
            >
              {visibleSection === section.id ? "Collapse Section" : "Expand Section"}
            </button>
          </div>

          {visibleSection === section.id && (
            <div style={{ display: 'grid', gap: '2rem' }}>
              <div style={{
                backgroundColor: '#eff6ff',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#1d4ed8',
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
                  color: '#1d4ed8',
                  marginBottom: '1rem'
                }}>Technical Deep Dive</h3>
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
                backgroundColor: '#dbeafe',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#1d4ed8',
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
                  border: '2px solid #93c5fd'
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

      {/* Diagnostic Table */}
      <div style={{
        marginTop: '3rem',
        padding: '2rem',
        backgroundColor: 'white',
        borderRadius: '16px',
        boxShadow: '0 5px 15px rgba(0,0,0,0.05)',
        border: '1px solid #bfdbfe'
      }}>
        <h2 style={{
          fontSize: '2rem',
          fontWeight: '700',
          color: '#1d4ed8',
          marginBottom: '2rem'
        }}>Bias-Variance Diagnostics</h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            textAlign: 'left'
          }}>
            <thead style={{
              backgroundColor: '#1d4ed8',
              color: 'white'
            }}>
              <tr>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Symptom</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Training Error</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Validation Error</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Likely Issue</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Solution</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["High error on both", "High", "High", "High bias (underfitting)", "Increase model complexity, add features"],
                ["Large gap between train/val", "Low", "High", "High variance (overfitting)", "Regularization, more data, simplify model"],
                ["Good performance", "Low", "Low (close to train)", "Well-balanced", "None needed"],
                ["Error decreases with more data", "Decreasing", "Decreasing", "High variance", "Get more training data"],
                ["Error plateaus with more data", "Stable", "Stable", "High bias", "Change model architecture"]
              ].map((row, index) => (
                <tr key={index} style={{
                  backgroundColor: index % 2 === 0 ? '#f8fafc' : 'white',
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
        backgroundColor: '#eff6ff',
        borderRadius: '16px',
        boxShadow: '0 5px 15px rgba(0,0,0,0.05)',
        border: '1px solid #bfdbfe'
      }}>
        <h3 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#1d4ed8',
          marginBottom: '1.5rem'
        }}>Practical Guidelines</h3>
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
              color: '#1d4ed8',
              marginBottom: '0.75rem'
            }}>Model Selection Strategy</h4>
            <ol style={{
              listStyleType: 'decimal',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Start with a simple model to establish a baseline
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Gradually increase complexity while monitoring validation performance
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Stop when validation error stops improving or starts increasing
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Apply regularization if the model shows signs of overfitting
              </li>
            </ol>
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
              color: '#1d4ed8',
              marginBottom: '0.75rem'
            }}>Algorithm-Specific Tips</h4>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
              gap: '1rem'
            }}>
              {[
                ["Linear Models", "Control bias with feature engineering, variance with regularization"],
                ["Decision Trees", "Control variance with max_depth/min_samples_leaf"],
                ["Neural Networks", "Control variance with dropout/weight decay"],
                ["SVM", "Control bias-variance with C and kernel parameters"],
                ["Ensembles", "Bagging reduces variance, boosting reduces bias"]
              ].map(([algorithm, tip], index) => (
                <div key={index} style={{
                  backgroundColor: '#eff6ff',
                  padding: '1rem',
                  borderRadius: '8px'
                }}>
                  <h5 style={{
                    fontSize: '1.1rem',
                    fontWeight: '600',
                    color: '#1d4ed8',
                    marginBottom: '0.5rem'
                  }}>{algorithm}</h5>
                  <p style={{ color: '#374151' }}>{tip}</p>
                </div>
              ))}
            </div>
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
              color: '#1d4ed8',
              marginBottom: '0.75rem'
            }}>When to Collect More Data</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              More training data primarily helps with <strong>high variance</strong> problems:
              <br/><br/>
              - If your model performs well on training data but poorly on validation data<br/>
              - If learning curves show validation error decreasing with more data<br/>
              - For complex models that have the capacity to learn but need more examples<br/><br/>
              
              More data <strong>won't help</strong> with high bias problems - you need better features or a different model.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default BiasVarianceTradeoff;