import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function Regression() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "📉 Simple & Multiple Linear Regression",
      id: "linear",
      description: "Modeling linear relationships between variables, fundamental for predictive analytics.",
      keyPoints: [
        "y = β₀ + β₁x₁ + ... + βₙxₙ + ε",
        "Ordinary Least Squares (OLS) estimation",
        "Assumptions: Linearity, independence, homoscedasticity, normality",
        "Interpretation of coefficients"
      ],
      detailedExplanation: [
        "Mathematical Formulation:",
        "- Simple: y = β₀ + β₁x + ε (one predictor)",
        "- Multiple: y = β₀ + β₁x₁ + ... + βₙxₙ + ε (multiple predictors)",
        "- β₀: Intercept, β₁..βₙ: Slope coefficients",
        "- ε: Error term (normally distributed)",
        "",
        "Implementation Considerations:",
        "- Feature scaling improves convergence",
        "- Handling categorical predictors (dummy variables)",
        "- Multicollinearity detection (VIF)",
        "- Outlier impact and leverage points",
        "",
        "Evaluation Metrics:",
        "- R² (coefficient of determination)",
        "- Adjusted R² for multiple predictors",
        "- Mean Squared Error (MSE)",
        "- Root Mean Squared Error (RMSE)"
      ],
      code: {
        python: `# Linear Regression Implementation
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 3)  # 100 samples, 3 features
y = 2.5 + 1.5*X[:,0] + 0.8*X[:,1] - 1.2*X[:,2] + np.random.randn(100)*0.2

# Scikit-learn implementation
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Evaluate
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Intercept: {model.intercept_:.3f}")
print(f"Coefficients: {model.coef_}")
print(f"MSE: {mse:.3f}, R²: {r2:.3f}")

# Statsmodels for detailed statistics
X_with_const = sm.add_constant(X)  # Adds intercept term
sm_model = sm.OLS(y, X_with_const).fit()
print(sm_model.summary())

# Manual OLS implementation (educational)
def ols_fit(X, y):
    X_with_const = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ y
    return beta

manual_coef = ols_fit(X, y)
print("Manual coefficients:", manual_coef)`,
        complexity: "OLS: O(n²p + p³) where n=samples, p=features"
      }
    },
    {
      title: "📈 Polynomial Regression",
      id: "polynomial",
      description: "Extending linear models to capture nonlinear relationships through polynomial features.",
      keyPoints: [
        "y = β₀ + β₁x + β₂x² + ... + βₙxⁿ + ε",
        "Feature engineering with polynomial terms",
        "Degree selection and overfitting",
        "Regularization approaches"
      ],
      detailedExplanation: [
        "When to Use:",
        "- Nonlinear relationships between variables",
        "- Interaction effects between features",
        "- Approximation of complex functions",
        "",
        "Implementation Details:",
        "- PolynomialFeatures for feature transformation",
        "- Scaling becomes critical for higher degrees",
        "- Visualizing the fitted curve",
        "- Bias-variance tradeoff considerations",
        "",
        "Practical Considerations:",
        "- Degree selection via cross-validation",
        "- Regularization to prevent overfitting",
        "- Interpretation challenges with high degrees",
        "- Computational complexity with many features",
        "",
        "Applications:",
        "- Physical systems with known nonlinearities",
        "- Economic modeling (diminishing returns)",
        "- Biological growth curves",
        "- Sensor calibration"
      ],
      code: {
        python: `# Polynomial Regression Example
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Generate nonlinear data
np.random.seed(42)
X = np.linspace(-3, 3, 100)
y = 0.5*X**3 - 2*X**2 + X + np.random.randn(100)*0.5

# Reshape for sklearn
X = X.reshape(-1, 1)

# Create polynomial regression pipeline
degrees = [1, 3, 5, 7]
plt.figure(figsize=(10,6))

for degree in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    
    # Plot results
    plt.scatter(X, y, color='blue', alpha=0.3, label='Data' if degree==1 else None)
    plt.plot(X, y_pred, label=f'Degree {degree} (MSE: {mse:.2f})')
    
plt.title('Polynomial Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Regularized polynomial regression
from sklearn.linear_model import Ridge

degree = 5
model = Pipeline([
    ('poly', PolynomialFeatures(degree=degree)),
    ('ridge', Ridge(alpha=1.0))  # L2 regularization
])
model.fit(X, y)`,
        complexity: "O(d²n + d³) where d=degree, n=samples"
      }
    },
    {
      title: "⚖️ Ridge & Lasso Regression",
      id: "regularized",
      description: "Regularized linear models that prevent overfitting and perform feature selection.",
      keyPoints: [
        "Ridge (L2): Minimizes β₀ + Σ(yᵢ - ŷᵢ)² + λΣβⱼ²",
        "Lasso (L1): Minimizes β₀ + Σ(yᵢ - ŷᵢ)² + λΣ|βⱼ|",
        "Hyperparameter tuning (λ/α)",
        "Feature selection with Lasso"
      ],
      detailedExplanation: [
        "Ridge Regression:",
        "- Shrinks coefficients toward zero",
        "- Handles multicollinearity well",
        "- All features remain in the model",
        "- λ controls regularization strength",
        "",
        "Lasso Regression:",
        "- Can drive coefficients to exactly zero",
        "- Performs automatic feature selection",
        "- Useful for high-dimensional data",
        "- May struggle with correlated features",
        "",
        "Elastic Net:",
        "- Combines L1 and L2 penalties",
        "- Balance between Ridge and Lasso",
        "- Two hyperparameters to tune",
        "",
        "Implementation Guide:",
        "- Standardization is crucial",
        "- Cross-validation for λ selection",
        "- Path algorithms for efficient computation",
        "- Warm starts for hyperparameter search"
      ],
      code: {
        python: `# Regularized Regression Examples
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

# Generate data with redundant features
np.random.seed(42)
X = np.random.randn(100, 10)  # 10 features
y = 1.5*X[:,0] + 0.8*X[:,1] - 1.2*X[:,2] + np.random.randn(100)*0.5

# Ridge Regression
ridge = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)
ridge.fit(X, y)
print("Ridge coefficients:", ridge.named_steps['ridge'].coef_)

# Lasso Regression
lasso = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.1)
)
lasso.fit(X, y)
print("Lasso coefficients:", lasso.named_steps['lasso'].coef_)

# Elastic Net
elastic = make_pipeline(
    StandardScaler(),
    ElasticNet(alpha=0.1, l1_ratio=0.5)
)
elastic.fit(X, y)
print("Elastic Net coefficients:", elastic.named_steps['elasticnet'].coef_)

# Hyperparameter tuning
param_grid = {
    'ridge__alpha': np.logspace(-4, 4, 20)
}
grid = GridSearchCV(ridge, param_grid, cv=5)
grid.fit(X, y)
print("Best Ridge alpha:", grid.best_params_)

# Coefficient paths
from sklearn.linear_model import lasso_path

alphas, coefs, _ = lasso_path(X, y)
plt.figure(figsize=(10,6))
plt.plot(alphas, coefs.T)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficient value')
plt.title('Lasso Path')
plt.show()`,
        complexity: "Ridge: O(n²p + p³), Lasso: O(n²p) to O(n²p²)"
      }
    },
    {
      title: "📊 Evaluation Metrics",
      id: "metrics",
      description: "Quantitative measures to assess regression model performance.",
      keyPoints: [
        "Mean Absolute Error (MAE)",
        "Mean Squared Error (MSE)",
        "R² (coefficient of determination)",
        "Adjusted R², AIC, BIC"
      ],
      detailedExplanation: [
        "Error Metrics:",
        "- MAE: Robust to outliers, interpretable units",
        "- MSE: Emphasizes larger errors, differentiable",
        "- RMSE: Same units as target variable",
        "- MAPE: Percentage error interpretation",
        "",
        "Goodness-of-Fit Metrics:",
        "- R²: Proportion of variance explained",
        "- Adjusted R²: Penalizes extra predictors",
        "- AIC/BIC: Balance fit and complexity",
        "",
        "Diagnostic Checks:",
        "- Residual plots (patterns indicate problems)",
        "- Q-Q plots for normality assessment",
        "- Cook's distance for influential points",
        "- Durbin-Watson for autocorrelation",
        "",
        "Business Metrics:",
        "- Conversion to business KPIs",
        "- Error cost functions",
        "- Decision thresholds",
        "- ROI of model improvements"
      ],
      code: {
        python: `# Regression Evaluation Metrics
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

# Generate predictions
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

# Calculate metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)

print(f"MAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")
print(f"MAPE: {mape:.3f}")

# Adjusted R²
def adjusted_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

n_samples = len(y_true)
n_features = 3  # Number of predictors
adj_r2 = adjusted_r2(r2, n_samples, n_features)
print(f"Adjusted R²: {adj_r2:.3f}")

# Residual analysis
residuals = y_true - y_pred
plt.figure(figsize=(10,4))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Q-Q plot for normality
import statsmodels.api as sm
sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot of Residuals')
plt.show()`,
        complexity: "Metrics: O(n) where n=samples"
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
        Regression Techniques
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
        }}>Supervised Learning → Regression</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Regression models predict continuous outcomes by learning relationships between input features 
          and target variables. These techniques form the foundation of many predictive analytics 
          applications in machine learning.
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
                backgroundColor: '#ecf9ff',
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
        }}>Regression Techniques Comparison</h2>
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
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Best For</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Pros</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Cons</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Linear", "Linear relationships, interpretability", "Simple, fast, interpretable", "Limited to linear patterns"],
                ["Polynomial", "Nonlinear relationships", "Flexible, can fit complex patterns", "Prone to overfitting"],
                ["Ridge", "Multicollinearity, many features", "Stable with correlated features", "All features remain"],
                ["Lasso", "Feature selection, high dimensions", "Automatic feature selection", "Unstable with correlated features"],
                ["Elastic Net", "Balanced approach", "Combines Ridge and Lasso benefits", "Two hyperparameters to tune"]
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
        backgroundColor: '#ecf9ff',
        borderRadius: '16px',
        boxShadow: '0 5px 15px rgba(0,0,0,0.05)',
        border: '1px solid #bae6fd'
      }}>
        <h3 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#0369a1',
          marginBottom: '1.5rem'
        }}>Regression Best Practices</h3>
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
            }}>Model Selection Guide</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Start with simple linear regression as baseline
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Use polynomial terms when nonlinearity is suspected
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Apply regularization with many features (Ridge/Lasso)
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Consider Elastic Net when features are correlated
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
              <strong>Preprocessing:</strong> Scale features for regularized models<br/>
              <strong>Evaluation:</strong> Use multiple metrics and residual analysis<br/>
              <strong>Diagnostics:</strong> Check assumptions (linearity, normality)<br/>
              <strong>Deployment:</strong> Monitor for concept drift over time
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
            }}>Advanced Applications</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Bayesian Regression:</strong> Incorporating prior knowledge<br/>
              <strong>Quantile Regression:</strong> Modeling different percentiles<br/>
              <strong>Generalized Linear Models:</strong> Non-normal distributions<br/>
              <strong>Time Series Regression:</strong> Handling temporal dependencies
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Regression;