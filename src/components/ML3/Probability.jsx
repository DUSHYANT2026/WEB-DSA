import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function Probability() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "📊 Probability Distributions",
      id: "distributions",
      description: "Mathematical functions that describe the likelihood of different outcomes in ML contexts.",
      keyPoints: [
        "Normal (Gaussian) distribution: Bell curve for continuous data",
        "Binomial distribution: Discrete counts with fixed trials",
        "Poisson distribution: Modeling rare event counts",
        "Exponential distribution: Time between events"
      ],
      detailedExplanation: [
        "Key distributions in machine learning:",
        "- Normal: Used in Gaussian processes, noise modeling",
        "- Binomial: Binary classification outcomes",
        "- Poisson: Count data in NLP (word occurrences)",
        "- Exponential: Survival analysis, time-to-event data",
        "",
        "Distribution properties:",
        "- Parameters (μ, σ for Normal; λ for Poisson)",
        "- Moments (mean, variance, skewness, kurtosis)",
        "- Probability density/mass functions",
        "",
        "Applications in ML:",
        "- Assumptions in linear models (normality of errors)",
        "- Naive Bayes classifier distributions",
        "- Prior distributions in Bayesian methods",
        "- Noise modeling in probabilistic models"
      ],
      code: {
        python: `# Working with distributions in ML
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Normal distribution for modeling errors
error_dist = stats.norm(loc=0, scale=1)
samples = error_dist.rvs(1000)

# Binomial for classification confidence
n_trials = 100
p_success = 0.7
binom_dist = stats.binom(n_trials, p_success)
confidence = binom_dist.pmf(70)  # P(70 successes)

# Poisson for word counts in NLP
lambda_words = 5  # Average words per document
poisson_dist = stats.poisson(lambda_words)
word_prob = poisson_dist.pmf(3)  # P(3 words)

# Plotting distributions
x = np.linspace(0, 10, 100)
plt.plot(x, stats.norm.pdf(x, 2, 1), label='Normal')
plt.plot(x, stats.poisson.pmf(x, 3), label='Poisson')
plt.legend()
plt.title('ML Probability Distributions')
plt.show()`,
        complexity: "Sampling: O(1) per sample, PDF/PMF: O(1)"
      }
    },
    {
      title: "📉 Descriptive Statistics",
      id: "descriptive",
      description: "Measures that summarize important features of datasets in machine learning.",
      keyPoints: [
        "Central tendency: Mean, median, mode",
        "Dispersion: Variance, standard deviation, IQR",
        "Shape: Skewness, kurtosis",
        "Correlation: Pearson, Spearman coefficients"
      ],
      detailedExplanation: [
        "Essential statistics for ML:",
        "- Mean: Sensitive to outliers (use trimmed mean for robustness)",
        "- Median: Robust central value for skewed data",
        "- Variance: Measures spread of features",
        "- Correlation: Identifies feature relationships",
        "",
        "Data exploration with statistics:",
        "- Detecting outliers (z-scores, IQR method)",
        "- Feature scaling decisions (standard vs. robust scaling)",
        "- Identifying skewed features for transformation",
        "- Multicollinearity detection in regression",
        "",
        "Implementation considerations:",
        "- Numerical stability in calculations",
        "- Handling missing values",
        "- Weighted statistics for imbalanced data",
        "- Streaming/online computation for big data"
      ],
      code: {
        python: `# Descriptive statistics for ML datasets
import numpy as np
import pandas as pd
from scipy import stats

# Sample dataset
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 200],  # Contains outlier
    'income': [50000, 60000, 70000, 80000, 90000, 100000]
})

# Robust statistics
median = np.median(data['age'])
iqr = stats.iqr(data['age'])
trimmed_mean = stats.trim_mean(data['age'], 0.1)  # 10% trimmed

# Correlation analysis
pearson_corr = data.corr(method='pearson')
spearman_corr = data.corr(method='spearman')

# Outlier detection
z_scores = np.abs(stats.zscore(data))
outliers = (z_scores > 3).any(axis=1)

# Feature scaling info
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
robust_scale = (data - median) / iqr

print(f"Skewness: {stats.skew(data['income'])}")
print(f"Kurtosis: {stats.kurtosis(data['income'])}")`,
        complexity: "Basic stats: O(n), Correlation: O(n²), Outlier detection: O(n)"
      }
    },
    {
      title: "🔄 Bayes' Theorem",
      id: "bayes",
      description: "Fundamental rule for updating probabilities based on new evidence in ML models.",
      keyPoints: [
        "P(A|B) = P(B|A)P(A)/P(B)",
        "Prior, likelihood, and posterior probabilities",
        "Naive Bayes classifier assumptions",
        "Bayesian vs frequentist approaches"
      ],
      detailedExplanation: [
        "Bayesian machine learning:",
        "- Prior: Initial belief about parameters",
        "- Likelihood: Probability of data given parameters",
        "- Posterior: Updated belief after seeing data",
        "- Evidence: Marginal probability of data",
        "",
        "Applications in ML:",
        "- Naive Bayes for text classification",
        "- Bayesian networks for probabilistic reasoning",
        "- Bayesian optimization for hyperparameter tuning",
        "- Markov Chain Monte Carlo (MCMC) for inference",
        "",
        "Computational aspects:",
        "- Conjugate priors for analytical solutions",
        "- Approximate inference methods (Variational Bayes)",
        "- Probabilistic programming (PyMC3, Stan)",
        "- Bayesian neural networks"
      ],
      code: {
        python: `# Bayesian ML Examples
from sklearn.naive_bayes import GaussianNB
import pymc3 as pm
import numpy as np

# Naive Bayes Classifier
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
model = GaussianNB()
model.fit(X, y)

# Bayesian Inference with PyMC3
with pm.Model() as bayesian_model:
    # Priors
    mu = pm.Normal('mu', mu=0, sigma=1)
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    # Likelihood
    obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=np.random.randn(100))
    
    # Inference
    trace = pm.sample(1000, tune=1000)

# Bayesian Linear Regression
with pm.Model() as linear_model:
    # Priors
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    # Linear model
    mu = alpha + beta[0]*X[:,0] + beta[1]*X[:,1]
    
    # Likelihood
    likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
    
    # Inference
    trace = pm.sample(2000)`,
        complexity: "Naive Bayes: O(nd), MCMC: O(n²) or worse"
      }
    },
    {
      title: "📝 Hypothesis Testing",
      id: "testing",
      description: "Statistical methods for making data-driven decisions in ML model evaluation.",
      keyPoints: [
        "Null and alternative hypotheses",
        "p-values and significance levels",
        "Type I and Type II errors",
        "Common tests: t-test, ANOVA, chi-square"
      ],
      detailedExplanation: [
        "ML applications of hypothesis testing:",
        "- Feature selection (testing feature importance)",
        "- Model comparison (A/B testing different models)",
        "- Detecting data drift (testing distribution changes)",
        "- Evaluating treatment effects in causal ML",
        "",
        "Key concepts:",
        "- Test statistics and their distributions",
        "- Confidence intervals vs hypothesis tests",
        "- Multiple testing correction (Bonferroni, FDR)",
        "- Power analysis for test design",
        "",
        "Practical considerations:",
        "- Assumptions of tests (normality, independence)",
        "- Non-parametric alternatives (Mann-Whitney)",
        "- Bootstrap methods for complex cases",
        "- Bayesian hypothesis testing"
      ],
      code: {
        python: `# Hypothesis Testing in ML
from scipy import stats
import numpy as np
from statsmodels.stats.multitest import multipletests

# Compare two models' accuracies
model_a_scores = np.array([0.85, 0.82, 0.83, 0.86, 0.84])
model_b_scores = np.array([0.87, 0.89, 0.88, 0.86, 0.87])

# Paired t-test (same test set)
t_stat, p_value = stats.ttest_rel(model_a_scores, model_b_scores)

# Multiple feature tests
features = np.random.randn(100, 10)  # 100 samples, 10 features
target = np.random.randn(100)
p_values = [stats.pearsonr(features[:,i], target)[1] for i in range(10)]

# Correct for multiple testing
rejected, corrected_p, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

# ANOVA for multiple models
model_c_scores = np.array([0.88, 0.87, 0.89, 0.90, 0.88])
f_stat, anova_p = stats.f_oneway(model_a_scores, model_b_scores, model_c_scores)

# Bootstrap hypothesis test
def bootstrap_test(x, y, n_boot=10000):
    obs_diff = np.mean(x) - np.mean(y)
    combined = np.concatenate([x, y])
    bs_diffs = []
    for _ in range(n_boot):
        shuffled = np.random.permutation(combined)
        bs_diff = np.mean(shuffled[:len(x)]) - np.mean(shuffled[len(x):])
        bs_diffs.append(bs_diff)
    p_value = (np.abs(bs_diffs) >= np.abs(obs_diff)).mean()
    return p_value`,
        complexity: "t-tests: O(n), ANOVA: O(kn), Bootstrap: O(n_boot * n)"
      }
    },
    {
      title: "📏 Confidence Intervals",
      id: "intervals",
      description: "Range estimates that quantify uncertainty in ML model parameters and predictions.",
      keyPoints: [
        "Frequentist confidence intervals",
        "Bayesian credible intervals",
        "Bootstrap confidence intervals",
        "Interpretation and common misconceptions"
      ],
      detailedExplanation: [
        "Usage in machine learning:",
        "- Model parameter uncertainty (weight confidence)",
        "- Performance metric ranges (accuracy intervals)",
        "- Prediction intervals for probabilistic forecasts",
        "- Hyperparameter optimization uncertainty",
        "",
        "Construction methods:",
        "- Normal approximation (Wald intervals)",
        "- Student's t-distribution for small samples",
        "- Profile likelihood for complex models",
        "- Non-parametric bootstrap resampling",
        "",
        "Advanced topics:",
        "- Simultaneous confidence bands",
        "- Bayesian highest density intervals",
        "- Conformal prediction intervals",
        "- Uncertainty quantification in deep learning"
      ],
      code: {
        python: `# Confidence Intervals in ML
import numpy as np
from scipy import stats
from sklearn.utils import resample

# Linear regression confidence intervals
X = np.random.randn(100, 3)  # 100 samples, 3 features
y = 2*X[:,0] + 0.5*X[:,1] - X[:,2] + np.random.randn(100)
X_with_intercept = np.column_stack([np.ones(100), X])

# Calculate OLS parameters and confidence intervals
params = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
residuals = y - X_with_intercept @ params
sigma_squared = residuals.T @ residuals / (100 - 4)
param_cov = sigma_squared * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
std_errors = np.sqrt(np.diag(param_cov))
ci_lower = params - 1.96 * std_errors
ci_upper = params + 1.96 * std_errors

# Bootstrap confidence for model accuracy
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

y_true = np.random.randint(0, 2, 100)
y_pred = np.random.randint(0, 2, 100)
accuracies = [accuracy(*resample(y_true, y_pred)) for _ in range(1000)]
bootstrap_ci = np.percentile(accuracies, [2.5, 97.5])

# Bayesian credible interval (using PyMC3 trace)
# Assuming trace from previous Bayesian model
credible_interval = np.percentile(trace['mu'], [2.5, 97.5])

# Prediction intervals
def prediction_interval(X_new, X, y, alpha=0.05):
    n = len(X)
    X_with_intercept = np.column_stack([np.ones(n), X])
    params = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
    pred = params[0] + X_new @ params[1:]
    se = np.sqrt(np.sum((y - X_with_intercept @ params)**2) / (n - 2))
    t_val = stats.t.ppf(1 - alpha/2, n-2)
    return pred - t_val*se, pred + t_val*se`,
        complexity: "Analytical CIs: O(n²), Bootstrap: O(n_boot * n), Bayesian: depends on sampler"
      }
    }
  ];

  return (
    <div style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '2rem',
      background: 'linear-gradient(to bottom right, #f0f9ff, #f0fdf4)',
      borderRadius: '20px',
      boxShadow: '0 10px 30px rgba(0,0,0,0.1)'
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
        Probability & Statistics for Machine Learning
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
        }}>Mathematics for ML → Probability and Statistics</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Probability theory and statistics form the foundation for understanding uncertainty and making 
          data-driven decisions in machine learning. This section covers the essential concepts with 
          direct applications to ML models, including probability distributions, statistical inference, 
          and Bayesian methods.
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
            border: '1px solid #e0f2fe',
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
                backgroundColor: '#ecfdf5',
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
                backgroundColor: '#ecfeff',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#0ea5e9',
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
                backgroundColor: '#f0fdfa',
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
                  color: '#374151',
                  fontWeight: '600',
                  marginBottom: '1rem',
                  fontSize: '1.1rem'
                }}>{section.code.complexity}</p>
                <div style={{
                  borderRadius: '8px',
                  overflow: 'hidden',
                  border: '2px solid #a5f3fc'
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
        border: '1px solid #e0f2fe'
      }}>
        <h2 style={{
          fontSize: '2rem',
          fontWeight: '700',
          color: '#0ea5e9',
          marginBottom: '2rem'
        }}>Statistical Concepts in ML</h2>
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
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Concept</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>ML Application</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Example Use Case</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Key Libraries</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Probability Distributions", "Modeling uncertainty", "Naive Bayes classifiers", "scipy.stats, TensorFlow Probability"],
                ["Descriptive Statistics", "Data exploration", "Feature engineering", "numpy, pandas"],
                ["Bayes' Theorem", "Probabilistic modeling", "Bayesian networks", "PyMC3, Stan"],
                ["Hypothesis Testing", "Model evaluation", "Feature selection", "scipy.stats, statsmodels"],
                ["Confidence Intervals", "Uncertainty quantification", "Model performance reporting", "scipy, bootstrapped"]
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
        backgroundColor: '#fffbeb',
        borderRadius: '16px',
        boxShadow: '0 5px 15px rgba(0,0,0,0.05)',
        border: '1px solid #fef3c7'
      }}>
        <h3 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#0ea5e9',
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
              color: '#0ea5e9',
              marginBottom: '0.75rem'
            }}>Essential Statistics for ML</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Understanding distributions helps select appropriate models and loss functions
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Bayesian methods provide principled uncertainty quantification
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Hypothesis testing validates model improvements and feature importance
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Confidence intervals communicate model reliability to stakeholders
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
            }}>Practical Implementation Tips</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>For large datasets:</strong> Use streaming algorithms for statistics<br/>
              <strong>For high dimensions:</strong> Regularize covariance estimates<br/>
              <strong>For non-normal data:</strong> Apply appropriate transformations<br/>
              <strong>For production:</strong> Monitor statistical properties for drift
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
              <strong>Probabilistic Programming:</strong> Flexible Bayesian modeling<br/>
              <strong>Causal Inference:</strong> Understanding treatment effects<br/>
              <strong>Time Series:</strong> Modeling temporal dependencies<br/>
              <strong>Reinforcement Learning:</strong> Uncertainty-aware policies
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Probability;