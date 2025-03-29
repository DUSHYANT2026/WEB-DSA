import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function EvaluationMetrics() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "📊 Classification Metrics",
      id: "classification",
      description: "Measures for evaluating the performance of classification models.",
      keyPoints: [
        "Accuracy: Overall correctness of predictions",
        "Precision: Quality of positive predictions",
        "Recall: Coverage of actual positives",
        "F1 Score: Harmonic mean of precision and recall",
        "ROC-AUC: Overall model discrimination ability"
      ],
      detailedExplanation: [
        "When to use which metric:",
        "- Accuracy: Balanced classes and equal error costs",
        "- Precision: When false positives are costly (e.g., spam filtering)",
        "- Recall: When false negatives are costly (e.g., medical diagnosis)",
        "- F1: When seeking balance between precision and recall",
        "- ROC-AUC: Comparing models overall performance",
        "",
        "Advanced metrics:",
        "- Cohen's Kappa: Agreement accounting for chance",
        "- Matthews Correlation Coefficient (MCC): Balanced measure for binary classes",
        "- Log Loss: Probabilistic confidence assessment",
        "- Brier Score: Calibration of probability estimates"
      ],
      code: {
        python: `# Classification Metrics in Python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# True labels and predictions
y_true = [0, 1, 0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 1, 1, 0, 0, 1, 0]
y_proba = [0.1, 0.9, 0.4, 0.8, 0.3, 0.2, 0.7, 0.1]  # Predicted probabilities

# Basic metrics
print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
print(f"Precision: {precision_score(y_true, y_pred):.2f}")
print(f"Recall: {recall_score(y_true, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_true, y_pred):.2f}")
print(f"ROC AUC: {roc_auc_score(y_true, y_proba):.2f}")

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Advanced metrics
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, log_loss
print(f"\nCohen's Kappa: {cohen_kappa_score(y_true, y_pred):.2f}")
print(f"MCC: {matthews_corrcoef(y_true, y_pred):.2f}")
print(f"Log Loss: {log_loss(y_true, y_proba):.2f}")`,
        complexity: "All metrics: O(n) where n is number of samples"
      }
    },
    {
      title: "📈 Regression Metrics",
      id: "regression",
      description: "Measures for evaluating the performance of regression models.",
      keyPoints: [
        "MAE: Mean Absolute Error",
        "MSE: Mean Squared Error",
        "RMSE: Root Mean Squared Error",
        "R²: Coefficient of Determination",
        "MAPE: Mean Absolute Percentage Error"
      ],
      detailedExplanation: [
        "When to use which metric:",
        "- MAE: Interpretable, robust to outliers",
        "- MSE/RMSE: Emphasizes larger errors (sensitive to outliers)",
        "- R²: Comparing models on same dataset",
        "- MAPE: Relative error interpretation",
        "",
        "Advanced metrics:",
        "- MSLE: Mean Squared Log Error (relative errors)",
        "- RMSLE: Root Mean Squared Log Error",
        "- Quantile Loss: Asymmetric cost functions",
        "- Explained Variance: Proportion of variance explained",
        "",
        "Special considerations:",
        "- Scaling: Some metrics are scale-dependent",
        "- Interpretation: Units of measurement matter",
        "- Outliers: Impact varies by metric",
        "- Probabilistic: Metrics for uncertainty estimation"
      ],
      code: {
        python: `# Regression Metrics in Python
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, 
    r2_score, mean_absolute_percentage_error,
    mean_squared_log_error
)
import numpy as np

# True values and predictions
y_true = [3.5, 2.0, 4.0, 5.5, 4.2]
y_pred = [3.0, 2.5, 3.5, 5.0, 4.0]

# Basic metrics
print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_true, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
print(f"R²: {r2_score(y_true, y_pred):.2f}")
print(f"MAPE: {mean_absolute_percentage_error(y_true, y_pred):.2f}")

# Advanced metrics
print(f"MSLE: {mean_squared_log_error(y_true, y_pred):.2f}")
print(f"RMSLE: {np.sqrt(mean_squared_log_error(y_true, y_pred)):.2f}")

# Quantile loss
def quantile_loss(y_true, y_pred, q=0.5):
    e = y_true - y_pred
    return np.mean(np.maximum(q * e, (q - 1) * e))

print(f"0.9 Quantile Loss: {quantile_loss(y_true, y_pred, q=0.9):.2f}")

# Explained variance
from sklearn.metrics import explained_variance_score
print(f"Explained Variance: {explained_variance_score(y_true, y_pred):.2f}")`,
        complexity: "All metrics: O(n) where n is number of samples"
      }
    },
    {
      title: "📊 Model Comparison & Selection",
      id: "comparison",
      description: "Techniques for comparing models and selecting the best performing one.",
      keyPoints: [
        "Cross-validation: k-fold, stratified, time-series",
        "Statistical tests: t-tests, ANOVA",
        "Bayesian comparison methods",
        "Multiple hypothesis testing correction"
      ],
      detailedExplanation: [
        "Cross-validation strategies:",
        "- k-fold: Standard approach for i.i.d. data",
        "- Stratified: Preserves class distribution",
        "- Time-series: Maintains temporal ordering",
        "- Leave-one-out: Extreme k-fold (k=n)",
        "",
        "Statistical comparison:",
        "- Paired t-tests: Compare two models",
        "- ANOVA: Compare multiple models",
        "- Wilcoxon signed-rank: Non-parametric alternative",
        "- McNemar's test: For binary classifiers",
        "",
        "Advanced methods:",
        "- Bayesian correlated t-test",
        "- Nemenyi test for multiple comparisons",
        "- Critical difference diagrams",
        "- Effect size measures (Cohen's d)"
      ],
      code: {
        python: `# Model Comparison in Python
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
from scipy import stats
import numpy as np

# Sample model scores (accuracy)
model_a_scores = [0.85, 0.82, 0.83, 0.86, 0.84]  # 5-fold CV results
model_b_scores = [0.87, 0.89, 0.88, 0.86, 0.87]

# Cross-validation
model = RandomForestClassifier()
cv = StratifiedKFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"CV Accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

# Time-series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
scores_ts = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')

# Statistical tests
# Paired t-test
t_stat, p_value = stats.ttest_rel(model_a_scores, model_b_scores)
print(f"Paired t-test p-value: {p_value:.4f}")

# Wilcoxon signed-rank test
wilcoxon_stat, wilcoxon_p = stats.wilcoxon(model_a_scores, model_b_scores)
print(f"Wilcoxon p-value: {wilcoxon_p:.4f}")

# Bayesian correlated t-test (requires external library)
# from baycomp import correlated_ttest
# prob = correlated_ttest(model_a_scores, model_b_scores, rope=0.01)
# print(f"Probability model B better: {prob:.2f}")

# Multiple model comparison
from scipy.stats import friedmanchisquare
model_c_scores = [0.88, 0.87, 0.89, 0.90, 0.88]
friedman_stat, friedman_p = friedmanchisquare(model_a_scores, model_b_scores, model_c_scores)
print(f"Friedman test p-value: {friedman_p:.4f}")`,
        complexity: "CV: O(k*n), Statistical tests: O(n)"
      }
    },
    {
      title: "📉 Error Analysis & Interpretation",
      id: "error-analysis",
      description: "Techniques for understanding model errors and improving performance.",
      keyPoints: [
        "Confusion matrix analysis",
        "Error distribution visualization",
        "Residual analysis for regression",
        "Learning curves"
      ],
      detailedExplanation: [
        "Classification error analysis:",
        "- False positive/negative patterns",
        "- Class-specific performance",
        "- Decision boundary visualization",
        "- Confidence calibration analysis",
        "",
        "Regression error analysis:",
        "- Residual plots (heteroscedasticity)",
        "- Error distribution (normality)",
        "- Partial dependence plots",
        "- Influence plots",
        "",
        "Diagnostic tools:",
        "- Learning curves (bias-variance)",
        "- Validation curves (hyperparameter effects)",
        "- Permutation importance",
        "- SHAP values for feature importance",
        "",
        "Actionable insights:",
        "- Identify systematic errors",
        "- Detect data quality issues",
        "- Guide feature engineering",
        "- Inform model selection"
      ],
      code: {
        python: `# Error Analysis in Python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.inspection import (
    permutation_importance,
    partial_dependence,
    PartialDependenceDisplay
)

# Classification error analysis
ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
plt.title("Confusion Matrix")
plt.show()

# Error cases
errors = np.where(y_true != y_pred)[0]
print(f"Error indices: {errors}")

# Regression error analysis
residuals = y_true - y_pred
plt.figure(figsize=(10,6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Learning curves
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(
    estimator=model, X=X, y=y, cv=5,
    scoring='neg_mean_squared_error'
)

plt.figure(figsize=(10,6))
plt.plot(train_sizes, -train_scores.mean(axis=1), label='Training error')
plt.plot(train_sizes, -test_scores.mean(axis=1), label='Validation error')
plt.xlabel("Training examples")
plt.ylabel("MSE")
plt.legend()
plt.title("Learning Curve")
plt.show()

# Feature importance
result = permutation_importance(model, X_test, y_test, n_repeats=10)
sorted_idx = result.importances_mean.argsort()

plt.figure(figsize=(10,6))
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=X.columns[sorted_idx])
plt.title("Permutation Importance")
plt.show()`,
        complexity: "Visualizations: O(n), Permutation importance: O(n*m)"
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
        Model Evaluation Metrics
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
        }}>Model Evaluation and Optimization → Evaluation Metrics</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Proper evaluation metrics are crucial for assessing model performance, comparing algorithms,
          and making data-driven decisions in machine learning. This section covers both standard and
          advanced metrics for classification, regression, and model comparison.
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
                  color: '#0369a1',
                  marginBottom: '1rem'
                }}>Core Metrics</h3>
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
        }}>Metric Selection Guide</h2>
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
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Problem Type</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Primary Metrics</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Secondary Metrics</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>When to Use</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Binary Classification", "Accuracy, ROC-AUC", "Precision, Recall, F1", "Imbalanced data needs class-specific metrics"],
                ["Multiclass Classification", "Accuracy, Log Loss", "Per-class F1, MCC", "When class distribution matters"],
                ["Regression", "RMSE, R²", "MAE, MAPE", "Interpretability vs outlier sensitivity"],
                ["Multi-label Classification", "Hamming Loss", "Jaccard Similarity", "When labels aren't mutually exclusive"],
                ["Probabilistic Forecasting", "CRPS", "Log Score", "Evaluating full distribution"]
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
        }}>Evaluation Best Practices</h3>
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
            }}>Metric Selection Principles</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Align metrics with business objectives and costs
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Consider class imbalance and error costs
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Use multiple complementary metrics for robust evaluation
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Include probabilistic metrics when uncertainty matters
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
            }}>Common Pitfalls</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Accuracy Paradox:</strong> High accuracy on imbalanced data<br/>
              <strong>Overfitting Metrics:</strong> Optimizing single metric too much<br/>
              <strong>Ignoring Variance:</strong> Not reporting confidence intervals<br/>
              <strong>Data Leakage:</strong> Improper cross-validation setup
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
              <strong>Cost-Sensitive Metrics:</strong> Incorporate business costs<br/>
              <strong>Fairness Metrics:</strong> Demographic parity, equal opportunity<br/>
              <strong>Model Calibration:</strong> Reliability diagrams<br/>
              <strong>Uncertainty Quantification:</strong> Prediction intervals
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default EvaluationMetrics;