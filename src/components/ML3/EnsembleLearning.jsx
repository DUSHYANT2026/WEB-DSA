import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function EnsembleLearning() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "👜 Bagging (Bootstrap Aggregating)",
      id: "bagging",
      description: "Ensemble method that reduces variance by averaging multiple models trained on different data subsets.",
      keyPoints: [
        "Creates multiple datasets via bootstrap sampling",
        "Trains independent models in parallel",
        "Averages predictions (regression) or votes (classification)",
        "Most effective with high-variance models (e.g., decision trees)"
      ],
      detailedExplanation: [
        "How Bagging Works:",
        "1. Generate multiple bootstrap samples from training data",
        "2. Train a base model on each sample",
        "3. Combine predictions through averaging or majority voting",
        "",
        "Key Characteristics:",
        "- Reduces variance without increasing bias",
        "- Models can be trained in parallel",
        "- Works well with unstable learners (models that change significantly with small data changes)",
        "- Less prone to overfitting than single models",
        "",
        "Mathematical Foundation:",
        "- Bootstrap sampling approximates multiple training sets",
        "- Variance reduction proportional to 1/n (n = number of models)",
        "- Out-of-bag (OOB) samples can be used for validation",
        "",
        "Common Implementations:",
        "- Random Forest (bagged decision trees)",
        "- Bagged neural networks",
        "- Bagged SVMs for unstable configurations"
      ],
      code: {
        python: `# Bagging Implementation Example
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create base estimator
base_estimator = DecisionTreeClassifier(max_depth=4)

# Create bagging ensemble
bagging = BaggingClassifier(
    estimator=base_estimator,
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    bootstrap_features=False,
    oob_score=True,
    random_state=42
)

# Train and evaluate
bagging.fit(X_train, y_train)
print(f"Training Accuracy: {bagging.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {bagging.score(X_test, y_test):.4f}")
print(f"OOB Score: {bagging.oob_score_:.4f}")

# Compare to single tree
single_tree = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
print(f"Single Tree Test Accuracy: {single_tree.score(X_test, y_test):.4f}")`,
        complexity: "Training: O(t*(n log n + p)) where t=number of trees, n=samples, p=features"
      }
    },
    {
      title: "🚀 Boosting",
      id: "boosting",
      description: "Sequential ensemble method that converts weak learners into strong learners by focusing on errors.",
      keyPoints: [
        "Models trained sequentially, each correcting previous errors",
        "Weights misclassified instances higher in next iteration",
        "Includes AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost",
        "Generally more accurate than bagging but can overfit"
      ],
      detailedExplanation: [
        "Boosting Mechanics:",
        "1. Train initial model on original data",
        "2. Calculate errors/residuals",
        "3. Train next model to predict errors",
        "4. Combine models with appropriate weights",
        "5. Repeat until stopping criteria met",
        "",
        "Key Algorithms:",
        "- AdaBoost (Adaptive Boosting): Reweights misclassified samples",
        "- Gradient Boosting: Fits new models to residual errors",
        "- XGBoost: Optimized gradient boosting with regularization",
        "- LightGBM: Histogram-based gradient boosting",
        "- CatBoost: Handles categorical features natively",
        "",
        "Mathematical Insights:",
        "- Minimizes loss function via gradient descent in function space",
        "- Learning rate controls contribution of each model",
        "- Early stopping prevents overfitting",
        "- Regularization terms in modern implementations",
        "",
        "Practical Considerations:",
        "- More sensitive to noisy data than bagging",
        "- Requires careful tuning of learning rate and tree depth",
        "- Generally achieves higher accuracy than bagging",
        "- Sequential nature limits parallelization"
      ],
      code: {
        python: `# Boosting Implementation Examples
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# AdaBoost
adaboost = AdaBoostClassifier(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)
adaboost.fit(X_train, y_train)
print(f"AdaBoost Accuracy: {accuracy_score(y_test, adaboost.predict(X_test)):.4f}")

# Gradient Boosting
gbm = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gbm.fit(X_train, y_train)
print(f"GBM Accuracy: {accuracy_score(y_test, gbm.predict(X_test)):.4f}")

# XGBoost
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    reg_lambda=1,  # L2 regularization
    reg_alpha=0,   # L1 regularization
    random_state=42
)
xgb.fit(X_train, y_train)
print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb.predict(X_test)):.4f}")

# LightGBM
lgbm = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    num_leaves=15,
    random_state=42
)
lgbm.fit(X_train, y_train)
print(f"LightGBM Accuracy: {accuracy_score(y_test, lgbm.predict(X_test)):.4f}")

# CatBoost
catboost = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    depth=3,
    cat_features=[],  # Specify categorical feature indices
    verbose=0,
    random_state=42
)
catboost.fit(X_train, y_train)
print(f"CatBoost Accuracy: {accuracy_score(y_test, catboost.predict(X_test)):.4f}")`,
        complexity: "Training: O(t*n*p) where t=iterations, n=samples, p=features"
      }
    },
    {
      title: "🧱 Stacking",
      id: "stacking",
      description: "Advanced ensemble method that combines multiple models via a meta-learner.",
      keyPoints: [
        "Trains diverse base models (level-0)",
        "Uses base model predictions as features for meta-model (level-1)",
        "Can combine different types of models (e.g., SVM + NN + RF)",
        "Requires careful validation to avoid data leakage"
      ],
      detailedExplanation: [
        "Stacking Architecture:",
        "1. Train diverse base models on training data",
        "2. Generate cross-validated predictions (meta-features)",
        "3. Train meta-model on these predictions",
        "4. Final prediction combines base models through meta-model",
        "",
        "Implementation Variants:",
        "- Single-level stacking: One meta-model",
        "- Multi-level stacking: Multiple stacking layers",
        "- Blending: Similar but uses holdout set instead of CV",
        "",
        "Key Considerations:",
        "- Base models should be diverse (different algorithms)",
        "- Meta-model is typically simple (linear model)",
        "- Cross-validation essential to prevent overfitting",
        "- Computational cost higher than bagging/boosting",
        "",
        "Practical Applications:",
        "- Winning solution in many Kaggle competitions",
        "- Combining strengths of different model types",
        "- When no single model clearly outperforms others",
        "- For extracting maximum performance from available data"
      ],
      code: {
        python: `# Stacking Implementation Example
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Define base models
estimators = [
    ('svm', SVC(probability=True, kernel='rbf', random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('dt', DecisionTreeClassifier(max_depth=3, random_state=42))
]

# Define meta-model
meta_model = LogisticRegression()

# Create stacking classifier
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    stack_method='auto',
    cv=5,
    passthrough=False,
    verbose=0
)

# Train and evaluate
stacking.fit(X_train, y_train)
print(f"Stacking Accuracy: {stacking.score(X_test, y_test):.4f}")

# Compare to individual models
for name, model in estimators:
    model.fit(X_train, y_train)
    print(f"{name} Accuracy: {model.score(X_test, y_test):.4f}")

# Advanced stacking with feature passthrough
stacking_advanced = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    passthrough=True,  # Include original features
    cv=5
)
stacking_advanced.fit(X_train, y_train)
print(f"Advanced Stacking Accuracy: {stacking_advanced.score(X_test, y_test):.4f}")`,
        complexity: "Training: O(k*(m*n + p)) where k=CV folds, m=base models, n=samples, p=features"
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
        Ensemble Learning Methods
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
        }}>Advanced Machine Learning Algorithms → Ensemble Learning</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Ensemble methods combine multiple machine learning models to achieve better performance
          than any individual model. These techniques are among the most powerful approaches
          in modern machine learning, often winning competitions and being deployed in production systems.
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
                backgroundColor: '#bae6fd',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#0369a1',
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
        }}>Ensemble Method Comparison</h2>
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
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Parallelizable</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Reduces</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Best For</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Key Libraries</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Bagging", "Yes", "Variance", "High-variance models (deep trees)", "scikit-learn"],
                ["Boosting", "No", "Bias", "Improving weak learners", "XGBoost, LightGBM, CatBoost"],
                ["Stacking", "Partially", "Both", "Combining diverse models", "scikit-learn, mlxtend"]
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
        }}>Ensemble Learning Best Practices</h3>
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
            }}>When to Use Each Method</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                <strong>Bagging:</strong> When your base model overfits (high variance)
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                <strong>Boosting:</strong> When you need to improve model accuracy (reduce bias)
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                <strong>Stacking:</strong> When you have several good but different models
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                <strong>Voting:</strong> For quick improvements with diverse models
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
            }}>Practical Implementation Tips</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>For Bagging:</strong> Use deep trees as base learners<br/>
              <strong>For Boosting:</strong> Start with shallow trees and tune learning rate<br/>
              <strong>For Stacking:</strong> Ensure base model diversity<br/>
              <strong>For All:</strong> Use early stopping and cross-validation
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
              <strong>Time Series:</strong> Sequential ensemble methods<br/>
              <strong>Anomaly Detection:</strong> Isolation Forest ensembles<br/>
              <strong>Feature Selection:</strong> Using feature importance across ensembles<br/>
              <strong>Model Interpretation:</strong> SHAP values for ensemble models
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default EnsembleLearning;