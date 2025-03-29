import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function SplittingData() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "✂️ Train-Test Split",
      id: "train-test",
      description: "The fundamental technique for evaluating model performance by separating data into training and testing sets.",
      keyPoints: [
        "Random splitting of dataset into two subsets",
        "Typical splits: 70-30, 80-20, or similar ratios",
        "Preserves distribution of important features",
        "Prevents data leakage between sets"
      ],
      detailedExplanation: [
        "Key considerations:",
        "- Size of test set depends on dataset size and variability",
        "- Stratified splitting for imbalanced datasets",
        "- Time-based splitting for temporal data",
        "- Group-based splitting for dependent samples",
        "",
        "Implementation details:",
        "- Random state for reproducibility",
        "- Shuffling before splitting (except time series)",
        "- Feature scaling after splitting to prevent leakage",
        "- Multiple splits for more reliable evaluation",
        "",
        "Common pitfalls:",
        "- Test set too small for reliable evaluation",
        "- Data leakage through improper preprocessing",
        "- Non-representative splits (e.g., sorted data)",
        "- Ignoring temporal or group dependencies"
      ],
      code: {
        python: `# Train-Test Split Examples
from sklearn.model_selection import train_test_split
import numpy as np

# Basic random split
X = np.random.rand(1000, 10)  # 1000 samples, 10 features
y = np.random.randint(0, 2, 1000)  # Binary target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # 20% test
    random_state=42,  # For reproducibility
    shuffle=True,
    stratify=y  # Preserve class distribution
)

# Time-based splitting
time_series_data = np.random.randn(365, 5)  # 1 year daily data
split_point = int(0.8 * len(time_series_data))  # 80% train
X_train_time = time_series_data[:split_point]
X_test_time = time_series_data[split_point:]

# Group-based splitting
from sklearn.model_selection import GroupShuffleSplit

groups = np.random.randint(0, 10, 1000)  # 10 groups
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train_group, X_test_group = X[train_idx], X[test_idx]
y_train_group, y_test_group = y[train_idx], y[test_idx]`,
        complexity: "O(n) time complexity, where n is number of samples"
      }
    },
    {
      title: "🔄 Cross-Validation",
      id: "cross-validation",
      description: "Robust technique for model evaluation by systematically creating multiple train-test splits.",
      keyPoints: [
        "K-Fold: Dividing data into K equal parts",
        "Stratified K-Fold: Preserving class distribution",
        "Leave-One-Out: Extreme case where K = n",
        "Time Series CV: Specialized for temporal data"
      ],
      detailedExplanation: [
        "Why use cross-validation:",
        "- More reliable estimate of model performance",
        "- Better utilization of limited data",
        "- Reduces variance in performance estimates",
        "- Helps detect overfitting",
        "",
        "Common variants:",
        "1. K-Fold: Standard approach for most problems",
        "2. Repeated K-Fold: Multiple runs with different splits",
        "3. Stratified K-Fold: For imbalanced datasets",
        "4. Leave-One-Out: For very small datasets",
        "5. Time Series Split: Ordered splits for temporal data",
        "",
        "Implementation best practices:",
        "- Choose K based on dataset size (typically 5 or 10)",
        "- Ensure proper shuffling (except time series)",
        "- Maintain same preprocessing within each fold",
        "- Aggregate results across folds properly"
      ],
      code: {
        python: `# Cross-Validation Examples
from sklearn.model_selection import (
    KFold, 
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score
)
from sklearn.ensemble import RandomForestClassifier

# Standard K-Fold (k=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    RandomForestClassifier(),
    X, y,
    cv=kf,
    scoring='accuracy'
)
print(f"KFold scores: {cv_scores}")
print(f"Mean accuracy: {np.mean(cv_scores):.2f}")

# Stratified K-Fold for imbalanced data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
strat_scores = cross_val_score(
    RandomForestClassifier(),
    X, y,
    cv=skf,
    scoring='f1'
)

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
time_scores = []
for train_idx, test_idx in tscv.split(time_series_data):
    model.fit(time_series_data[train_idx], y[train_idx])
    score = model.score(time_series_data[test_idx], y[test_idx])
    time_scores.append(score)

# Nested CV for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {'max_depth': [3, 5, 7]}
grid = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=inner_cv
)
nested_score = cross_val_score(grid, X, y, cv=outer_cv)`,
        complexity: "O(k*n) where k is number of folds, n is dataset size"
      }
    },
    {
      title: "⚖️ Validation Sets",
      id: "validation",
      description: "Intermediate dataset used for tuning hyperparameters and model selection.",
      keyPoints: [
        "Separate from both training and test sets",
        "Used for model selection and hyperparameter tuning",
        "Typical splits: 60-20-20 or similar ratios",
        "Prevents overfitting to test set metrics"
      ],
      detailedExplanation: [
        "Validation set purposes:",
        "- Hyperparameter tuning",
        "- Early stopping in neural networks",
        "- Model architecture selection",
        "- Feature selection decisions",
        "",
        "Implementation approaches:",
        "1. Fixed validation set: Simple but reduces training data",
        "2. Cross-validation with validation: More data-efficient",
        "3. Nested cross-validation: Most rigorous but computationally expensive",
        "",
        "Best practices:",
        "- Never use test set for any decision making",
        "- Match validation distribution to expected test conditions",
        "- Consider multiple validation sets for robustness",
        "- Document all decisions made based on validation",
        "",
        "Special cases:",
        "- Time series: Forward validation (train on past, validate on future)",
        "- Small datasets: Cross-validation instead of fixed split",
        "- Grouped data: Keep groups together in splits"
      ],
      code: {
        python: `# Validation Set Strategies
# Simple train-val-test split
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
)

# Using cross-validation for validation
from sklearn.model_selection import validation_curve

param_range = np.logspace(-6, -1, 5)
train_scores, val_scores = validation_curve(
    RandomForestClassifier(),
    X_train_val, y_train_val,
    param_name="min_samples_split",
    param_range=param_range,
    cv=5,
    scoring="accuracy"
)

# Early stopping with validation set
from tensorflow.keras.callbacks import EarlyStopping

model = create_neural_network()  # Assume defined elsewhere
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

# Nested validation with GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01]}
grid = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid.fit(X_train_val, y_train_val)
best_model = grid.best_estimator_
test_score = best_model.score(X_test, y_test)`,
        complexity: "Similar to train-test split, plus additional model training"
      }
    },
    {
      title: "📊 Data Splitting Strategies",
      id: "strategies",
      description: "Specialized approaches for different data types and problem scenarios.",
      keyPoints: [
        "Stratified sampling for imbalanced classes",
        "Group-based splitting for dependent samples",
        "Time-based splitting for temporal data",
        "Cluster-based splitting for complex distributions"
      ],
      detailedExplanation: [
        "Advanced splitting techniques:",
        "- Stratified sampling: Preserves class ratios in splits",
        "- Group splitting: Keeps related samples together",
        "- Time series splitting: Maintains temporal order",
        "- Cluster splitting: Ensures diversity in splits",
        "",
        "Domain-specific considerations:",
        "- Medical data: Patient-wise splitting",
        "- NLP: Document or author-wise splitting",
        "- Recommender systems: User-wise splitting",
        "- Geospatial data: Location-based splitting",
        "",
        "Implementation tools:",
        "- Scikit-learn's GroupShuffleSplit, TimeSeriesSplit",
        "- Custom splitting functions for special cases",
        "- Synthetic data augmentation for small datasets",
        "- Active learning approaches for iterative splitting",
        "",
        "Evaluation metrics:",
        "- Check distribution similarity between splits",
        "- Verify no leakage between sets",
        "- Assess whether splits reflect real-world conditions",
        "- Measure stability of results across different splits"
      ],
      code: {
        python: `# Advanced Splitting Strategies
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    GroupShuffleSplit,
    TimeSeriesSplit
)

# Stratified splitting for imbalanced data
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

# Group splitting (e.g., patients in medical data)
groups = np.random.randint(0, 100, 1000)  # 100 groups
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in gss.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

# Time series splitting
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(time_series_data):
    X_train, X_test = time_series_data[train_idx], time_series_data[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

# Cluster-based splitting
from sklearn.cluster import KMeans

# Cluster data into 5 groups
clusters = KMeans(n_clusters=5).fit_predict(X)
unique_clusters = np.unique(clusters)
cluster_splits = np.array_split(np.random.permutation(unique_clusters), 2)

train_idx = np.where(np.isin(clusters, cluster_splits[0]))[0]
test_idx = np.where(np.isin(clusters, cluster_splits[1]))[0]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]`,
        complexity: "Varies by method, typically O(n) to O(n²)"
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
        Data Splitting Strategies
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
        }}>Data Preprocessing → Splitting Data</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Proper data splitting is crucial for developing robust machine learning models.
          This section covers techniques to partition datasets for training, validation,
          and testing while avoiding common pitfalls like data leakage.
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
              color: '#0ea5e9'
            }}>{section.title}</h2>
            <button
              onClick={() => toggleSection(section.id)}
              style={{
                background: 'linear-gradient(to right, #0ea5e9, #38bdf8)',
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
                backgroundColor: '#e0f2fe',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#0ea5e9',
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
                  color: '#0ea5e9',
                  marginBottom: '1rem'
                }}>Implementation</h3>
                <p style={{
                  color: '#374151',
                  fontWeight: '600',
                  marginBottom: '1rem',
                  fontSize: '1.1rem'
                }}>Computational Complexity: {section.code.complexity}</p>
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
          color: '#0ea5e9',
          marginBottom: '2rem'
        }}>Data Splitting Method Comparison</h2>
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
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Method</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Best For</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Pros</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Cons</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Train-Test Split", "Large datasets, quick evaluation", "Simple, fast", "Higher variance in estimates"],
                ["K-Fold CV", "Small to medium datasets", "Reliable estimates, uses all data", "Computationally expensive"],
                ["Stratified Split", "Imbalanced datasets", "Preserves class distribution", "More complex implementation"],
                ["Time Series Split", "Temporal data", "Maintains time ordering", "Can't shuffle, less flexible"],
                ["Group Split", "Dependent samples (e.g., patients)", "Prevents leakage", "Requires group labels"]
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
          color: '#0ea5e9',
          marginBottom: '1.5rem'
        }}>Best Practices</h3>
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
            }}>General Guidelines</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Always split data before any preprocessing or feature engineering
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                For small datasets, prefer cross-validation over simple splits
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Match your splitting strategy to your problem's real-world conditions
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Document your splitting methodology for reproducibility
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
            }}>Common Pitfalls to Avoid</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Data Leakage:</strong> When information from test set influences training<br/>
              <strong>Overfitting to Test Set:</strong> Repeated evaluation on same test data<br/>
              <strong>Improper Shuffling:</strong> Ordered data creates biased splits<br/>
              <strong>Ignoring Dependencies:</strong> Splitting correlated samples independently
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
            }}>Special Cases</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Time Series:</strong> Use forward-chaining validation approaches<br/>
              <strong>Imbalanced Data:</strong> Stratified sampling preserves class ratios<br/>
              <strong>Grouped Data:</strong> Keep all samples from same group together<br/>
              <strong>Active Learning:</strong> Iterative splitting based on model uncertainty
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SplittingData;