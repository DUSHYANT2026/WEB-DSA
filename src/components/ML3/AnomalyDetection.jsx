import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";
import { useTheme } from "../../ThemeContext.jsx";

function AnomalyDetection() {
  const { darkMode } = useTheme();
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "🕵️ Isolation Forest",
      id: "isolation",
      description: "An efficient algorithm for anomaly detection based on isolating outliers in high-dimensional data.",
      keyPoints: [
        "Works by randomly partitioning data",
        "Anomalies require fewer partitions to isolate",
        "No distance or density calculations needed",
        "Effective for high-dimensional data"
      ],
      detailedExplanation: [
        "How it works:",
        "- Builds an ensemble of isolation trees",
        "- Randomly selects features and split values",
        "- Anomalies have shorter path lengths in trees",
        "- Combines results from multiple trees",
        "",
        "Key advantages:",
        "- Low linear time complexity",
        "- Handles irrelevant features well",
        "- Works without feature scaling",
        "- Effective for multi-modal data",
        "",
        "Parameters to tune:",
        "- Number of estimators (trees)",
        "- Contamination (expected outlier fraction)",
        "- Maximum tree depth",
        "- Bootstrap sampling"
      ],
      code: {
        python: `# Isolation Forest Example
from sklearn.ensemble import IsolationForest
import numpy as np

# Generate sample data (95% normal, 5% anomalies)
X = 0.3 * np.random.randn(100, 2)
X = np.r_[X + 2, X - 2, X + [5, -3]]  # Add anomalies

# Train model
clf = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)
clf.fit(X)

# Predict anomalies (1=normal, -1=anomaly)
y_pred = clf.predict(X)

# Get anomaly scores (the lower, the more abnormal)
scores = clf.decision_function(X)

# Plot results
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm')
plt.title("Isolation Forest Anomaly Detection")
plt.show()`,
        complexity: "Training: O(n log n), Prediction: O(n)"
      }
    },
    {
      title: "🛡️ One-Class SVM",
      id: "svm",
      description: "A support vector machine approach that learns a decision boundary around normal data points.",
      keyPoints: [
        "Learns a tight boundary around normal data",
        "Uses kernel trick for non-linear boundaries",
        "Good for high-dimensional data",
        "Sensitive to kernel choice and parameters"
      ],
      detailedExplanation: [
        "Key concepts:",
        "- Maps data to high-dimensional feature space",
        "- Finds maximum margin hyperplane",
        "- Only uses normal data for training",
        "- Treats origin as the outlier class",
        "",
        "Implementation details:",
        "- Uses ν parameter to control outlier fraction",
        "- Supports RBF, polynomial, and sigmoid kernels",
        "- Requires careful parameter tuning",
        "- Needs feature scaling for best performance",
        "",
        "When to use:",
        "- When you only have normal class data",
        "- For high-dimensional feature spaces",
        "- When you need probabilistic outputs",
        "- For non-linear decision boundaries",
        "",
        "Limitations:",
        "- Computationally intensive for large datasets",
        "- Hard to interpret results",
        "- Sensitive to kernel parameters",
        "- Doesn't scale well to very high dimensions"
      ],
      code: {
        python: `# One-Class SVM Example
from sklearn.svm import OneClassSVM
import numpy as np

# Generate normal data (no anomalies in training)
X_train = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X_train + 2, X_train - 2]

# Add anomalies to test set
X_test = np.r_[X_train, [[5, -3], [6, 4]]]

# Train model
clf = OneClassSVM(
    kernel='rbf',
    gamma=0.1,
    nu=0.05  # expected outlier fraction
)
clf.fit(X_train)

# Predict anomalies (1=normal, -1=anomaly)
y_pred = clf.predict(X_test)

# Get decision scores (distance to boundary)
scores = clf.decision_function(X_test)

# Plot results
import matplotlib.pyplot as plt
xx, yy = np.meshgrid(np.linspace(-5, 10, 500), np.linspace(-5, 10, 500))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm')
plt.title("One-Class SVM Anomaly Detection")
plt.show()`,
        complexity: "Training: O(n²) to O(n³), Prediction: O(n)"
      }
    },
    {
      title: "📊 Local Outlier Factor (LOF)",
      id: "lof",
      description: "A density-based algorithm that compares local density of points to their neighbors.",
      keyPoints: [
        "Measures local deviation in density",
        "Identifies local outliers",
        "Works well with clustered data",
        "Sensitive to neighborhood size"
      ],
      detailedExplanation: [
        "Core algorithm:",
        "- Computes k-distance (distance to kth neighbor)",
        "- Calculates reachability distance",
        "- Determines local reachability density (LRD)",
        "- Compares LRD to neighbors' LRD",
        "",
        "Key parameters:",
        "- n_neighbors: Number of neighbors to consider",
        "- contamination: Expected outlier fraction",
        "- metric: Distance metric to use",
        "- algorithm: Nearest neighbors algorithm",
        "",
        "Advantages:",
        "- Detects local anomalies in clustered data",
        "- Provides outlier scores (not just binary)",
        "- Works with arbitrary distance metrics",
        "- Handles non-uniform density distributions",
        "",
        "Limitations:",
        "- Computationally expensive for large datasets",
        "- Sensitive to neighborhood size parameter",
        "- Struggles with high-dimensional data",
        "- Requires meaningful distance metric"
      ],
      code: {
        python: `# Local Outlier Factor Example
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

# Generate data with two clusters and some anomalies
X = 0.3 * np.random.randn(50, 2)
X = np.r_[X + 2, X - 2, [[5, -3], [6, 4], [0, 0]]]

# Fit model
clf = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.1,
    novelty=True  # predict on new data
)
y_pred = clf.fit_predict(X)

# Negative scores are outliers, higher = more normal
scores = clf.negative_outlier_factor_

# Plot results
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm')
plt.title("Local Outlier Factor Anomaly Detection")
plt.show()

# Decision function visualization
xx, yy = np.meshgrid(np.linspace(-5, 10, 500), np.linspace(-5, 10, 500))
Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm')
plt.title("LOF Decision Boundaries")
plt.show()`,
        complexity: "Training: O(n²), Prediction: O(n)"
      }
    },
    {
      title: "🔢 Statistical Methods",
      id: "statistical",
      description: "Classical statistical approaches for identifying anomalies based on distribution assumptions.",
      keyPoints: [
        "Z-score: Standard deviations from mean",
        "Modified Z-score: Robust to outliers",
        "IQR method: Uses quartile ranges",
        "Mahalanobis distance: Multivariate"
      ],
      detailedExplanation: [
        "Z-score method:",
        "- Assumes normal distribution",
        "- Threshold typically ±3 standard deviations",
        "- Simple but sensitive to extreme values",
        "",
        "Modified Z-score:",
        "- Uses median and MAD (median absolute deviation)",
        "- More robust to existing outliers",
        "- Recommended for real-world data",
        "",
        "IQR method:",
        "- Uses 25th and 75th percentiles",
        "- Outliers outside Q1 - 1.5*IQR or Q3 + 1.5*IQR",
        "- Non-parametric (no distribution assumptions)",
        "",
        "Mahalanobis distance:",
        "- Accounts for covariance between features",
        "- Measures distance from distribution center",
        "- Requires inverse covariance matrix",
        "- Sensitive to sample size and dimensionality",
        "",
        "When to use:",
        "- When you know the data distribution",
        "- For quick baseline implementations",
        "- When interpretability is important",
        "- For low-dimensional data"
      ],
      code: {
        python: `# Statistical Anomaly Detection
import numpy as np
from scipy import stats

# Generate data with outliers
data = np.concatenate([np.random.normal(0, 1, 100), 
                      [10, -8, 5.5, -4.2]])

# Z-score method
z_scores = np.abs(stats.zscore(data))
z_threshold = 3
z_outliers = np.where(z_scores > z_threshold)

# Modified Z-score (more robust)
median = np.median(data)
mad = np.median(np.abs(data - median))
modified_z = 0.6745 * (data - median) / mad  # 0.6745 = 0.75th percentile of N(0,1)
modz_outliers = np.where(np.abs(modified_z) > z_threshold)

# IQR method
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
iqr_outliers = np.where((data < lower_bound) | (data > upper_bound))

# Mahalanobis distance (multivariate)
from sklearn.covariance import EmpiricalCovariance
X = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
X = np.r_[X, [[5, 5], [-4, -4]]]  # Add outliers

cov = EmpiricalCovariance().fit(X)
mahalanobis_dist = cov.mahalanobis(X)
maha_threshold = np.percentile(mahalanobis_dist, 95)  # 95th percentile
maha_outliers = np.where(mahalanobis_dist > maha_threshold)`,
        complexity: "Z-score/IQR: O(n), Mahalanobis: O(n²) to O(n³)"
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
        : 'linear-gradient(to bottom right, #f0f9ff, #e0f2fe)',
      borderRadius: '20px',
      boxShadow: darkMode ? '0 10px 30px rgba(0,0,0,0.3)' : '0 10px 30px rgba(0,0,0,0.1)',
      color: darkMode ? '#e2e8f0' : '#1e293b'
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
        Anomaly Detection Techniques
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
          color: '#0369a1',
          marginBottom: '1rem'
        }}>Unsupervised Learning → Anomaly Detection</h2>
        <p style={{
          color: darkMode ? '#e2e8f0' : '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Anomaly detection identifies rare items, events or observations which raise suspicions 
          by differing significantly from the majority of the data. These techniques are widely 
          used in fraud detection, system health monitoring, and data cleaning.
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
            border: darkMode ? '1px solid #334155' : '1px solid #bae6fd',
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
                backgroundColor: darkMode ? '#1e3a8a' : '#e0f2fe',
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
                backgroundColor: darkMode ? '#064e3b' : '#ecfdf5',
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
                backgroundColor: darkMode ? '#164e63' : '#f0f9ff',
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
                  color: darkMode ? '#e2e8f0' : '#374151',
                  fontWeight: '600',
                  marginBottom: '1rem',
                  fontSize: '1.1rem'
                }}>{section.code.complexity}</p>
                <div style={{
                  borderRadius: '8px',
                  overflow: 'hidden',
                  border: darkMode ? '2px solid #0c4a6e' : '2px solid #7dd3fc'
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

      {/* Comparison Table */}
      <div style={{
        marginTop: '3rem',
        padding: '2rem',
        backgroundColor: darkMode ? '#1e293b' : 'white',
        borderRadius: '16px',
        boxShadow: darkMode ? '0 5px 15px rgba(0,0,0,0.3)' : '0 5px 15px rgba(0,0,0,0.05)',
        border: darkMode ? '1px solid #334155' : '1px solid #bae6fd'
      }}>
        <h2 style={{
          fontSize: '2rem',
          fontWeight: '700',
          color: '#0369a1',
          marginBottom: '2rem'
        }}>Anomaly Detection Algorithm Comparison</h2>
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
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Algorithm</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Strengths</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Weaknesses</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Best Use Cases</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Isolation Forest", "Fast, handles high dimensions", "Less precise on local anomalies", "High-dimensional data, large datasets"],
                ["One-Class SVM", "Flexible boundaries, kernel trick", "Slow, sensitive to parameters", "Non-linear boundaries, small datasets"],
                ["Local Outlier Factor", "Detects local anomalies", "Computationally expensive", "Clustered data, local anomalies"],
                ["Statistical Methods", "Simple, interpretable", "Strong distribution assumptions", "Low-dimensional data, quick baselines"]
              ].map((row, index) => (
                <tr key={index} style={{
                  backgroundColor: index % 2 === 0 
                    ? (darkMode ? '#334155' : '#f0f9ff') 
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
        border: darkMode ? '1px solid #1e40af' : '1px solid #a7f3d0'
      }}>
        <h3 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#0369a1',
          marginBottom: '1.5rem'
        }}>Practical Guidance</h3>
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
              color: '#0369a1',
              marginBottom: '0.75rem'
            }}>Choosing an Algorithm</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: darkMode ? '#e2e8f0' : '#374151', fontSize: '1.1rem' }}>
                For high-dimensional data: Isolation Forest or One-Class SVM
              </li>
              <li style={{ color: darkMode ? '#e2e8f0' : '#374151', fontSize: '1.1rem' }}>
                For clustered data: Local Outlier Factor
              </li>
              <li style={{ color: darkMode ? '#e2e8f0' : '#374151', fontSize: '1.1rem' }}>
                For interpretability: Statistical methods
              </li>
              <li style={{ color: darkMode ? '#e2e8f0' : '#374151', fontSize: '1.1rem' }}>
                For large datasets: Isolation Forest
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
              color: '#0369a1',
              marginBottom: '0.75rem'
            }}>Implementation Tips</h4>
            <p style={{
              color: darkMode ? '#e2e8f0' : '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Feature Scaling:</strong> Normalize before using distance-based methods<br/>
              <strong>Parameter Tuning:</strong> Adjust contamination rate based on domain knowledge<br/>
              <strong>Evaluation:</strong> Use precision@k when labeled anomalies are available<br/>
              <strong>Visualization:</strong> Plot anomaly scores to understand model behavior
            </p>
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
              color: '#0369a1',
              marginBottom: '0.75rem'
            }}>Advanced Applications</h4>
            <p style={{
              color: darkMode ? '#e2e8f0' : '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Time Series:</strong> Specialized models like STL decomposition<br/>
              <strong>Graph Data:</strong> Community detection approaches<br/>
              <strong>Image Data:</strong> Autoencoder reconstruction error<br/>
              <strong>Text Data:</strong> Rare topic or word pattern detection
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AnomalyDetection;