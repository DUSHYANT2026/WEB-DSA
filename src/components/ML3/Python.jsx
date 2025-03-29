import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function Python() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "🐍 Python Basics",
      id: "basics",
      description: "Fundamental programming concepts essential for ML implementation.",
      keyPoints: [
        "Variables and data types (int, float, str, bool)",
        "Control flow (if-else, loops)",
        "Functions and lambda expressions",
        "List comprehensions and generators"
      ],
      detailedExplanation: [
        "Core concepts for ML programming:",
        "- Dynamic typing for flexible data handling",
        "- Iterators and generators for memory efficiency",
        "- Functional programming patterns (map, filter, reduce)",
        "- Exception handling for robust ML pipelines",
        "",
        "ML-specific patterns:",
        "- Vectorized operations for performance",
        "- Generator functions for streaming large datasets",
        "- Decorators for logging and timing model training",
        "- Context managers for resource handling",
        "",
        "Performance considerations:",
        "- Avoiding global variables in ML scripts",
        "- Proper variable scoping in notebooks",
        "- Memory management with large datasets",
        "- Profiling computational bottlenecks"
      ],
      code: {
        python: `# Python Basics for ML
# Variables and types
batch_size = 64  # int
learning_rate = 0.001  # float
model_name = "resnet50"  # str
is_training = True  # bool

# Control flow
if batch_size > 32 and is_training:
    print("Using large batch training")
elif not is_training:
    print("Evaluation mode")
else:
    print("Small batch training")

# Loops
for epoch in range(10):  # 10 epochs
    for batch in batches:
        train(batch)

# Functions
def calculate_accuracy(y_true, y_pred):
    """Compute classification accuracy"""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)

# Lambda for simple transforms
square = lambda x: x**2
squared_data = list(map(square, raw_data))

# List comprehension for data processing
cleaned_data = [preprocess(x) for x in raw_data if x is not None]

# Generator for memory efficiency
def data_stream(file_path, chunk_size=1024):
    """Yield data in chunks"""
    with open(file_path) as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            yield process(data)`,
        complexity: "Basic operations: O(1), Loops: O(n), Comprehensions: O(n)"
      }
    },
    {
      title: "📚 Python Libraries",
      id: "libraries",
      description: "Essential scientific computing libraries for machine learning workflows.",
      keyPoints: [
        "NumPy: Numerical computing with arrays",
        "Pandas: Data manipulation and analysis",
        "Matplotlib: Data visualization",
        "Seaborn: Statistical visualization"
      ],
      detailedExplanation: [
        "NumPy for ML:",
        "- Efficient n-dimensional arrays",
        "- Broadcasting rules for vectorized operations",
        "- Linear algebra operations (dot product, SVD)",
        "- Random sampling from distributions",
        "",
        "Pandas for data preparation:",
        "- DataFrames for structured data",
        "- Handling missing data (NA, NaN)",
        "- Time series functionality",
        "- Merging and joining datasets",
        "",
        "Visualization tools:",
        "- Matplotlib for custom plots",
        "- Seaborn for statistical visualizations",
        "- Interactive plotting with widgets",
        "- Saving publication-quality figures",
        "",
        "Integration with ML:",
        "- Converting between Pandas and NumPy",
        "- Data preprocessing pipelines",
        "- Feature visualization",
        "- Model evaluation plots"
      ],
      code: {
        python: `# Python Libraries for ML
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NumPy arrays
features = np.random.randn(1000, 10)  # 1000 samples, 10 features
weights = np.zeros(10)
predictions = np.dot(features, weights)

# Pandas DataFrame
data = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(10)])
data['target'] = np.random.randint(0, 2, 1000)  # Binary target

# Data exploration
print(data.describe())
print(data.isna().sum())

# Matplotlib visualization
plt.figure(figsize=(10,6))
plt.scatter(data['feature_0'], data['feature_1'], c=data['target'])
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title("Feature Space")
plt.colorbar(label="Target")
plt.show()

# Seaborn visualization
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# Advanced visualization
g = sns.PairGrid(data.sample(100), vars=['feature_0', 'feature_1', 'feature_2'], hue='target')
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()`,
        complexity: "NumPy ops: O(n) to O(n³), Pandas ops: O(n) to O(n²)"
      }
    },
    {
      title: "🗃️ Data Structures",
      id: "data-structures",
      description: "Efficient data organization for machine learning applications.",
      keyPoints: [
        "Lists: Ordered, mutable collections",
        "Dictionaries: Key-value pairs for fast lookup",
        "Arrays: Homogeneous numerical data",
        "Specialized structures (sets, tuples, deques)"
      ],
      detailedExplanation: [
        "Choosing the right structure:",
        "- Lists for ordered sequences of items",
        "- Dictionaries for labeled data access",
        "- NumPy arrays for numerical computations",
        "- Sets for unique element collections",
        "",
        "Performance characteristics:",
        "- Time complexity of common operations",
        "- Memory usage considerations",
        "- Cache locality for numerical data",
        "- Parallel processing compatibility",
        "",
        "ML-specific patterns:",
        "- Feature dictionaries for NLP",
        "- Batched data as lists of arrays",
        "- Lookup tables for embeddings",
        "- Circular buffers for streaming",
        "",
        "Advanced structures:",
        "- Defaultdict for counting",
        "- Namedtuples for readable code",
        "- Deques for sliding windows",
        "- Sparse matrices for NLP/cv"
      ],
      code: {
        python: `# Data Structures for ML
from collections import defaultdict, deque, namedtuple
import numpy as np

# Lists for batched data
batches = []
for i in range(0, len(data), batch_size):
    batches.append(data[i:i+batch_size])

# Dictionaries for model config
model_config = {
    'hidden_layers': [128, 64, 32],
    'activation': 'relu',
    'dropout': 0.2,
    'learning_rate': 0.001
}

# NumPy arrays for features
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = np.array([0, 1, 0])

# Set for vocabulary
vocab = set()
for text in corpus:
    vocab.update(text.split())

# Defaultdict for counting
word_counts = defaultdict(int)
for word in text_corpus:
    word_counts[word] += 1

# Namedtuple for readable code
ModelOutput = namedtuple('ModelOutput', ['prediction', 'confidence', 'embedding'])
output = ModelOutput(prediction=1, confidence=0.92, embedding=np.zeros(256))

# Deque for sliding window
window = deque(maxlen=5)
for data_point in stream:
    window.append(data_point)
    if len(window) == 5:
        process_window(list(window))`,
        complexity: "Lists: O(1) access, O(n) insert; Dicts: O(1) average case"
      }
    },
    {
      title: "📂 File Handling",
      id: "file-handling",
      description: "Reading and writing data in formats commonly used in ML pipelines.",
      keyPoints: [
        "CSV: Tabular data storage",
        "JSON: Structured configuration and data",
        "XML: Hierarchical data representation",
        "Binary formats (HDF5, Pickle, Parquet)"
      ],
      detailedExplanation: [
        "CSV for tabular data:",
        "- Reading/writing with Pandas",
        "- Handling large files with chunks",
        "- Dealing with missing values",
        "- Type inference and specification",
        "",
        "JSON for configuration:",
        "- Model hyperparameters",
        "- Experiment configurations",
        "- Metadata storage",
        "- Schema validation",
        "",
        "Binary formats:",
        "- HDF5 for large numerical datasets",
        "- Pickle for Python object serialization",
        "- Parquet for columnar storage",
        "- Protocol buffers for efficient serialization",
        "",
        "Best practices:",
        "- Memory mapping large files",
        "- Streaming processing",
        "- Compression options",
        "- Versioning and schema evolution"
      ],
      code: {
        python: `# File Handling for ML
import pandas as pd
import json
import pickle
import h5py

# CSV files
# Reading
data = pd.read_csv('dataset.csv', nrows=1000)  # Read first 1000 rows
chunked_data = pd.read_csv('large_dataset.csv', chunksize=10000)  # Stream in chunks

# Writing
data.to_csv('processed.csv', index=False)

# JSON files
# Reading config
with open('config.json') as f:
    config = json.load(f)

# Writing results
results = {'accuracy': 0.92, 'loss': 0.15}
with open('experiment_1.json', 'w') as f:
    json.dump(results, f, indent=2)

# Binary formats
# HDF5 for large arrays
with h5py.File('embeddings.h5', 'w') as f:
    f.create_dataset('embeddings', data=embeddings_array)

# Pickle for Python objects
with open('model.pkl', 'wb') as f:
    pickle.dump(trained_model, f)

# Parquet for efficient storage
data.to_parquet('data.parquet', engine='pyarrow')

# Handling XML (less common in ML)
import xml.etree.ElementTree as ET
tree = ET.parse('config.xml')
root = tree.getroot()
params = {child.tag: child.text for child in root}`,
        complexity: "CSV/JSON: O(n), Binary formats: O(n) with better constant factors"
      }
    },
    {
      title: "📊 Data Visualization",
      id: "visualization",
      description: "Exploring and communicating insights from ML data and results.",
      keyPoints: [
        "Histograms: Distribution of features",
        "Box plots: Statistical summaries",
        "Scatter plots: Relationships between variables",
        "Advanced plots (violin, pair, heatmaps)"
      ],
      detailedExplanation: [
        "Exploratory data analysis:",
        "- Identifying data distributions",
        "- Spotting outliers and anomalies",
        "- Visualizing feature relationships",
        "- Checking class balance",
        "",
        "Model evaluation visuals:",
        "- ROC curves and precision-recall",
        "- Confusion matrices",
        "- Learning curves",
        "- Feature importance plots",
        "",
        "Advanced techniques:",
        "- Interactive visualization (Plotly, Bokeh)",
        "- Large dataset visualization strategies",
        "- Custom matplotlib styling",
        "- Animation for model dynamics",
        "",
        "Best practices:",
        "- Choosing appropriate chart types",
        "- Effective labeling and legends",
        "- Color palette selection",
        "- Accessibility considerations"
      ],
      code: {
        python: `# Data Visualization for ML
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Histogram of features
plt.figure(figsize=(10,6))
plt.hist(data['feature_0'], bins=30, alpha=0.5, label='Feature 0')
plt.hist(data['feature_1'], bins=30, alpha=0.5, label='Feature 1')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Feature Distributions')
plt.legend()
plt.show()

# Box plot of model errors
plt.figure(figsize=(10,6))
sns.boxplot(x='model_type', y='error', data=results_df)
plt.title('Model Error Comparison')
plt.xticks(rotation=45)
plt.show()

# Scatter plot with regression
plt.figure(figsize=(10,6))
sns.regplot(x='feature_0', y='target', data=data, scatter_kws={'alpha':0.3})
plt.title('Feature-Target Relationship')
plt.show()

# Advanced visualizations
# Violin plot
plt.figure(figsize=(10,6))
sns.violinplot(x='class', y='feature_2', data=data, inner='quartile')
plt.title('Feature Distribution by Class')
plt.show()

# Heatmap
corr = data.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()

# Pair plot for multivariate analysis
sns.pairplot(data.sample(100), vars=['feature_0', 'feature_1', 'feature_2'], hue='target')
plt.suptitle('Multivariate Feature Relationships', y=1.02)
plt.show()`,
        complexity: "Basic plots: O(n), Complex plots: O(n²) for pairwise relationships"
      }
    }
  ];

  return (
    <div style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '2rem',
      background: 'linear-gradient(to bottom right, #ecfdf5, #f0fdf4)',
      borderRadius: '20px',
      boxShadow: '0 10px 30px rgba(0,0,0,0.1)'
    }}>
      <h1 style={{
        fontSize: '3.5rem',
        fontWeight: '800',
        textAlign: 'center',
        background: 'linear-gradient(to right, #059669, #10b981)',
        WebkitBackgroundClip: 'text',
        backgroundClip: 'text',
        color: 'transparent',
        marginBottom: '3rem'
      }}>
        Python for Machine Learning
      </h1>

      <div style={{
        backgroundColor: 'rgba(5, 150, 105, 0.1)',
        padding: '2rem',
        borderRadius: '12px',
        marginBottom: '3rem',
        borderLeft: '4px solid #059669'
      }}>
        <h2 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#059669',
          marginBottom: '1rem'
        }}>Programming for ML</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Python is the dominant language for machine learning due to its simplicity and rich ecosystem.
          This section covers essential Python programming concepts specifically tailored for ML workflows,
          from basic syntax to advanced data handling and visualization.
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
            border: '1px solid #d1fae5',
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
              color: '#059669'
            }}>{section.title}</h2>
            <button
              onClick={() => toggleSection(section.id)}
              style={{
                background: 'linear-gradient(to right, #059669, #10b981)',
                color: 'white',
                padding: '0.75rem 1.5rem',
                borderRadius: '8px',
                border: 'none',
                cursor: 'pointer',
                fontWeight: '600',
                transition: 'all 0.2s ease',
                ':hover': {
                  transform: 'scale(1.05)',
                  boxShadow: '0 5px 15px rgba(5, 150, 105, 0.4)'
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
                  color: '#059669',
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
                backgroundColor: '#f0fdf4',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#059669',
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
                backgroundColor: '#dcfce7',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#059669',
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
                  border: '2px solid #a7f3d0'
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
        border: '1px solid #d1fae5'
      }}>
        <h2 style={{
          fontSize: '2rem',
          fontWeight: '700',
          color: '#059669',
          marginBottom: '2rem'
        }}>Python Tools for ML Workflows</h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            textAlign: 'left'
          }}>
            <thead style={{
              backgroundColor: '#059669',
              color: 'white'
            }}>
              <tr>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Category</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Key Libraries</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>ML Application</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Performance Tip</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Numerical Computing", "NumPy, SciPy", "Linear algebra, optimization", "Use vectorized operations"],
                ["Data Handling", "Pandas, Polars", "Data cleaning, feature engineering", "Avoid row-wise operations"],
                ["Visualization", "Matplotlib, Seaborn", "EDA, model evaluation", "Use figure-level functions"],
                ["File I/O", "H5Py, PyArrow", "Large dataset storage", "Use memory mapping"],
                ["Advanced ML", "Scikit-learn, XGBoost", "Model training, evaluation", "Prefer fit-transform"]
              ].map((row, index) => (
                <tr key={index} style={{
                  backgroundColor: index % 2 === 0 ? '#f0fdf4' : 'white',
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
        border: '1px solid #d1fae5'
      }}>
        <h3 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#059669',
          marginBottom: '1.5rem'
        }}>ML Engineer's Best Practices</h3>
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
              color: '#059669',
              marginBottom: '0.75rem'
            }}>Python Coding Standards</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Follow PEP 8 style guide for consistent code
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Use type hints for better maintainability
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Document functions with docstrings
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Structure projects with modular packages
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
              color: '#059669',
              marginBottom: '0.75rem'
            }}>Performance Optimization</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Vectorization:</strong> Prefer NumPy over native Python loops<br/>
              <strong>Memory:</strong> Use generators for large datasets<br/>
              <strong>Parallelism:</strong> Leverage multiprocessing for CPU-bound tasks<br/>
              <strong>JIT:</strong> Consider Numba for numerical code
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
              color: '#059669',
              marginBottom: '0.75rem'
            }}>Advanced Python for ML</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Metaprogramming:</strong> Dynamic model generation<br/>
              <strong>Decorators:</strong> Timing, logging, validation<br/>
              <strong>Context Managers:</strong> Resource handling<br/>
              <strong>Descriptors:</strong> Custom model attributes
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Python;