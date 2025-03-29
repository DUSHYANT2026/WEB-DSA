import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function Classification() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "📊 Logistic Regression",
      id: "logistic",
      description: "A fundamental linear classification algorithm that models probabilities using a sigmoid function.",
      keyPoints: [
        "Binary classification using sigmoid activation",
        "Linear decision boundary",
        "Outputs class probabilities",
        "Regularized variants (L1/L2)"
      ],
      detailedExplanation: [
        "How it works:",
        "- Models log-odds as linear combination of features",
        "- Applies sigmoid to get probabilities between 0 and 1",
        "- Uses maximum likelihood estimation for training",
        "",
        "Key advantages:",
        "- Computationally efficient",
        "- Provides probabilistic interpretation",
        "- Works well with linearly separable data",
        "- Feature importance through coefficients",
        "",
        "Limitations:",
        "- Assumes linear relationship between features and log-odds",
        "- Can underfit complex patterns",
        "- Sensitive to correlated features",
        "",
        "Hyperparameters:",
        "- Regularization strength (C)",
        "- Penalty type (L1/L2/elasticnet)",
        "- Solver algorithm (liblinear, saga, etc.)"
      ],
      code: {
        python: `# Logistic Regression Example
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load binary classification data
X, y = load_iris(return_X_y=True)
X = X[y != 2]  # Use only two classes
y = y[y != 2]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Training accuracy: {train_acc:.2f}")
print(f"Test accuracy: {test_acc:.2f}")

# Get probabilities
probs = model.predict_proba(X_test)
print("Class probabilities for first sample:", probs[0])

# Feature importance
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)`,
        complexity: "Training: O(n_samples × n_features), Prediction: O(n_features)"
      }
    },
    {
      title: "🌳 Decision Trees",
      id: "trees",
      description: "Non-parametric models that learn hierarchical decision rules from data.",
      keyPoints: [
        "Recursive binary splitting of feature space",
        "Split criteria: Gini impurity or entropy",
        "Prone to overfitting without regularization",
        "Can handle non-linear relationships"
      ],
      detailedExplanation: [
        "Learning process:",
        "- Start with all data at root node",
        "- Find best feature and threshold to split on",
        "- Recursively split until stopping criterion met",
        "",
        "Split criteria:",
        "- Gini impurity: Probability of misclassification",
        "- Information gain: Reduction in entropy",
        "- Variance reduction (for regression)",
        "",
        "Advantages:",
        "- No need for feature scaling",
        "- Handles mixed data types",
        "- Interpretable decision rules",
        "- Feature importance scores",
        "",
        "Regularization:",
        "- Maximum depth",
        "- Minimum samples per leaf",
        "- Minimum impurity decrease",
        "- Cost complexity pruning"
      ],
      code: {
        python: `# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Create and train model
tree = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    criterion='gini',
    random_state=42
)
tree.fit(X_train, y_train)

# Visualize the tree
plt.figure(figsize=(12,8))
plot_tree(tree, feature_names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], 
          class_names=['setosa', 'versicolor'], filled=True)
plt.show()

# Feature importance
importances = tree.feature_importances_
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.3f}")

# Evaluate
print(f"Training accuracy: {tree.score(X_train, y_train):.2f}")
print(f"Test accuracy: {tree.score(X_test, y_test):.2f}")`,
        complexity: "Training: O(n_samples × n_features × depth), Prediction: O(depth)"
      }
    },
    {
      title: "🌲 Random Forest",
      id: "forest",
      description: "Ensemble method that combines multiple decision trees via bagging.",
      keyPoints: [
        "Builds many trees on random subsets of data/features",
        "Averages predictions for better generalization",
        "Reduces variance compared to single trees",
        "Built-in feature importance"
      ],
      detailedExplanation: [
        "How it works:",
        "- Bootstrap sampling creates many training subsets",
        "- Each tree trained on random feature subset",
        "- Final prediction by majority vote (classification)",
        "",
        "Key benefits:",
        "- Handles high dimensional spaces well",
        "- Robust to outliers and noise",
        "- Parallelizable training",
        "- Doesn't require feature scaling",
        "",
        "Tuning parameters:",
        "- Number of trees",
        "- Maximum depth",
        "- Minimum samples per leaf",
        "- Maximum features per split",
        "",
        "Extensions:",
        "- Extremely Randomized Trees (ExtraTrees)",
        "- Feature importance scores",
        "- Out-of-bag error estimation",
        "- Partial dependence plots"
      ],
      code: {
        python: `# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Create and train model
forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1  # Use all cores
)
forest.fit(X_train, y_train)

# Evaluate
print("Training accuracy:", forest.score(X_train, y_train))
print("Test accuracy:", forest.score(X_test, y_test))
print(classification_report(y_test, forest.predict(X_test)))

# Feature importance
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

# Plot feature importance
plt.figure(figsize=(10,6))
plt.barh(features, importances, xerr=std, align='center')
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.show()`,
        complexity: "Training: O(n_trees × n_samples × n_features × depth), Prediction: O(n_trees × depth)"
      }
    },
    {
      title: "⚡ Support Vector Machines (SVM)",
      id: "svm",
      description: "Powerful classifiers that find optimal separating hyperplanes in high-dimensional spaces.",
      keyPoints: [
        "Finds maximum-margin decision boundary",
        "Kernel trick for non-linear classification",
        "Effective in high-dimensional spaces",
        "Memory intensive for large datasets"
      ],
      detailedExplanation: [
        "Key concepts:",
        "- Support vectors: Critical training instances",
        "- Margin: Distance between classes",
        "- Kernel functions: Implicit feature transformations",
        "",
        "Kernel types:",
        "- Linear: No transformation",
        "- Polynomial: Captures polynomial relationships",
        "- RBF: Handles complex non-linear boundaries",
        "- Sigmoid: Neural network-like transformation",
        "",
        "Advantages:",
        "- Effective in high dimensions",
        "- Versatile with different kernels",
        "- Robust to overfitting in high-D spaces",
        "",
        "Practical considerations:",
        "- Scaling features is critical",
        "- Regularization parameter C controls margin",
        "- Kernel choice affects performance",
        "- Can be memory intensive"
      ],
      code: {
        python: `# SVM Classifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create pipeline with scaling and SVM
svm = make_pipeline(
    StandardScaler(),
    SVC(
        kernel='rbf', 
        C=1.0,
        gamma='scale',
        probability=True  # Enable predict_proba
    )
)

# Train model
svm.fit(X_train, y_train)

# Evaluate
print(f"Training accuracy: {svm.score(X_train, y_train):.2f}")
print(f"Test accuracy: {svm.score(X_test, y_test):.2f}")

# Get support vectors
if hasattr(svm.named_steps['svc'], 'support_vectors_'):
    print(f"Number of support vectors: {len(svm.named_steps['svc'].support_vectors_)}")

# Plot decision boundary (for 2D data)
def plot_decision_boundary(clf, X, y):
    # Create grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Predict on grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title("SVM Decision Boundary")
    plt.show()

# For 2D data only:
# plot_decision_boundary(svm, X_train[:, :2], y_train)`,
        complexity: "Training: O(n_samples² to n_samples³), Prediction: O(n_support_vectors × n_features)"
      }
    },
    {
      title: "🧠 Neural Networks for Classification",
      id: "nn",
      description: "Flexible function approximators that can learn complex decision boundaries.",
      keyPoints: [
        "Multi-layer perceptrons (MLPs) for classification",
        "Backpropagation for training",
        "Non-linear activation functions",
        "Requires careful hyperparameter tuning"
      ],
      detailedExplanation: [
        "Architecture components:",
        "- Input layer (feature dimension)",
        "- Hidden layers with non-linear activations",
        "- Output layer with softmax (multi-class) or sigmoid (binary)",
        "",
        "Key hyperparameters:",
        "- Number and size of hidden layers",
        "- Activation functions (ReLU, tanh, etc.)",
        "- Learning rate and optimizer",
        "- Regularization (dropout, weight decay)",
        "",
        "Training process:",
        "- Forward pass computes predictions",
        "- Loss function measures error",
        "- Backpropagation computes gradients",
        "- Optimization updates weights",
        "",
        "Practical considerations:",
        "- Feature scaling is essential",
        "- Batch normalization helps training",
        "- Early stopping prevents overfitting",
        "- Architecture search is important"
      ],
      code: {
        python: `# Neural Network Classifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Create pipeline with scaling and MLP
mlp = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,  # L2 regularization
        batch_size=32,
        learning_rate='adaptive',
        max_iter=200,
        early_stopping=True,
        random_state=42
    )
)

# Train model
mlp.fit(X_train, y_train)

# Evaluate
print(f"Training accuracy: {mlp.score(X_train, y_train):.2f}")
print(f"Test accuracy: {mlp.score(X_test, y_test):.2f}")

# Loss curve
plt.plot(mlp.named_steps['mlpclassifier'].loss_curve_)
plt.title("Loss Curve During Training")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()

# Using Keras/TensorFlow
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=32,
                    validation_split=0.2)
"""`,
        complexity: "Training: O(n_samples × n_features × width × depth × epochs), Prediction: O(width × depth × n_features)"
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
        Classification Algorithms
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
        }}>Supervised Learning → Classification</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Classification algorithms predict discrete class labels from input features. 
          This section covers fundamental classification methods from linear models to 
          deep neural networks, with practical implementation examples.
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
                backgroundColor: '#e0f2fe',
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
                backgroundColor: '#f0f9ff',
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
          color: '#0ea5e9',
          marginBottom: '2rem'
        }}>Classification Algorithm Comparison</h2>
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
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Algorithm</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Strengths</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Weaknesses</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Best Use Cases</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Logistic Regression", "Fast, interpretable, probabilistic", "Linear decision boundary", "Baseline, linearly separable data"],
                ["Decision Trees", "Non-linear, feature importance", "Prone to overfitting", "Interpretable models, mixed data types"],
                ["Random Forest", "Robust, handles non-linearity", "Less interpretable", "General purpose, feature importance"],
                ["SVM", "Effective in high-D, kernel trick", "Memory intensive", "Small/medium datasets, clear margin"],
                ["Neural Networks", "Flexible, state-of-the-art", "Data hungry, complex tuning", "Complex patterns, large datasets"]
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
        backgroundColor: '#e0f2fe',
        borderRadius: '16px',
        boxShadow: '0 5px 15px rgba(0,0,0,0.05)',
        border: '1px solid #bae6fd'
      }}>
        <h3 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#0ea5e9',
          marginBottom: '1.5rem'
        }}>Classification Best Practices</h3>
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
            }}>Model Selection Guidelines</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Start simple (logistic regression) before trying complex models
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Use tree-based models for interpretability and mixed data types
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Choose neural networks for complex patterns and large datasets
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Consider SVM for small/medium datasets with clear margins
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
            }}>Evaluation Metrics</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              Beyond accuracy, consider:
              <br/><br/>
              - <strong>Precision/Recall</strong>: For imbalanced classes<br/>
              - <strong>ROC AUC</strong>: Overall ranking performance<br/>
              - <strong>F1 Score</strong>: Balance of precision and recall<br/>
              - <strong>Confusion Matrix</strong>: Detailed error analysis
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
            }}>Practical Tips</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              - Scale features for SVM and neural networks<br/>
              - Handle class imbalance with weighting or resampling<br/>
              - Use cross-validation for reliable performance estimates<br/>
              - Monitor training/validation curves for overfitting
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Classification;