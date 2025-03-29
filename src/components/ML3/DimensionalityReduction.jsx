import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function DimensionalityReduction() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "📊 Principal Component Analysis (PCA)",
      id: "pca",
      description: "Linear dimensionality reduction technique that projects data onto directions of maximum variance.",
      keyPoints: [
        "Orthogonal transformation to uncorrelated components",
        "Components ordered by explained variance",
        "Sensitive to feature scaling",
        "Assumes linear relationships"
      ],
      detailedExplanation: [
        "Mathematical foundation:",
        "- Computes eigenvectors of covariance matrix",
        "- Projects data onto principal components",
        "- Can be viewed as minimizing projection error",
        "",
        "Key parameters:",
        "- n_components: Number of components to keep",
        "- whiten: Whether to normalize component scales",
        "- svd_solver: Algorithm for computation",
        "",
        "Applications in ML:",
        "- Noise reduction in high-dimensional data",
        "- Visualization of high-D datasets",
        "- Speeding up learning algorithms",
        "- Removing multicollinearity in features",
        "",
        "Limitations:",
        "- Only captures linear relationships",
        "- Sensitive to outliers",
        "- Interpretation can be challenging",
        "- Global structure preservation only"
      ],
      code: {
        python: `# PCA Implementation Example
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.dot(np.random.rand(100, 2), np.random.rand(2, 10))  # 100 samples, 10 features

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize results
plt.figure(figsize=(10,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Projection')
plt.show()

# Explained variance
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# Cumulative explained variance
plt.figure(figsize=(10,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid()
plt.show()`,
        complexity: "O(n³) for exact solver, O(nd²) for randomized (n samples, d features)"
      }
    },
    {
      title: "🌌 t-SNE (t-Distributed Stochastic Neighbor Embedding)",
      id: "tsne",
      description: "Non-linear technique particularly well-suited for visualizing high-dimensional data in 2D or 3D.",
      keyPoints: [
        "Preserves local neighborhood structure",
        "Probabilistic approach using Student-t distribution",
        "Excellent for visualization",
        "Computationally intensive"
      ],
      detailedExplanation: [
        "How it works:",
        "- Models pairwise similarities in high-D and low-D space",
        "- Uses heavy-tailed distribution to avoid crowding problem",
        "- Minimizes KL divergence between distributions",
        "",
        "Key parameters:",
        "- perplexity: Balances local/global aspects (~5-50)",
        "- learning_rate: Typically 10-1000",
        "- n_iter: Number of iterations (at least 250)",
        "- metric: Distance metric ('euclidean', 'cosine')",
        "",
        "Applications:",
        "- Visualizing clusters in high-D data",
        "- Exploring neural network representations",
        "- Understanding feature relationships",
        "- Data exploration before modeling",
        "",
        "Limitations:",
        "- Not suitable for feature preprocessing",
        "- Results vary with hyperparameters",
        "- No global structure preservation",
        "- Cannot transform new data"
      ],
      code: {
        python: `# t-SNE Implementation Example
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data (e.g., word embeddings)
np.random.seed(42)
X = np.random.randn(200, 50)  # 200 samples, 50 features

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualize results
plt.figure(figsize=(10,6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7)
plt.title('t-SNE Projection')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid()
plt.show()

# With color-coded classes (if available)
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)`,
        complexity: "O(n²) memory, O(n² log n) time (n samples)"
      }
    },
    {
      title: "🌐 UMAP (Uniform Manifold Approximation and Projection)",
      id: "umap",
      description: "Modern non-linear dimensionality reduction technique that preserves both local and global structure.",
      keyPoints: [
        "Based on Riemannian geometry and algebraic topology",
        "Faster than t-SNE with similar quality",
        "Can transform new data",
        "Preserves more global structure than t-SNE"
      ],
      detailedExplanation: [
        "Theoretical foundations:",
        "- Models data as a fuzzy topological structure",
        "- Optimizes low-dimensional representation",
        "- Uses stochastic gradient descent",
        "",
        "Key parameters:",
        "- n_neighbors: Balances local/global structure (~5-50)",
        "- min_dist: Controls clustering tightness (0.1-0.5)",
        "- metric: Distance metric ('euclidean', 'cosine')",
        "- n_components: Output dimensions (typically 2-3)",
        "",
        "Advantages over t-SNE:",
        "- Better preservation of global structure",
        "- Faster computation",
        "- Ability to transform new data",
        "- More stable embeddings",
        "",
        "Applications:",
        "- General-purpose dimensionality reduction",
        "- Visualization of complex datasets",
        "- Preprocessing for clustering",
        "- Feature extraction"
      ],
      code: {
        python: `# UMAP Implementation Example
from umap import UMAP
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(300, 40)  # 300 samples, 40 features

# Apply UMAP
umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean')
X_umap = umap.fit_transform(X)

# Visualize results
plt.figure(figsize=(10,6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.7)
plt.title('UMAP Projection')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.grid()
plt.show()

# Transform new data (unlike t-SNE)
new_data = np.random.randn(10, 40)
new_embedding = umap.transform(new_data)`,
        complexity: "O(n^1.14) empirically, much faster than t-SNE for large n"
      }
    },
    {
      title: "🔍 LDA (Linear Discriminant Analysis)",
      id: "lda",
      description: "Supervised dimensionality reduction that maximizes class separability.",
      keyPoints: [
        "Projects data to maximize between-class variance",
        "Minimizes within-class variance",
        "Assumes normal distribution of features",
        "Limited by number of classes"
      ],
      detailedExplanation: [
        "Comparison with PCA:",
        "- PCA is unsupervised, LDA is supervised",
        "- PCA maximizes variance, LDA maximizes class separation",
        "- LDA limited to (n_classes - 1) dimensions",
        "",
        "Mathematical formulation:",
        "- Computes between-class scatter matrix",
        "- Computes within-class scatter matrix",
        "- Solves generalized eigenvalue problem",
        "",
        "Applications:",
        "- Preprocessing for classification tasks",
        "- Feature extraction when labels are available",
        "- Improving model performance on small datasets",
        "- Reducing overfitting in supervised learning",
        "",
        "Limitations:",
        "- Assumes Gaussian class distributions",
        "- Sensitive to outliers",
        "- Requires labeled data",
        "- Limited dimensionality reduction"
      ],
      code: {
        python: `# LDA Implementation Example
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data with classes
np.random.seed(42)
X = np.vstack([
    np.random.randn(50, 20) + 1,  # Class 0
    np.random.randn(50, 20) - 1,  # Class 1
    np.random.randn(50, 20) * 2   # Class 2
])
y = np.array([0]*50 + [1]*50 + [2]*50)

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Visualize results
plt.figure(figsize=(10,6))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('LDA Projection')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.colorbar(label='Class')
plt.grid()
plt.show()

# Classification after LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2f}")`,
        complexity: "O(nd² + d³) where d is original dimension"
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
        Dimensionality Reduction Techniques
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
        }}>Unsupervised Learning → Dimensionality Reduction</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Dimensionality reduction techniques transform high-dimensional data into lower-dimensional spaces
          while preserving important structure. These methods are essential for visualization, noise reduction,
          and improving the efficiency of machine learning algorithms.
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
                backgroundColor: '#ecfdf5',
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
                }}>Computational Complexity: {section.code.complexity}</p>
                <div style={{
                  borderRadius: '8px',
                  overflow: 'hidden',
                  border: '2px solid #bae6fd'
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
        }}>Dimensionality Reduction Techniques Comparison</h2>
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
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Type</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Preserves</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Best For</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["PCA", "Linear", "Global variance", "Numerical data, linear relationships"],
                ["t-SNE", "Non-linear", "Local structure", "Visualization, clustering"],
                ["UMAP", "Non-linear", "Local & some global", "General-purpose reduction"],
                ["LDA", "Supervised linear", "Class separation", "Preprocessing for classification"]
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
        backgroundColor: '#ecfdf5',
        borderRadius: '16px',
        boxShadow: '0 5px 15px rgba(0,0,0,0.05)',
        border: '1px solid #a7f3d0'
      }}>
        <h3 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#0369a1',
          marginBottom: '1.5rem'
        }}>Practical Guidance</h3>
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
            }}>Choosing a Technique</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                <strong>PCA:</strong> When you need fast, linear reduction and feature extraction
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                <strong>t-SNE:</strong> For visualizing high-dimensional clusters and patterns
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                <strong>UMAP:</strong> When you need both local and global structure preservation
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                <strong>LDA:</strong> When you have labeled data and want to maximize class separation
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
              <strong>Always scale your data</strong> before applying linear methods like PCA<br/>
              <strong>Experiment with hyperparameters</strong> (perplexity in t-SNE, n_neighbors in UMAP)<br/>
              <strong>Visualize explained variance</strong> to choose the right number of components<br/>
              <strong>Consider computational complexity</strong> for large datasets (use IncrementalPCA or randomized SVD)
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
              <strong>Manifold learning:</strong> Combining multiple techniques<br/>
              <strong>Autoencoders:</strong> Neural networks for non-linear reduction<br/>
              <strong>Kernel PCA:</strong> Non-linear extensions of PCA<br/>
              <strong>Multidimensional scaling:</strong> Preserving distances between samples
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DimensionalityReduction;