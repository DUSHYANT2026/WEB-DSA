import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";
import { useTheme } from "../../ThemeContext.jsx";

function LinearAlgebra() {
  const { darkMode } = useTheme();
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "📐 Vectors and Matrices",
      id: "vectors-matrices",
      description:
        "Fundamental building blocks for representing data and transformations in machine learning.",
      keyPoints: [
        "Vectors: Ordered collections of numbers representing points in space",
        "Matrices: Rectangular arrays for representing linear transformations",
        "Special matrices: Identity, diagonal, symmetric, orthogonal",
        "Vector spaces and subspaces in ML contexts",
      ],
      detailedExplanation: [
        "In machine learning:",
        "- Feature vectors represent data points (n-dimensional vectors)",
        "- Weight matrices transform data between layers in neural networks",
        "- Similarity between vectors (cosine similarity) used in recommendation systems",
        "",
        "Key operations:",
        "- Dot product measures similarity between vectors",
        "- Matrix-vector multiplication applies transformations",
        "- Hadamard (element-wise) product used in attention mechanisms",
        "",
        "Important properties:",
        "- Linear independence determines model capacity",
        "- Span defines the space reachable by combinations",
        "- Norms (L1, L2) used in regularization",
      ],
      code: {
        python: `# Vectors and Matrices in ML with NumPy
import numpy as np

# Feature vector (4 features)
sample = np.array([5.1, 3.5, 1.4, 0.2])  

# Weight matrix (3x4) for a layer with 3 neurons
weights = np.random.randn(3, 4)  

# Apply transformation
transformed = weights @ sample  # Matrix-vector multiplication

# Cosine similarity between two samples
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# L2 regularization
def l2_regularization(weights, lambda_):
    return lambda_ * np.sum(weights ** 2)`,
        complexity:
          "Vector ops: O(n), Matrix-vector: O(mn), Matrix-matrix: O(mnp)",
      },
    },
    {
      title: "🔄 Matrix Operations",
      id: "operations",
      description:
        "Essential operations for manipulating and understanding transformations in ML models.",
      keyPoints: [
        "Transpose: Flipping rows and columns (Aᵀ)",
        "Inverse: Matrix that reverses transformation (A⁻¹)",
        "Trace: Sum of diagonal elements",
        "Matrix multiplication: Composition of transformations",
      ],
      detailedExplanation: [
        "In machine learning contexts:",
        "- Transpose used in backpropagation equations",
        "- Pseudoinverse for solving overdetermined systems",
        "- Matrix multiplication as fundamental NN operation",
        "",
        "Special cases:",
        "- Orthogonal matrices preserve distances (QᵀQ = I)",
        "- Diagonal matrices for efficient computations",
        "- Triangular matrices in decomposition methods",
        "",
        "Computational considerations:",
        "- Broadcasting rules in vectorized implementations",
        "- Sparse matrices for memory efficiency",
        "- GPU acceleration for large matrices",
      ],
      code: {
        python: `# Matrix Operations in ML Context
import numpy as np

# Transpose for gradient calculation
X = np.array([[1, 2], [3, 4], [5, 6]])  # Design matrix (3x2)
y = np.array([0.5, 1.2, 2.1])           # Targets (3,)

# Linear regression weights: (XᵀX)⁻¹Xᵀy
XT = X.T
XTX = XT @ X
XTX_inv = np.linalg.inv(XTX)
weights = XTX_inv @ XT @ y

# Efficient computation using solve
weights = np.linalg.solve(XTX, XT @ y)

# Batch matrix multiplication for neural networks
# Input batch (100 samples, 64 features)
batch = np.random.randn(100, 64)  
# Weight matrix (64 features → 32 neurons)
W = np.random.randn(64, 32)       
# Output (100, 32)
output = batch @ W`,
        complexity: "Inversion: O(n³), Solve: O(n³), Matmul: O(mnp)",
      },
    },
    {
      title: "🔍 Eigenvalues and Eigenvectors",
      id: "eigen",
      description:
        "Characteristic directions and scaling factors of linear transformations.",
      keyPoints: [
        "Eigenvectors: Directions unchanged by transformation",
        "Eigenvalues: Scaling factors along eigenvector directions",
        "Diagonalization: Decomposing matrices into simpler forms",
        "Positive definite matrices in optimization",
      ],
      detailedExplanation: [
        "Machine learning applications:",
        "- Principal Component Analysis (PCA) for dimensionality reduction",
        "- Eigenfaces for facial recognition",
        "- Spectral clustering in unsupervised learning",
        "- Stability analysis of learning algorithms",
        "",
        "Key concepts:",
        "- Characteristic polynomial: det(A - λI) = 0",
        "- Eigendecomposition: A = QΛQ⁻¹",
        "- Power iteration method for dominant eigenpair",
        "",
        "Special cases:",
        "- Markov chains and stationary distributions",
        "- Google's PageRank algorithm",
        "- Hessian matrix in optimization",
      ],
      code: {
        python: `# Eigenanalysis in ML
import numpy as np
from sklearn.decomposition import PCA

# Covariance matrix from data
data = np.random.randn(100, 10)  # 100 samples, 10 features
cov = np.cov(data, rowvar=False)

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(cov)

# PCA using SVD (more numerically stable)
pca = PCA(n_components=2)
reduced = pca.fit_transform(data)

# Power iteration for dominant eigenvector
def power_iteration(A, num_iterations=100):
    b = np.random.rand(A.shape[1])
    for _ in range(num_iterations):
        b = A @ b
        b = b / np.linalg.norm(b)
    eigenvalue = b.T @ A @ b
    return eigenvalue, b

# Hessian matrix analysis
def hessian(f, x, eps=1e-5):
    n = x.shape[0]
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x1 = x.copy()
            x1[i] += eps
            x1[j] += eps
            x2 = x.copy()
            x2[i] += eps
            x2[j] -= eps
            x3 = x.copy()
            x3[i] -= eps
            x3[j] += eps
            x4 = x.copy()
            x4[i] -= eps
            x4[j] -= eps
            hess[i,j] = (f(x1)-f(x2)-f(x3)+f(x4))/(4*eps*eps)
    return hess`,
        complexity:
          "Full eigendecomposition: O(n³), Power iteration: O(n² per iteration)",
      },
    },
    {
      title: "🌀 Singular Value Decomposition (SVD)",
      id: "svd",
      description:
        "Fundamental matrix factorization with wide applications in machine learning.",
      keyPoints: [
        "Generalization of eigendecomposition to non-square matrices",
        "Singular values: Non-negative scaling factors",
        "Orthonormal basis for row and column spaces",
        "Low-rank approximations for dimensionality reduction",
      ],
      detailedExplanation: [
        "ML applications:",
        "- Dimensionality reduction (PCA implementation)",
        "- Collaborative filtering (recommender systems)",
        "- Latent semantic analysis in NLP",
        "- Matrix completion problems",
        "",
        "Mathematical formulation:",
        "- A = UΣVᵀ where U and V are orthogonal",
        "- Σ contains singular values in descending order",
        "- Truncated SVD keeps top k singular values",
        "",
        "Computational aspects:",
        "- Randomized SVD for large matrices",
        "- Relationship to PCA: SVD on centered data",
        "- Regularization via singular value thresholding",
      ],
      code: {
        python: `# SVD Applications in ML
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Recommendation system matrix (users x items)
ratings = np.random.randint(0, 5, size=(100, 50))  # 100 users, 50 items
ratings = ratings.astype(float)

# Fill missing values with mean
mean = np.nanmean(ratings, axis=0)
ratings = np.where(np.isnan(ratings), mean, ratings)

# Perform SVD
U, s, Vt = np.linalg.svd(ratings, full_matrices=False)

# Truncated SVD for dimensionality reduction
svd = TruncatedSVD(n_components=10)
reduced = svd.fit_transform(ratings)

# Matrix completion via SVD
def complete_matrix(matrix, rank, num_iters=10):
    completed = np.where(np.isnan(matrix), 0, matrix)
    for _ in range(num_iters):
        U, s, Vt = np.linalg.svd(completed, full_matrices=False)
        s[rank:] = 0
        completed = U @ np.diag(s) @ Vt
        completed = np.where(np.isnan(matrix), completed, matrix)
    return completed

# Image compression example
from skimage import data
image = data.camera()  # 512x512 grayscale image
U, s, Vt = np.linalg.svd(image)
k = 50  # Keep top 50 singular values
compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]`,
        complexity: "Full SVD: O(min(mn², m²n)), Randomized SVD: O(mnk)",
      },
    },
  ];

  return (
    <div
      style={{
        maxWidth: "1200px",
        margin: "0 auto",
        padding: "2rem",
        background: darkMode
          ? "linear-gradient(to bottom right, #1e1b4b, #1e1b4b)"
          : "linear-gradient(to bottom right, #f0f4ff, #f9f0ff)",
        borderRadius: "20px",
        boxShadow: darkMode
          ? "0 10px 30px rgba(0,0,0,0.3)"
          : "0 10px 30px rgba(0,0,0,0.1)",
        color: darkMode ? "#e2e8f0" : "#1e293b",
      }}
    >
      <h1
        style={{
          fontSize: "3.5rem",
          fontWeight: "800",
          textAlign: "center",
          background: "linear-gradient(to right, #4f46e5, #7c3aed)",
          WebkitBackgroundClip: "text",
          backgroundClip: "text",
          color: "transparent",
          marginBottom: "3rem",
        }}
      >
        Linear Algebra for Machine Learning
      </h1>

      <div
        style={{
          backgroundColor: darkMode
            ? "rgba(79, 70, 229, 0.2)"
            : "rgba(79, 70, 229, 0.1)",
          padding: "2rem",
          borderRadius: "12px",
          marginBottom: "3rem",
          borderLeft: "4px solid #4f46e5",
        }}
      >
        <h2
          style={{
            fontSize: "1.8rem",
            fontWeight: "700",
            color: "#4f46e5",
            marginBottom: "1rem",
          }}
        >
          Mathematics for ML → Linear Algebra
        </h2>
        <p
          style={{
            color: darkMode ? "#e2e8f0" : "#374151",
            fontSize: "1.1rem",
            lineHeight: "1.6",
          }}
        >
          Linear algebra forms the mathematical foundation for machine learning
          algorithms. This section covers the essential concepts with direct
          applications to ML models, including vector/matrix operations,
          eigendecomposition, and SVD.
        </p>
      </div>

      {content.map((section) => (
        <div
          key={section.id}
          style={{
            marginBottom: "3rem",
            padding: "2rem",
            backgroundColor: darkMode ? "#1e293b" : "white",
            borderRadius: "16px",
            boxShadow: darkMode
              ? "0 5px 15px rgba(0,0,0,0.3)"
              : "0 5px 15px rgba(0,0,0,0.05)",
            transition: "all 0.3s ease",
            border: darkMode ? "1px solid #334155" : "1px solid #e0e7ff",
            ":hover": {
              boxShadow: darkMode
                ? "0 8px 25px rgba(0,0,0,0.4)"
                : "0 8px 25px rgba(0,0,0,0.1)",
              transform: "translateY(-2px)",
            },
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "1.5rem",
              cursor: "pointer",
            }}
            onClick={() => toggleSection(section.id)}
          >
            <h2
              style={{
                fontSize: "2rem",
                fontWeight: "700",
                color: "#4f46e5",
              }}
            >
              {section.title}
            </h2>
            <div
              style={{
                width: "36px",
                height: "36px",
                borderRadius: "50%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                backgroundColor: darkMode
                  ? "rgba(79, 70, 229, 0.2)"
                  : "rgba(79, 70, 229, 0.1)",
                transition: "all 0.3s ease",
                transform:
                  visibleSection === section.id
                    ? "rotate(180deg)"
                    : "rotate(0deg)",
                ":hover": {
                  transform:
                    visibleSection === section.id
                      ? "rotate(180deg) scale(1.1)"
                      : "rotate(0deg) scale(1.1)",
                  backgroundColor: darkMode
                    ? "rgba(79, 70, 229, 0.3)"
                    : "rgba(79, 70, 229, 0.2)",
                },
              }}
            >
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M6 9L12 15L18 9"
                  stroke={darkMode ? "#e2e8f0" : "#4f46e5"}
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>
          </div>

          {visibleSection === section.id && (
            <div style={{ display: "grid", gap: "2rem" }}>
              <div
                style={{
                  backgroundColor: darkMode ? "#1e3a8a" : "#f0f9ff",
                  padding: "1.5rem",
                  borderRadius: "12px",
                }}
              >
                <h3
                  style={{
                    fontSize: "1.5rem",
                    fontWeight: "600",
                    color: "#4f46e5",
                    marginBottom: "1rem",
                  }}
                >
                  Core Concepts
                </h3>
                <p
                  style={{
                    color: darkMode ? "#e2e8f0" : "#374151",
                    fontSize: "1.1rem",
                    lineHeight: "1.6",
                    marginBottom: "1rem",
                  }}
                >
                  {section.description}
                </p>
                <ul
                  style={{
                    listStyleType: "disc",
                    paddingLeft: "1.5rem",
                    display: "grid",
                    gap: "0.5rem",
                  }}
                >
                  {section.keyPoints.map((point, index) => (
                    <li
                      key={index}
                      style={{
                        color: darkMode ? "#e2e8f0" : "#374151",
                        fontSize: "1.1rem",
                      }}
                    >
                      {point}
                    </li>
                  ))}
                </ul>
              </div>

              <div
                style={{
                  backgroundColor: darkMode ? "#064e3b" : "#f0fdf4",
                  padding: "1.5rem",
                  borderRadius: "12px",
                }}
              >
                <h3
                  style={{
                    fontSize: "1.5rem",
                    fontWeight: "600",
                    color: "#4f46e5",
                    marginBottom: "1rem",
                  }}
                >
                  Technical Deep Dive
                </h3>
                <div style={{ display: "grid", gap: "1rem" }}>
                  {section.detailedExplanation.map((paragraph, index) => (
                    <p
                      key={index}
                      style={{
                        color: darkMode ? "#e2e8f0" : "#374151",
                        fontSize: "1.1rem",
                        lineHeight: "1.6",
                        margin: paragraph === "" ? "0.5rem 0" : "0",
                      }}
                    >
                      {paragraph}
                    </p>
                  ))}
                </div>
              </div>

              <div
                style={{
                  backgroundColor: darkMode ? "#4c1d95" : "#f5f3ff",
                  padding: "1.5rem",
                  borderRadius: "12px",
                }}
              >
                <h3
                  style={{
                    fontSize: "1.5rem",
                    fontWeight: "600",
                    color: "#4f46e5",
                    marginBottom: "1rem",
                  }}
                >
                  ML Implementation
                </h3>
                <p
                  style={{
                    color: darkMode ? "#e2e8f0" : "#374151",
                    fontWeight: "600",
                    marginBottom: "1rem",
                    fontSize: "1.1rem",
                  }}
                >
                  {section.code.complexity}
                </p>
                <div
                  style={{
                    borderRadius: "8px",
                    overflow: "hidden",
                    border: darkMode
                      ? "2px solid #5b21b6"
                      : "2px solid #e9d5ff",
                  }}
                >
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
      <div
        style={{
          marginTop: "3rem",
          padding: "2rem",
          backgroundColor: darkMode ? "#1e293b" : "white",
          borderRadius: "16px",
          boxShadow: darkMode
            ? "0 5px 15px rgba(0,0,0,0.3)"
            : "0 5px 15px rgba(0,0,0,0.05)",
          border: darkMode ? "1px solid #334155" : "1px solid #e0e7ff",
        }}
      >
        <h2
          style={{
            fontSize: "2rem",
            fontWeight: "700",
            color: "#4f46e5",
            marginBottom: "2rem",
          }}
        >
          Linear Algebra in ML: Key Concepts
        </h2>
        <div style={{ overflowX: "auto" }}>
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              textAlign: "left",
            }}
          >
            <thead
              style={{
                backgroundColor: "#4f46e5",
                color: "white",
              }}
            >
              <tr>
                <th
                  style={{
                    padding: "1rem",
                    fontSize: "1.1rem",
                    fontWeight: "600",
                  }}
                >
                  Concept
                </th>
                <th
                  style={{
                    padding: "1rem",
                    fontSize: "1.1rem",
                    fontWeight: "600",
                  }}
                >
                  ML Application
                </th>
                <th
                  style={{
                    padding: "1rem",
                    fontSize: "1.1rem",
                    fontWeight: "600",
                  }}
                >
                  Example Use Case
                </th>
                <th
                  style={{
                    padding: "1rem",
                    fontSize: "1.1rem",
                    fontWeight: "600",
                  }}
                >
                  Key Libraries
                </th>
              </tr>
            </thead>
            <tbody>
              {[
                [
                  "Vectors",
                  "Feature representation",
                  "Word embeddings in NLP",
                  "NumPy, PyTorch",
                ],
                [
                  "Matrix Operations",
                  "Neural network layers",
                  "Fully connected layers",
                  "TensorFlow, JAX",
                ],
                [
                  "Eigendecomposition",
                  "Dimensionality reduction",
                  "PCA for feature extraction",
                  "scikit-learn",
                ],
                [
                  "SVD",
                  "Recommendation systems",
                  "Collaborative filtering",
                  "Surprise, LightFM",
                ],
              ].map((row, index) => (
                <tr
                  key={index}
                  style={{
                    backgroundColor:
                      index % 2 === 0
                        ? darkMode
                          ? "#334155"
                          : "#f8fafc"
                        : darkMode
                        ? "#1e293b"
                        : "white",
                    borderBottom: darkMode
                      ? "1px solid #334155"
                      : "1px solid #e2e8f0",
                  }}
                >
                  {row.map((cell, cellIndex) => (
                    <td
                      key={cellIndex}
                      style={{
                        padding: "1rem",
                        color: darkMode ? "#e2e8f0" : "#334155",
                      }}
                    >
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
      <div
        style={{
          marginTop: "3rem",
          padding: "2rem",
          backgroundColor: darkMode ? "#1e3a8a" : "#fff7ed",
          borderRadius: "16px",
          boxShadow: darkMode
            ? "0 5px 15px rgba(0,0,0,0.3)"
            : "0 5px 15px rgba(0,0,0,0.05)",
          border: darkMode ? "1px solid #1e40af" : "1px solid #ffedd5",
        }}
      >
        <h3
          style={{
            fontSize: "1.8rem",
            fontWeight: "700",
            color: "#4f46e5",
            marginBottom: "1.5rem",
          }}
        >
          ML Practitioner's Perspective
        </h3>
        <div style={{ display: "grid", gap: "1.5rem" }}>
          <div
            style={{
              backgroundColor: darkMode ? "#1e293b" : "white",
              padding: "1.5rem",
              borderRadius: "12px",
              boxShadow: darkMode
                ? "0 2px 8px rgba(0,0,0,0.3)"
                : "0 2px 8px rgba(0,0,0,0.05)",
            }}
          >
            <h4
              style={{
                fontSize: "1.3rem",
                fontWeight: "600",
                color: "#4f46e5",
                marginBottom: "0.75rem",
              }}
            >
              Essential Linear Algebra for ML
            </h4>
            <ul
              style={{
                listStyleType: "disc",
                paddingLeft: "1.5rem",
                display: "grid",
                gap: "0.75rem",
              }}
            >
              <li
                style={{
                  color: darkMode ? "#e2e8f0" : "#374151",
                  fontSize: "1.1rem",
                }}
              >
                Vector/matrix operations form the backbone of neural networks
              </li>
              <li
                style={{
                  color: darkMode ? "#e2e8f0" : "#374151",
                  fontSize: "1.1rem",
                }}
              >
                Eigendecomposition powers dimensionality reduction techniques
              </li>
              <li
                style={{
                  color: darkMode ? "#e2e8f0" : "#374151",
                  fontSize: "1.1rem",
                }}
              >
                SVD enables efficient matrix approximations in large systems
              </li>
              <li
                style={{
                  color: darkMode ? "#e2e8f0" : "#374151",
                  fontSize: "1.1rem",
                }}
              >
                Understanding these concepts helps debug and optimize models
              </li>
            </ul>
          </div>

          <div
            style={{
              backgroundColor: darkMode ? "#1e293b" : "white",
              padding: "1.5rem",
              borderRadius: "12px",
              boxShadow: darkMode
                ? "0 2px 8px rgba(0,0,0,0.3)"
                : "0 2px 8px rgba(0,0,0,0.05)",
            }}
          >
            <h4
              style={{
                fontSize: "1.3rem",
                fontWeight: "600",
                color: "#4f46e5",
                marginBottom: "0.75rem",
              }}
            >
              Computational Considerations
            </h4>
            <p
              style={{
                color: darkMode ? "#e2e8f0" : "#374151",
                fontSize: "1.1rem",
                lineHeight: "1.6",
              }}
            >
              Modern ML implementations leverage:
              <br />
              <br />
              <strong>Vectorization:</strong> Using matrix operations instead of
              loops
              <br />
              <strong>GPU Acceleration:</strong> Parallel processing of large
              matrices
              <br />
              <strong>Sparse Representations:</strong> For high-dimensional data
              <br />
              <strong>Numerical Stability:</strong> Careful implementation to
              avoid errors
            </p>
          </div>

          <div
            style={{
              backgroundColor: darkMode ? "#1e293b" : "white",
              padding: "1.5rem",
              borderRadius: "12px",
              boxShadow: darkMode
                ? "0 2px 8px rgba(0,0,0,0.3)"
                : "0 2px 8px rgba(0,0,0,0.05)",
            }}
          >
            <h4
              style={{
                fontSize: "1.3rem",
                fontWeight: "600",
                color: "#4f46e5",
                marginBottom: "0.75rem",
              }}
            >
              Advanced Applications
            </h4>
            <p
              style={{
                color: darkMode ? "#e2e8f0" : "#374151",
                fontSize: "1.1rem",
                lineHeight: "1.6",
              }}
            >
              <strong>Graph Neural Networks:</strong> Adjacency matrix
              operations
              <br />
              <strong>Attention Mechanisms:</strong> Matrix products for
              similarity
              <br />
              <strong>Kernel Methods:</strong> High-dimensional feature spaces
              <br />
              <strong>Optimization:</strong> Hessian matrix in second-order
              methods
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default LinearAlgebra;
