import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function Clustering() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "🔢 K-Means Clustering",
      id: "kmeans",
      description: "A centroid-based algorithm that partitions data into K distinct clusters.",
      keyPoints: [
        "Requires specifying number of clusters (K) beforehand",
        "Uses Euclidean distance as similarity measure",
        "Iteratively minimizes within-cluster variance",
        "Scalable to large datasets"
      ],
      detailedExplanation: [
        "Algorithm Steps:",
        "1. Randomly initialize K cluster centroids",
        "2. Assign each point to nearest centroid",
        "3. Recalculate centroids as mean of assigned points",
        "4. Repeat until convergence (no more changes)",
        "",
        "Key Parameters:",
        "- K: Number of clusters (use elbow method or silhouette score)",
        "- Initialization: k-means++ improves convergence",
        "- Max iterations: Prevent infinite loops",
        "- Tolerance: Convergence threshold",
        "",
        "Advantages:",
        "- Simple to implement and understand",
        "- Efficient for large datasets (O(n))",
        "- Works well with spherical clusters",
        "",
        "Limitations:",
        "- Sensitive to initial centroids",
        "- Assumes clusters of similar size",
        "- Struggles with non-convex shapes"
      ],
      code: {
        python: `# K-Means Clustering Example
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.concatenate([
    np.random.normal((0,0), 1, (300,2)),
    np.random.normal((5,5), 1, (300,2)),
    np.random.normal((10,0), 1, (300,2))
])

# Fit K-Means
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualize clusters
plt.figure(figsize=(10,6))
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', alpha=0.5)
plt.scatter(centroids[:,0], centroids[:,1], c='red', s=200, marker='x')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Elbow method to find optimal K
inertias = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10,6))
plt.plot(range(1,10), inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()`,
        complexity: "O(n*k*i*d) where n=samples, k=clusters, i=iterations, d=dimensions"
      }
    },
    {
      title: "🌳 Hierarchical Clustering",
      id: "hierarchical",
      description: "Builds a hierarchy of clusters either through agglomerative (bottom-up) or divisive (top-down) approaches.",
      keyPoints: [
        "Creates a dendrogram showing cluster relationships",
        "No need to specify number of clusters initially",
        "Agglomerative is more commonly used",
        "Different linkage criteria (ward, complete, average, single)"
      ],
      detailedExplanation: [
        "Agglomerative Approach:",
        "1. Start with each point as its own cluster",
        "2. Merge closest pairs of clusters iteratively",
        "3. Continue until all points are in one cluster",
        "",
        "Linkage Criteria:",
        "- Ward: Minimizes variance of merged clusters",
        "- Complete: Uses maximum distance between clusters",
        "- Average: Uses average distance between clusters",
        "- Single: Uses minimum distance between clusters",
        "",
        "Key Parameters:",
        "- Number of clusters (can cut dendrogram at desired level)",
        "- Distance metric (Euclidean, Manhattan, etc.)",
        "- Linkage method (affects cluster shapes)",
        "",
        "Advantages:",
        "- Visual dendrogram provides intuitive interpretation",
        "- Doesn't require specifying K beforehand",
        "- Can handle non-spherical clusters",
        "",
        "Limitations:",
        "- Computationally expensive (O(n³))",
        "- Sensitive to noise and outliers",
        "- Once merged, clusters cannot be split"
      ],
      code: {
        python: `# Hierarchical Clustering Example
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.concatenate([
    np.random.normal((0,0), 1, (100,2)),
    np.random.normal((5,5), 1, (100,2)),
    np.random.normal((10,0), 1, (100,2))
])

# Fit Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X)

# Visualize clusters
plt.figure(figsize=(10,6))
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.title('Agglomerative Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Create and plot dendrogram
linked = linkage(X, 'ward')
plt.figure(figsize=(12,6))
dendrogram(linked, orientation='top', distance_sort='descending')
plt.title('Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# Different linkage methods comparison
linkage_methods = ['ward', 'complete', 'average', 'single']
plt.figure(figsize=(15,10))
for i, method in enumerate(linkage_methods, 1):
    plt.subplot(2,2,i)
    agg = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = agg.fit_predict(X)
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
    plt.title(f'Linkage: {method}')
plt.tight_layout()
plt.show()`,
        complexity: "O(n³) time, O(n²) space for standard implementations"
      }
    },
    {
      title: "🌌 DBSCAN (Density-Based Clustering)",
      id: "dbscan",
      description: "Density-based algorithm that identifies clusters as areas of high density separated by areas of low density.",
      keyPoints: [
        "Does not require specifying number of clusters",
        "Can find arbitrarily shaped clusters",
        "Identifies outliers as noise points",
        "Based on core points, border points, and noise"
      ],
      detailedExplanation: [
        "Key Concepts:",
        "- Core point: Has at least min_samples neighbors within ε distance",
        "- Border point: Has fewer than min_samples but is reachable from a core point",
        "- Noise point: Neither core nor border point",
        "",
        "Algorithm Steps:",
        "1. Randomly select an unvisited point",
        "2. Find all density-reachable points (ε and min_samples)",
        "3. If core point, form a cluster",
        "4. Repeat until all points are visited",
        "",
        "Key Parameters:",
        "- ε (eps): Maximum distance between points",
        "- min_samples: Minimum points to form dense region",
        "- Metric: Distance calculation method",
        "",
        "Advantages:",
        "- Can find arbitrarily shaped clusters",
        "- Robust to outliers",
        "- Doesn't require specifying number of clusters",
        "",
        "Limitations:",
        "- Struggles with varying density clusters",
        "- Sensitive to parameter choices",
        "- Not suitable for high-dimensional data"
      ],
      code: {
        python: `# DBSCAN Example
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data with non-spherical clusters
np.random.seed(42)
n = 200
theta = np.linspace(0, 2*np.pi, n)
circle1 = np.column_stack([np.cos(theta), np.sin(theta)]) * 2 + np.random.normal(0, 0.1, (n,2))
circle2 = np.column_stack([np.cos(theta), np.sin(theta)]) * 5 + np.random.normal(0, 0.1, (n,2))
X = np.vstack([circle1, circle2])

# Fit DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# Visualize clusters
plt.figure(figsize=(10,6))
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Parameter sensitivity analysis
plt.figure(figsize=(15,10))
for i, eps in enumerate([0.2, 0.5, 1.0, 2.0], 1):
    plt.subplot(2,2,i)
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(X)
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
    plt.title(f'EPS: {eps}, Clusters: {len(set(labels))-1}')
plt.tight_layout()
plt.show()

# Handling noise points
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)
core_samples = dbscan.core_sample_indices_
noise = labels == -1

plt.figure(figsize=(10,6))
plt.scatter(X[~noise,0], X[~noise,1], c=labels[~noise], cmap='viridis')
plt.scatter(X[noise,0], X[noise,1], c='red', marker='x', label='Noise')
plt.title('DBSCAN with Noise Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()`,
        complexity: "O(n log n) with spatial indexing, O(n²) without"
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
        Clustering Algorithms
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
        }}>Unsupervised Learning → Clustering</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Clustering groups similar data points together without predefined labels.
          These algorithms discover inherent patterns and structures in data,
          enabling applications like customer segmentation, anomaly detection,
          and data exploration.
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
                backgroundColor: '#e0f2fe',
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
        }}>Clustering Algorithm Comparison</h2>
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
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Cluster Shape</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Scalability</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Noise Handling</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Best Use Case</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["K-Means", "Spherical", "High (O(n))", "Poor", "Large datasets with clear separation"],
                ["Hierarchical", "Arbitrary (depends on linkage)", "Low (O(n³))", "Moderate", "Small datasets, need hierarchy"],
                ["DBSCAN", "Arbitrary", "Moderate (O(n log n))", "Excellent", "Noisy data, arbitrary shapes"]
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
          color: '#0369a1',
          marginBottom: '1.5rem'
        }}>Practical Considerations</h3>
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
            }}>Algorithm Selection Guide</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                <strong>K-Means:</strong> When you know K and need speed
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                <strong>Hierarchical:</strong> When you need cluster relationships
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                <strong>DBSCAN:</strong> When dealing with noise and arbitrary shapes
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                <strong>GMM:</strong> When clusters may overlap (not covered here)
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
            }}>Preprocessing Tips</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Normalization:</strong> Essential for distance-based algorithms<br/>
              <strong>Dimensionality Reduction:</strong> Helps with high-dimensional data<br/>
              <strong>Outlier Handling:</strong> Critical for centroid-based methods<br/>
              <strong>Feature Selection:</strong> Improves cluster interpretability
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
            }}>Evaluation Methods</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Silhouette Score:</strong> Measures cluster cohesion/separation<br/>
              <strong>Davies-Bouldin Index:</strong> Lower values indicate better clustering<br/>
              <strong>Calinski-Harabasz Index:</strong> Ratio of between/within cluster dispersion<br/>
              <strong>Visual Inspection:</strong> Always validate with domain knowledge
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Clustering;