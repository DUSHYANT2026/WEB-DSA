import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function SavingModels() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "🥒 Pickle Serialization",
      id: "pickle",
      description: "Python's native serialization protocol for saving and loading Python objects.",
      keyPoints: [
        "Built-in Python module (no additional dependencies)",
        "Can serialize most Python objects",
        "Supports protocol versions (latest is most efficient)",
        "Security considerations for untrusted sources"
      ],
      detailedExplanation: [
        "How Pickle works:",
        "- Converts Python objects to byte streams (pickling)",
        "- Reconstructs objects from byte streams (unpickling)",
        "- Uses stack-based virtual machine for reconstruction",
        "",
        "Best practices for ML:",
        "- Use highest protocol version (protocol=4 or 5)",
        "- Handle large models with pickle.HIGHEST_PROTOCOL",
        "- Consider security risks (never unpickle untrusted data)",
        "- Use with caution in production environments",
        "",
        "Performance characteristics:",
        "- Generally faster than JSON for complex objects",
        "- Creates smaller files than human-readable formats",
        "- Slower than joblib for large NumPy arrays",
        "- Not suitable for very large models (>4GB without protocol 5)"
      ],
      code: {
        python: `# Saving and Loading Models with Pickle
import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Train a sample model
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)
model = RandomForestClassifier()
model.fit(X, y)

# Save model to file
with open('model.pkl', 'wb') as f:  # 'wb' for write binary
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

# Load model from file
with open('model.pkl', 'rb') as f:  # 'rb' for read binary
    loaded_model = pickle.load(f)

# Verify the loaded model
print(loaded_model.predict(X[:5]))

# Advanced: Saving multiple objects
preprocessor = StandardScaler()
preprocessor.fit(X)

with open('pipeline.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'preprocessor': preprocessor,
        'metadata': {
            'training_date': '2023-07-15',
            'version': '1.0'
        }
    }, f)

# Loading multiple objects
with open('pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)
    loaded_model = pipeline['model']
    loaded_preprocessor = pipeline['preprocessor']
    metadata = pipeline['metadata']`,
        complexity: "Serialization: O(n), Deserialization: O(n)"
      }
    },
    {
      title: "📦 Joblib Serialization",
      id: "joblib",
      description: "Optimized serialization for Python objects containing large NumPy arrays.",
      keyPoints: [
        "Part of scikit-learn ecosystem (optimized for ML)",
        "More efficient than pickle for large NumPy arrays",
        "Supports memory mapping for large objects",
        "Parallel compression capabilities"
      ],
      detailedExplanation: [
        "Advantages over pickle:",
        "- Optimized for NumPy arrays (common in ML models)",
        "- Can memory map arrays for zero-copy loading",
        "- Supports compressed storage (zlib, lz4, etc.)",
        "- Parallel compression for faster saving",
        "",
        "Usage patterns:",
        "- Ideal for scikit-learn models and pipelines",
        "- Works well with large neural network weights",
        "- Suitable for production deployment",
        "- Commonly used in ML model serving",
        "",
        "Performance considerations:",
        "- Faster than pickle for models with large arrays",
        "- Compression can significantly reduce file size",
        "- Memory mapping enables efficient loading of large models",
        "- Parallel compression speeds up saving"
      ],
      code: {
        python: `# Saving and Loading Models with Joblib
from joblib import dump, load
from sklearn.svm import SVC
import numpy as np

# Train a sample model
X = np.random.rand(1000, 100)  # Larger dataset
y = np.random.randint(0, 2, 1000)
model = SVC(probability=True)
model.fit(X, y)

# Save model with compression
dump(model, 'model.joblib', compress=3, protocol=4)  # Medium compression

# Load model (with memory mapping for large files)
loaded_model = load('model.joblib', mmap_mode='r')

# Verify the loaded model
print(loaded_model.predict_proba(X[:5]))

# Advanced: Saving pipeline with parallel compression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', SVC())
])
pipeline.fit(X, y)

# Save with parallel compression
dump(pipeline, 'pipeline.joblib', compress=('zlib', 3), protocol=4)

# Loading with memory mapping
large_model = load('pipeline.joblib', mmap_mode='r')

# Custom serialization with joblib
def save_model_with_metadata(model, filepath, metadata=None):
    """Save model with additional metadata"""
    data = {
        'model': model,
        'metadata': metadata or {},
        'version': '1.0'
    }
    dump(data, filepath)

def load_model_with_metadata(filepath):
    """Load model with metadata"""
    data = load(filepath)
    return data['model'], data['metadata']`,
        complexity: "Serialization: O(n), Deserialization: O(n) (faster than pickle for arrays)"
      }
    },
    {
      title: "⚖️ Comparison & Best Practices",
      id: "comparison",
      description: "Choosing the right serialization method and following ML model deployment best practices.",
      keyPoints: [
        "Pickle vs Joblib: When to use each",
        "Version compatibility considerations",
        "Security implications of model serialization",
        "Production deployment patterns"
      ],
      detailedExplanation: [
        "Serialization Method Comparison:",
        "- Pickle: More general-purpose, better for non-NumPy objects",
        "- Joblib: Optimized for NumPy/scikit-learn, better for large arrays",
        "- ONNX/PMML: Cross-platform, but limited model support",
        "- Custom formats: Framework-specific (TensorFlow SavedModel, PyTorch .pt)",
        "",
        "Versioning and Compatibility:",
        "- Python version compatibility (pickle protocols)",
        "- Library version mismatches can cause errors",
        "- Strategies for backward compatibility",
        "- Using wrapper classes for version tolerance",
        "",
        "Security Best Practices:",
        "- Never load untrusted serialized models",
        "- Sign and verify model artifacts",
        "- Use secure storage for model files",
        "- Consider checksums for integrity verification",
        "",
        "Production Deployment:",
        "- Containerization with Docker",
        "- Model versioning strategies",
        "- A/B testing deployment patterns",
        "- Monitoring model performance in production"
      ],
      code: {
        python: `# Serialization Best Practices
import pickle
import joblib
import hashlib
import json
from datetime import datetime

# 1. Secure Serialization
def save_model_secure(model, filepath, secret_key):
    """Save model with integrity check"""
    # Serialize model
    model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Create checksum
    checksum = hashlib.sha256(model_bytes + secret_key.encode()).hexdigest()
    
    # Save with metadata
    with open(filepath, 'wb') as f:
        pickle.dump({
            'model': model_bytes,
            'checksum': checksum,
            'created_at': datetime.now().isoformat()
        }, f)

def load_model_secure(filepath, secret_key):
    """Load model with integrity verification"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Verify checksum
    expected_checksum = hashlib.sha256(data['model'] + secret_key.encode()).hexdigest()
    if expected_checksum != data['checksum']:
        raise ValueError("Model checksum verification failed!")
    
    return pickle.loads(data['model'])

# 2. Version Tolerant Serialization
class ModelWrapper:
    """Wrapper for version-tolerant serialization"""
    def __init__(self, model, metadata=None):
        self.model = model
        self.metadata = metadata or {}
        self.version = "1.1"
        self.created_at = datetime.now().isoformat()
    
    def save(self, filepath):
        """Save wrapped model"""
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load wrapped model with backward compatibility"""
        wrapper = joblib.load(filepath)
        if not hasattr(wrapper, 'version'):
            # Handle version 1.0 format
            wrapper.version = "1.0"
        return wrapper

# 3. Production Deployment Pattern
def deploy_model(model, model_name, version):
    """Standardized model deployment"""
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_v{version}_{timestamp}.joblib"
    joblib.dump(model, filename, compress=3)
    
    # Generate metadata
    metadata = {
        'model_name': model_name,
        'version': version,
        'deployed_at': timestamp,
        'dependencies': {
            'python': '3.8.10',
            'sklearn': '1.0.2',
            'numpy': '1.21.5'
        }
    }
    
    # Save metadata
    with open(f"{filename}.meta", 'w') as f:
        json.dump(metadata, f)
    
    return filename`,
        complexity: "Varies by implementation: Checksums O(n), Wrappers O(1)"
      }
    }
  ];

  return (
    <div style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '2rem',
      background: 'linear-gradient(to bottom right, #f0fdf4, #ecfdf5)',
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
        Saving and Loading ML Models
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
        }}>Deployment and Real-World Projects → Saving and Loading Models</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Proper model serialization is crucial for deploying machine learning models in production.
          This section covers the essential techniques for saving and loading models using Python's
          most common serialization libraries, with best practices for real-world applications.
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
                }}>Key Concepts</h3>
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
                backgroundColor: '#d1fae5',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#059669',
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
                backgroundColor: '#a7f3d0',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#059669',
                  marginBottom: '1rem'
                }}>Implementation</h3>
                <p style={{
                  color: '#374151',
                  fontWeight: '600',
                  marginBottom: '1rem',
                  fontSize: '1.1rem'
                }}>Complexity: {section.code.complexity}</p>
                <div style={{
                  borderRadius: '8px',
                  overflow: 'hidden',
                  border: '2px solid #059669'
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
        }}>Serialization Method Comparison</h2>
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
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Feature</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Pickle</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Joblib</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Best For</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Speed", "Fast", "Faster for large arrays", "Joblib for NumPy-heavy models"],
                ["File Size", "Medium", "Smaller with compression", "Joblib with compression"],
                ["Security", "Unsafe", "Unsafe", "Neither for untrusted data"],
                ["Python Objects", "All", "Most (optimized for arrays)", "Pickle for complex objects"],
                ["Parallelism", "No", "Yes (compression)", "Joblib for large models"],
                ["Memory Mapping", "No", "Yes", "Joblib for very large models"]
              ].map((row, index) => (
                <tr key={index} style={{
                  backgroundColor: index % 2 === 0 ? '#ecfdf5' : 'white',
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
        }}>Production Deployment Guidelines</h3>
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
            }}>Serialization Best Practices</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Always use highest protocol version for compatibility
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Include metadata (version, training date, metrics)
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Implement integrity checks (checksums, signatures)
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Consider security implications of deserialization
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
            }}>Model Versioning Strategy</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Semantic Versioning:</strong> MAJOR.MINOR.PATCH (e.g., 2.1.0)<br/>
              <strong>Timestamp Versioning:</strong> YYYYMMDD_HHMMSS (e.g., 20230715_143022)<br/>
              <strong>Hybrid Approach:</strong> v1.0.3_20230715<br/>
              <strong>Metadata Files:</strong> Include version info in separate JSON
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
            }}>Advanced Deployment Patterns</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Model Packages:</strong> Combine model + preprocessing in one artifact<br/>
              <strong>Containerization:</strong> Docker images with all dependencies<br/>
              <strong>Model Registries:</strong> Centralized storage and version control<br/>
              <strong>Canary Deployments:</strong> Gradual rollout to monitor performance
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SavingModels;