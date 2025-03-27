import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function FlaskFastAPI() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "⚗️ Flask for ML Deployment",
      id: "flask",
      description: "Lightweight Python web framework ideal for simple ML model serving.",
      keyPoints: [
        "Minimalist and flexible microframework",
        "Simple REST API creation",
        "WSGI-based synchronous architecture",
        "Large ecosystem of extensions"
      ],
      detailedExplanation: [
        "Why Flask for ML:",
        "- Quick to set up for prototyping",
        "- Easy integration with Python ML stack",
        "- Minimal overhead for simple services",
        "- Well-established in production environments",
        "",
        "Core Components:",
        "- Route decorators for API endpoints",
        "- Request/response handling",
        "- Template rendering (for demo UIs)",
        "- Extension system (for database, auth, etc.)",
        "",
        "Deployment Patterns:",
        "- Standalone server for development",
        "- Gunicorn + Nginx for production",
        "- Docker containers for portability",
        "- Serverless deployments (AWS Lambda, etc.)",
        "",
        "Performance Considerations:",
        "- Synchronous nature limits throughput",
        "- Global interpreter lock (GIL) constraints",
        "- Optimal for low-to-medium traffic",
        "- Horizontal scaling recommended"
      ],
      code: {
        python: `# Flask ML Deployment Example
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json()
    
    # Convert to numpy array and reshape
    features = np.array(data['features']).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return prediction as JSON
    return jsonify({
        'prediction': prediction[0].item(),
        'status': 'success'
    })

@app.route('/')
def home():
    return "ML Model Serving API - Ready for predictions"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# Production setup with Gunicorn:
# gunicorn -w 4 -b :5000 app:app

# Dockerfile example:
# FROM python:3.8-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install -r requirements.txt
# COPY . .
# CMD ["gunicorn", "-w", "4", "-b", ":5000", "app:app"]`,
        complexity: "Setup: O(1), Request handling: O(model complexity)"
      }
    },
    {
      title: "⚡ FastAPI for ML Deployment",
      id: "fastapi",
      description: "Modern, high-performance framework for building ML APIs with Python.",
      keyPoints: [
        "ASGI-based asynchronous support",
        "Automatic OpenAPI/Swagger documentation",
        "Data validation with Pydantic",
        "High performance (comparable to NodeJS/Go)"
      ],
      detailedExplanation: [
        "Why FastAPI for ML:",
        "- Built-in async support for concurrent requests",
        "- Automatic API documentation",
        "- Type hints for better code quality",
        "- Excellent performance characteristics",
        "",
        "Key Features:",
        "- Dependency injection system",
        "- Background tasks for post-processing",
        "- WebSocket support for real-time apps",
        "- Easy integration with ML libraries",
        "",
        "Deployment Options:",
        "- Uvicorn/ASGI servers for production",
        "- Kubernetes for scaling",
        "- Serverless deployments",
        "- Edge deployments with compiled Python",
        "",
        "Performance Advantages:",
        "- Handles more concurrent requests",
        "- Lower latency for I/O bound tasks",
        "- Efficient background processing",
        "- Better vertical scaling"
      ],
      code: {
        python: `# FastAPI ML Deployment Example
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load trained model
model = joblib.load('model.joblib')

# Define request model
class PredictionRequest(BaseModel):
    features: list[float]

@app.post('/predict')
async def predict(request: PredictionRequest):
    # Convert to numpy array
    features = np.array(request.features).reshape(1, -1)
    
    # Make prediction (async if model supports it)
    prediction = model.predict(features)
    
    return {
        'prediction': prediction[0].item(),
        'status': 'success'
    }

@app.get('/')
async def health_check():
    return {"status": "ready"}

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000

# Dockerfile example:
# FROM python:3.8-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install -r requirements.txt
# COPY . .
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# For production:
# uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4`,
        complexity: "Setup: O(1), Request handling: O(model complexity) with better concurrency"
      }
    },
    {
      title: "🔄 Comparing Flask and FastAPI",
      id: "comparison",
      description: "Choosing the right framework based on your ML deployment needs.",
      keyPoints: [
        "Flask: Simpler, more mature, synchronous",
        "FastAPI: Faster, async, modern features",
        "Development speed vs production performance",
        "Community support and learning curve"
      ],
      detailedExplanation: [
        "When to Choose Flask:",
        "- Simple prototypes and MVPs",
        "- Teams with existing Flask expertise",
        "- Applications requiring Jinja2 templating",
        "- Projects with many Flask extensions",
        "",
        "When to Choose FastAPI:",
        "- High-performance API requirements",
        "- Async/await patterns in your code",
        "- Automatic API documentation needs",
        "- Type-heavy codebases",
        "",
        "Performance Benchmarks:",
        "- FastAPI handles 2-3x more requests/sec",
        "- Lower latency under concurrent load",
        "- Better resource utilization",
        "- More efficient I/O handling",
        "",
        "Ecosystem Considerations:",
        "- Flask has more third-party extensions",
        "- FastAPI has built-in modern features",
        "- Both integrate well with ML stack",
        "- Deployment patterns are similar"
      ],
      code: {
        python: `# Hybrid Approach: Using FastAPI with Flask-style routes
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

# Flask-style route
@app.get("/flask-style/")
def flask_style_route():
    return JSONResponse(content={"message": "This looks like Flask!"})

# Async route
@app.get("/fastapi-style/")
async def fastapi_style_route():
    return {"message": "This leverages FastAPI async"}

# Migration Tips:
# 1. Replace Flask's jsonify with FastAPI's return dicts
# 2. Convert route decorators (@app.route → @app.get/post)
# 3. Use Pydantic models instead of manual request parsing
# 4. Leverage dependency injection over global variables`,
        complexity: "Migration: O(n) where n is route complexity"
      }
    },
    {
      title: "🚀 Advanced Deployment Patterns",
      id: "advanced",
      description: "Production-grade strategies for serving ML models at scale.",
      keyPoints: [
        "Containerization with Docker",
        "Orchestration with Kubernetes",
        "Load testing and auto-scaling",
        "Monitoring and logging"
      ],
      detailedExplanation: [
        "Containerization Best Practices:",
        "- Multi-stage builds to reduce image size",
        "- Non-root users for security",
        "- Health checks and readiness probes",
        "- Environment-specific configurations",
        "",
        "Scaling Strategies:",
        "- Horizontal pod autoscaling in Kubernetes",
        "- Queue-based workload distribution",
        "- Model caching and warm-up",
        "- Canary deployments for model updates",
        "",
        "Performance Optimization:",
        "- Model quantization for faster inference",
        "- Batch prediction endpoints",
        "- Async processing for heavy models",
        "- GPU acceleration in containers",
        "",
        "Observability:",
        "- Prometheus metrics integration",
        "- Distributed tracing with Jaeger",
        "- Structured logging (JSON format)",
        "- Alerting on prediction latency/drift"
      ],
      code: {
        python: `# Advanced FastAPI Setup with Monitoring
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import logging
import time

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Logging configuration
logging.basicConfig(
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"Method={request.method} Path={request.url.path} "
        f"Status={response.status_code} Duration={process_time:.2f}ms"
    )
    return response

# Kubernetes Deployment Example:
# apiVersion: apps/v1
# kind: Deployment
# metadata:
#   name: ml-api
# spec:
#   replicas: 3
#   selector:
#     matchLabels:
#       app: ml-api
#   template:
#     metadata:
#       labels:
#         app: ml-api
#     spec:
#       containers:
#       - name: ml-api
#         image: your-registry/ml-api:latest
#         ports:
#         - containerPort: 8000
#         resources:
#           limits:
#             cpu: "1"
#             memory: "1Gi"
#         readinessProbe:
#           httpGet:
#             path: /
#             port: 8000
#           initialDelaySeconds: 5
#           periodSeconds: 10`,
        complexity: "Setup: O(1), Maintenance: O(n) for cluster size"
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
        Flask & FastAPI for ML Deployment
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
        }}>Deployment and Real-World Projects → Flask and FastAPI</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Flask and FastAPI are the two most popular Python frameworks for deploying machine learning models
          as web services. This section covers their features, trade-offs, and production deployment patterns.
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
                }}>Key Features</h3>
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
        }}>Framework Comparison</h2>
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
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Flask</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>FastAPI</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Architecture", "WSGI (Synchronous)", "ASGI (Asynchronous)"],
                ["Performance", "Good for low concurrency", "Excellent for high concurrency"],
                ["Learning Curve", "Gentle, simple concepts", "Steeper (async concepts)"],
                ["Documentation", "Manual or extensions", "Automatic OpenAPI/Swagger"],
                ["Data Validation", "Manual or extensions", "Built-in with Pydantic"],
                ["Best For", "Simple APIs, prototypes", "High-performance APIs, production"],
                ["Community", "Large, mature", "Growing rapidly"],
                ["Extensions", "Very extensive", "Smaller but growing"]
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
        }}>Deployment Best Practices</h3>
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
            }}>Choosing Between Flask and FastAPI</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                <strong>Choose Flask</strong> when you need simplicity, templating, or have existing Flask expertise
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                <strong>Choose FastAPI</strong> when you need performance, async support, or automatic docs
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Both can be containerized and deployed similarly
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Consider team skills and project requirements
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
            }}>Production Deployment Checklist</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>1. Containerization:</strong> Use Docker with multi-stage builds<br/>
              <strong>2. Orchestration:</strong> Kubernetes for scaling and management<br/>
              <strong>3. Monitoring:</strong> Prometheus metrics and logging<br/>
              <strong>4. Security:</strong> HTTPS, rate limiting, input validation<br/>
              <strong>5. Performance:</strong> Load testing and optimization
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
            }}>Advanced Considerations</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Model Versioning:</strong> Endpoints for multiple model versions<br/>
              <strong>Canary Deployments:</strong> Gradually roll out new models<br/>
              <strong>Feature Stores:</strong> Consistent feature engineering<br/>
              <strong>Shadow Mode:</strong> Test new models against production traffic
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default FlaskFastAPI;