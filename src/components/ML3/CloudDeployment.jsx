import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function CloudDeployment() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "☁️ AWS SageMaker",
      id: "sagemaker",
      description: "Amazon's fully managed service for building, training, and deploying ML models.",
      keyPoints: [
        "End-to-end ML workflow management",
        "Built-in algorithms and notebooks",
        "One-click deployment to endpoints",
        "AutoML capabilities (Autopilot)"
      ],
      detailedExplanation: [
        "Key Features:",
        "- Jupyter notebooks for experimentation",
        "- Distributed training across multiple instances",
        "- Model monitoring and A/B testing",
        "- Integration with other AWS services",
        "",
        "Workflow:",
        "1. Prepare data in S3",
        "2. Train model using SageMaker",
        "3. Deploy to endpoint",
        "4. Monitor performance",
        "",
        "Use Cases:",
        "- Large-scale model training",
        "- Production deployment pipelines",
        "- Managed AutoML solutions",
        "- Batch transform jobs"
      ],
      code: {
        python: `# SageMaker Deployment Example
import sagemaker
from sagemaker.sklearn import SKLearn
from sagemaker import Model

# Initialize session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Upload training data to S3
train_data = sagemaker_session.upload_data(
    path='data/train.csv', 
    bucket='my-ml-bucket',
    key_prefix='data'
)

# Create estimator
sklearn_estimator = SKLearn(
    entry_script='train.py',
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    framework_version='0.23-1',
    output_path=f's3://my-ml-bucket/output'
)

# Train model
sklearn_estimator.fit({'train': train_data})

# Deploy model
predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)

# Make prediction
result = predictor.predict([[1, 2, 3]])
print(result)`,
        complexity: "Deployment: O(1) API calls, Training: O(n) based on data size"
      }
    },
    {
      title: "☁️ GCP AI Platform",
      id: "gcp",
      description: "Google Cloud's unified platform for ML development and deployment.",
      keyPoints: [
        "Integrated with Google's AI services",
        "Supports TensorFlow, scikit-learn, XGBoost",
        "Vertex AI for end-to-end workflows",
        "Explainable AI tools"
      ],
      detailedExplanation: [
        "Key Components:",
        "- Notebooks: Managed Jupyter environments",
        "- Training: Custom and AutoML options",
        "- Prediction: Online and batch serving",
        "- Pipelines: ML workflow orchestration",
        "",
        "Advantages:",
        "- Tight integration with BigQuery",
        "- Pre-trained models via AI APIs",
        "- Advanced monitoring with Vertex AI",
        "- Explainability and fairness tools",
        "",
        "Deployment Options:",
        "- Online prediction for real-time",
        "- Batch prediction for large datasets",
        "- Custom containers for complex models",
        "- Edge deployment to IoT devices"
      ],
      code: {
        python: `# GCP AI Platform Deployment
from google.cloud import aiplatform

# Initialize client
aiplatform.init(project="my-project", location="us-central1")

# Create and run training job
job = aiplatform.CustomTrainingJob(
    display_name="my-training-job",
    script_path="train.py",
    container_uri="gcr.io/cloud-aiplatform/training/tf-cpu.2-3:latest",
    requirements=["scikit-learn"],
    model_serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/sklearn-cpu.0-24:latest"
)

model = job.run(
    machine_type="n1-standard-4",
    replica_count=1
)

# Deploy model
endpoint = model.deploy(
    machine_type="n1-standard-2",
    min_replica_count=1,
    max_replica_count=3
)

# Make prediction
prediction = endpoint.predict(instances=[[1, 2, 3]])
print(prediction)`,
        complexity: "Deployment: O(1) API calls, Training: O(n) based on data size"
      }
    },
    {
      title: "☁️ Azure ML",
      id: "azure",
      description: "Microsoft's enterprise-grade platform for ML lifecycle management.",
      keyPoints: [
        "Studio interface for no-code ML",
        "Automated machine learning (AutoML)",
        "MLOps for DevOps integration",
        "Azure Kubernetes Service (AKS) deployment"
      ],
      detailedExplanation: [
        "Core Features:",
        "- Designer: Drag-and-drop model building",
        "- Datasets: Versioned data management",
        "- Experiments: Track training runs",
        "- Pipelines: Reproducible workflows",
        "",
        "Deployment Options:",
        "- Real-time endpoints (ACI, AKS)",
        "- Batch endpoints for offline scoring",
        "- Edge modules for IoT devices",
        "- ONNX runtime for cross-platform",
        "",
        "Enterprise Capabilities:",
        "- Role-based access control",
        "- Private link for secure access",
        "- Model monitoring and drift detection",
        "- Integration with Power BI"
      ],
      code: {
        python: `# Azure ML Deployment
from azureml.core import Workspace, Experiment, Model
from azureml.core.webservice import AciWebservice, AksWebservice
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment

# Connect to workspace
ws = Workspace.from_config()

# Register model
model = Model.register(
    workspace=ws,
    model_path="model.pkl",
    model_name="sklearn-model",
    description="Scikit-learn model"
)

# Create inference config
env = Environment.from_conda_specification(
    name="sklearn-env",
    file_path="conda_dependencies.yml"
)

inference_config = InferenceConfig(
    entry_script="score.py",
    environment=env
)

# Deploy to ACI
aci_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1
)

service = Model.deploy(
    workspace=ws,
    name="my-sklearn-service",
    models=[model],
    inference_config=inference_config,
    deployment_config=aci_config
)

service.wait_for_deployment(show_output=True)
print(service.get_logs())`,
        complexity: "Deployment: O(1) API calls, Training: O(n) based on data size"
      }
    },
    {
      title: "🐳 Docker for ML",
      id: "docker",
      description: "Containerization approach for portable and reproducible ML deployments.",
      keyPoints: [
        "Package models with dependencies",
        "Consistent environments across stages",
        "Lightweight deployment option",
        "Integration with cloud services"
      ],
      detailedExplanation: [
        "Why Docker for ML:",
        "- Solve 'works on my machine' problems",
        "- Version control for model environments",
        "- Isolate dependencies between projects",
        "- Scale deployments horizontally",
        "",
        "Key Components:",
        "- Dockerfile: Environment specification",
        "- Images: Built containers",
        "- Containers: Running instances",
        "- Registries: Storage for images",
        "",
        "Best Practices:",
        "- Multi-stage builds to reduce size",
        "- .dockerignore to exclude files",
        "- Environment variables for config",
        "- Health checks for monitoring",
        "",
        "Cloud Integration:",
        "- AWS ECS/EKS",
        "- GCP Cloud Run/GKE",
        "- Azure Container Instances/Service"
      ],
      code: {
        dockerfile: `# Dockerfile for ML Model
# Build stage
FROM python:3.8-slim as builder

WORKDIR /app
COPY requirements.txt .

RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.8-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Model files
COPY model.pkl /app/model.pkl
COPY app.py /app/app.py

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]`,
        complexity: "Build: O(1), Deployment: O(1) per container"
      }
    },
    {
      title: "🔄 CI/CD for ML",
      id: "cicd",
      description: "Continuous integration and deployment pipelines for machine learning models.",
      keyPoints: [
        "Automated testing of ML code",
        "Model versioning and tracking",
        "Canary deployments for models",
        "Rollback strategies"
      ],
      detailedExplanation: [
        "ML Pipeline Components:",
        "- Data validation tests",
        "- Model training automation",
        "- Performance benchmarking",
        "- Approval gates for promotion",
        "",
        "Tools and Platforms:",
        "- GitHub Actions",
        "- GitLab CI/CD",
        "- Azure DevOps",
        "- CircleCI",
        "",
        "Best Practices:",
        "- Separate data and code pipelines",
        "- Model versioning with metadata",
        "- Automated performance testing",
        "- Blue-green deployments",
        "",
        "Challenges Specific to ML:",
        "- Large binary assets (models)",
        "- Reproducibility concerns",
        "- Data drift detection",
        "- Model explainability checks"
      ],
      code: {
        yaml: `# GitHub Actions CI/CD for ML
name: ML Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    - name: Run tests
      run: |
        pytest --cov=./ --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v1

  train:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    - name: Train model
      run: |
        python train.py
    - name: Save model
      uses: actions/upload-artifact@v2
      with:
        name: model
        path: model.pkl

  deploy:
    needs: train
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v2
    - uses: azure/login@v1
      with:
    - name: Deploy to Azure ML
      run: |
        pip install azureml-sdk
        python deploy.py`,
        complexity: "Varies by pipeline complexity: O(n) for testing, O(1) for deployment steps"
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
        Cloud Deployment for Machine Learning
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
        }}>Deployment and Real-World Projects → Cloud Deployment</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Modern machine learning deployments leverage cloud platforms for scalability, reliability, 
          and ease of management. This section covers the major cloud providers and best practices 
          for deploying ML models in production environments.
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
                }}>Core Features</h3>
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
                  color: '#0369a1',
                  marginBottom: '1rem'
                }}>Implementation Example</h3>
                <div style={{
                  borderRadius: '8px',
                  overflow: 'hidden',
                  border: '2px solid #7dd3fc'
                }}>
                  <SyntaxHighlighter
                    language={section.id === 'cicd' ? 'yaml' : section.id === 'docker' ? 'dockerfile' : 'python'}
                    style={tomorrow}
                    customStyle={{
                      padding: "1.5rem",
                      fontSize: "0.95rem",
                      background: "#f9f9f9",
                      borderRadius: "0.5rem",
                    }}
                  >
                    {section.id === 'cicd' ? section.code.yaml : 
                     section.id === 'docker' ? section.code.dockerfile : 
                     section.code.python}
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
        }}>Cloud ML Services Comparison</h2>
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
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Feature</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>AWS SageMaker</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>GCP AI Platform</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Azure ML</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Managed Notebooks", "✓", "✓", "✓"],
                ["AutoML", "✓", "✓", "✓"],
                ["Custom Training", "✓", "✓", "✓"],
                ["Model Registry", "✓", "✓", "✓"],
                ["Explainability", "✓", "✓ (Advanced)", "✓"],
                ["Edge Deployment", "✓", "✓", "✓"],
                ["Workflow Pipelines", "✓", "✓ (Vertex AI)", "✓"],
                ["Prebuilt Models", "Marketplace", "AI APIs", "Cognitive Services"],
                ["Best For", "AWS ecosystem users", "Google stack users", "Microsoft enterprise"]
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
        }}>ML Deployment Best Practices</h3>
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
            }}>Architecture Considerations</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Choose between real-time and batch processing based on needs
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Implement proper monitoring for model performance and drift
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Design for scalability from the beginning
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Plan for A/B testing and canary deployments
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
            }}>Cost Optimization</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Right-size instances:</strong> Match resources to workload needs<br/>
              <strong>Spot instances:</strong> Use for fault-tolerant workloads<br/>
              <strong>Auto-scaling:</strong> Scale down during low traffic<br/>
              <strong>Model optimization:</strong> Smaller models cost less to serve
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
            }}>Security & Compliance</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Data encryption:</strong> At rest and in transit<br/>
              <strong>Access control:</strong> Principle of least privilege<br/>
              <strong>Audit logging:</strong> Track all model access<br/>
              <strong>Compliance:</strong> GDPR, HIPAA, etc. as needed
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CloudDeployment;