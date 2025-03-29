import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function MLPipeline() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "📊 Data Collection",
      id: "collection",
      description: "The foundation of any ML project - acquiring and organizing raw data.",
      keyPoints: [
        "Identifying relevant data sources",
        "Web scraping and APIs for data acquisition",
        "Data labeling strategies (manual, semi-supervised, active learning)",
        "Legal and ethical considerations"
      ],
      detailedExplanation: [
        "Key aspects of data collection:",
        "- Volume: Ensuring sufficient data for training",
        "- Variety: Capturing diverse scenarios and edge cases",
        "- Veracity: Maintaining data quality and accuracy",
        "- Velocity: Handling streaming vs batch data",
        "",
        "Common data sources:",
        "- Public datasets (Kaggle, UCI, government data)",
        "- Internal company databases and logs",
        "- Web scraping (with proper permissions)",
        "- IoT devices and sensors",
        "- User-generated content",
        "",
        "Tools and techniques:",
        "- Scrapy, BeautifulSoup for web scraping",
        "- LabelImg, CVAT for image annotation",
        "- Prodigy, Label Studio for text annotation",
        "- Synthetic data generation when real data is scarce"
      ],
      code: {
        python: `# Data Collection Examples
import requests
import pandas as pd
from bs4 import BeautifulSoup

# API Data Collection
def fetch_weather_data(api_key, location):
    url = f"https://api.weather.com/v3/{location}"
    response = requests.get(url, params={'key': api_key})
    return response.json()

# Web Scraping
def scrape_news_headlines():
    url = "https://news.example.com"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = [h.text for h in soup.select('.headline')]
    return pd.DataFrame(headlines, columns=['headline'])

# Synthetic Data Generation
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3)

# Data Labeling (simple example)
def label_sentiment(text):
    positive_words = ['good', 'great', 'excellent']
    negative_words = ['bad', 'terrible', 'awful']
    if any(word in text.lower() for word in positive_words):
        return 'positive'
    elif any(word in text.lower() for word in negative_words):
        return 'negative'
    return 'neutral'`,
        complexity: "Varies by source: APIs O(1), Web scraping O(n), Synthetic data O(m*n)"
      }
    },
    {
      title: "🧹 Data Preprocessing",
      id: "preprocessing",
      description: "Transforming raw data into a format suitable for machine learning models.",
      keyPoints: [
        "Handling missing values (imputation, deletion)",
        "Outlier detection and treatment",
        "Feature scaling (normalization, standardization)",
        "Categorical encoding (one-hot, label, embeddings)"
      ],
      detailedExplanation: [
        "Common preprocessing steps:",
        "- Data cleaning: Handling missing values, duplicates, inconsistencies",
        "- Feature engineering: Creating new informative features",
        "- Transformation: Scaling, log transforms, power transforms",
        "- Dimensionality reduction: PCA, t-SNE, feature selection",
        "",
        "Specialized techniques:",
        "- Text: Tokenization, stemming, lemmatization",
        "- Images: Resizing, normalization, augmentation",
        "- Time series: Resampling, windowing, differencing",
        "- Audio: Spectrogram conversion, MFCC extraction",
        "",
        "Implementation considerations:",
        "- Train-test split before preprocessing to avoid leakage",
        "- Creating reproducible preprocessing pipelines",
        "- Handling skewed/imbalanced data",
        "- Maintaining preprocessing artifacts for inference"
      ],
      code: {
        python: `# Data Preprocessing Pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Sample dataset
data = pd.DataFrame({
    'age': [25, 30, None, 40, 45],
    'income': [50000, 60000, 70000, None, 90000],
    'gender': ['M', 'F', 'M', 'F', 'M'],
    'purchased': [1, 0, 1, 1, 0]
})

# Define preprocessing
numeric_features = ['age', 'income']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['gender']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Apply preprocessing
X = data.drop('purchased', axis=1)
y = data['purchased']
X_processed = preprocessor.fit_transform(X)

# Advanced preprocessing with feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

text_preprocessor = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('svd', TruncatedSVD(n_components=100))
])`,
        complexity: "Scaling: O(n), Imputation: O(n), Encoding: O(n*k) for k categories"
      }
    },
    {
      title: "🤖 Model Training",
      id: "training",
      description: "The core process of teaching algorithms to recognize patterns in data.",
      keyPoints: [
        "Algorithm selection based on problem type",
        "Hyperparameter tuning (grid search, random search, Bayesian)",
        "Training strategies (cross-validation, early stopping)",
        "Performance metrics (accuracy, precision, recall, RMSE)"
      ],
      detailedExplanation: [
        "Training process components:",
        "- Loss function: Quantifies model error",
        "- Optimization algorithm: Gradient descent variants",
        "- Regularization: Preventing overfitting (L1/L2, dropout)",
        "- Validation: Monitoring generalization performance",
        "",
        "Advanced techniques:",
        "- Transfer learning: Leveraging pretrained models",
        "- Ensemble methods: Combining multiple models",
        "- AutoML: Automated model selection and tuning",
        "- Distributed training: Handling large datasets",
        "",
        "Implementation patterns:",
        "- Experiment tracking (MLflow, Weights & Biases)",
        "- Model versioning and reproducibility",
        "- Hardware acceleration (GPU/TPU utilization)",
        "- Progressive loading for large datasets"
      ],
      code: {
        python: `# Model Training Examples
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import xgboost as xgb

# Basic training
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2)

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
print(classification_report(y_test, rf.predict(X_test)))

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# XGBoost with early stopping
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.1
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    early_stopping_rounds=10,
    evals=[(dtest, 'test')]
)

# Deep Learning with Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)`,
        complexity: "Varies by algorithm: Linear models O(n*m), Trees O(n*log(n)), Neural nets O(n^2)"
      }
    },
    {
      title: "📊 Model Evaluation",
      id: "evaluation",
      description: "Assessing model performance and generalization capabilities.",
      keyPoints: [
        "Train-test-validation split strategies",
        "Cross-validation techniques (k-fold, stratified, time-series)",
        "Performance metrics selection (problem-dependent)",
        "Error analysis and model interpretation"
      ],
      detailedExplanation: [
        "Evaluation methodologies:",
        "- Holdout validation: Simple train-test split",
        "- Cross-validation: Robust performance estimation",
        "- Bootstrapping: Confidence intervals for metrics",
        "- Time-series validation: Walk-forward testing",
        "",
        "Key metrics by problem type:",
        "- Classification: Accuracy, Precision, Recall, F1, ROC-AUC",
        "- Regression: MSE, RMSE, MAE, R-squared",
        "- Clustering: Silhouette score, Davies-Bouldin index",
        "- Ranking: NDCG, Mean Average Precision",
        "",
        "Advanced analysis:",
        "- Confusion matrix analysis",
        "- Feature importance/attribution",
        "- Error pattern detection",
        "- Bias and fairness metrics",
        "",
        "Visual evaluation:",
        "- ROC and precision-recall curves",
        "- Residual plots for regression",
        "- Calibration curves",
        "- SHAP and LIME explanations"
      ],
      code: {
        python: `# Model Evaluation Examples
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                           roc_auc_score, classification_report)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Basic metrics
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Cross-validation
cv_scores = cross_val_score(rf, X_processed, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Feature Importance
importances = rf.feature_importances_
features = numeric_features + list(preprocessor.named_transformers_['cat']
                                  .named_steps['onehot']
                                  .get_feature_names_out())

plt.barh(features, importances)
plt.title("Feature Importance")
plt.show()

# Advanced evaluation with SHAP
import shap

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=features)`,
        complexity: "Metrics: O(n), Cross-validation: O(k*n), SHAP: O(n*m)"
      }
    },
    {
      title: "🚀 Deployment & Monitoring",
      id: "deployment",
      description: "Moving models from development to production and ensuring continued performance.",
      keyPoints: [
        "Model serialization and packaging",
        "Deployment options (batch, real-time, edge)",
        "API development (Flask, FastAPI)",
        "Monitoring and model retraining strategies"
      ],
      detailedExplanation: [
        "Deployment architectures:",
        "- Batch processing: Scheduled model runs",
        "- Real-time APIs: REST/gRPC endpoints",
        "- Embedded models: On-device inference",
        "- Serverless: Event-driven execution",
        "",
        "Production considerations:",
        "- Model versioning and rollback",
        "- A/B testing of model versions",
        "- Canary deployments",
        "- Blue-green deployment strategies",
        "",
        "Monitoring aspects:",
        "- Performance metrics drift",
        "- Data/concept drift detection",
        "- Infrastructure monitoring",
        "- Business impact tracking",
        "",
        "Tools and platforms:",
        "- Containerization: Docker, Kubernetes",
        "- Serving: TensorFlow Serving, TorchServe",
        "- Monitoring: Prometheus, Grafana",
        "- Full platforms: Sagemaker, Vertex AI, MLflow"
      ],
      code: {
        python: `# Model Deployment Examples
import pickle
from flask import Flask, request, jsonify
import pandas as pd

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Flask API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    processed = preprocessor.transform(df)
    predictions = model.predict(processed)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# Monitoring example (simplified)
from datetime import datetime
import numpy as np

class ModelMonitor:
    def __init__(self, window_size=100):
        self.predictions = []
        self.timestamps = []
        self.window_size = window_size
    
    def log_prediction(self, pred, actual=None):
        self.predictions.append((pred, actual))
        self.timestamps.append(datetime.now())
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            self.timestamps.pop(0)
    
    def check_drift(self):
        if len(self.predictions) < self.window_size:
            return False
        
        # Simple drift detection (could use more sophisticated methods)
        preds, actuals = zip(*self.predictions)
        recent_acc = np.mean([p == a for p, a in zip(preds[-50:], actuals[-50:])])
        old_acc = np.mean([p == a for p, a in zip(preds[:50], actuals[:50])])
        
        return (old_acc - recent_acc) > 0.1  # 10% accuracy drop

# Dockerfile example (would be in separate file)
"""
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
"""`,
        complexity: "API: O(1) per request, Monitoring: O(n), Retraining: depends on algorithm"
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
        Machine Learning Pipeline
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
        }}>End-to-End ML Process</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          A complete machine learning project involves multiple stages from data collection to deployment.
          This section covers the entire pipeline with practical implementations and considerations
          for production-grade ML systems.
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

      {/* Pipeline Visualization */}
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
        }}>ML Pipeline Overview</h2>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '1rem'
        }}>
          {content.map((section, index) => (
            <div key={section.id} style={{
              flex: '1',
              minWidth: '150px',
              textAlign: 'center',
              padding: '1.5rem 1rem',
              backgroundColor: '#ecfdf5',
              borderRadius: '12px',
              border: '2px solid #a7f3d0',
              position: 'relative'
            }}>
              {index < content.length - 1 && (
                <div style={{
                  position: 'absolute',
                  right: '-20px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  color: '#059669',
                  fontSize: '1.5rem'
                }}>→</div>
              )}
              <div style={{
                fontSize: '2rem',
                marginBottom: '0.5rem'
              }}>{section.title.split(' ')[0]}</div>
              <div style={{ color: '#374151' }}>{section.description}</div>
            </div>
          ))}
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
        }}>ML Pipeline Best Practices</h3>
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
            }}>Pipeline Design Principles</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Modularity: Keep components independent and interchangeable
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Reproducibility: Version data, code, and models
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Automation: Minimize manual steps
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Monitoring: Track performance and data quality at each stage
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
            }}>Common Pitfalls</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Data Leakage:</strong> Preventing test data from influencing training<br/>
              <strong>Concept Drift:</strong> Models degrading over time as data changes<br/>
              <strong>Technical Debt:</strong> Quick prototypes becoming production systems<br/>
              <strong>Reproducibility:</strong> Failing to capture all dependencies
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
            }}>Tools & Platforms</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>End-to-End:</strong> Kubeflow, MLflow, SageMaker Pipelines<br/>
              <strong>Data Versioning:</strong> DVC, Delta Lake<br/>
              <strong>Feature Stores:</strong> Feast, Tecton<br/>
              <strong>Model Serving:</strong> Seldon Core, BentoML<br/>
              <strong>Monitoring:</strong> Evidently, WhyLogs
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MLPipeline;