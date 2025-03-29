import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function FeatureEngineering() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "🔍 Feature Selection",
      id: "selection",
      description: "Identifying the most informative features to improve model performance and reduce complexity.",
      keyPoints: [
        "Filter methods: Statistical measures (correlation, mutual info)",
        "Wrapper methods: Model-based selection (forward/backward)",
        "Embedded methods: Built into model training (Lasso, RF importance)",
        "Dimensionality reduction: PCA, LDA, t-SNE"
      ],
      detailedExplanation: [
        "Filter Methods:",
        "- Pearson correlation for linear relationships",
        "- Mutual information for non-linear dependencies",
        "- Chi-square for categorical features",
        "- ANOVA F-value for feature variance",
        "",
        "Wrapper Methods:",
        "- Recursive Feature Elimination (RFE)",
        "- Sequential feature selection",
        "- Genetic algorithms for feature subsets",
        "- Exhaustive search (best subset selection)",
        "",
        "Embedded Methods:",
        "- L1 regularization (Lasso) for sparsity",
        "- Tree-based feature importance",
        "- Neural network attention weights",
        "- Gradient boosting feature contributions",
        "",
        "Practical Considerations:",
        "- Computational cost vs. benefit",
        "- Stability of selected features",
        "- Domain knowledge integration",
        "- Feature selection pipelines"
      ],
      code: {
        python: `# Feature Selection Techniques
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# Filter method - Select top 10 features by ANOVA F-value
selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)

# Wrapper method - Recursive Feature Elimination
model = RandomForestClassifier()
rfe = RFE(model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)

# Embedded method - L1-based feature selection
from sklearn.linear_model import LassoCV
lasso = LassoCV().fit(X, y)
sfm = SelectFromModel(lasso, prefit=True)
X_lasso = sfm.transform(X)

# Dimensionality reduction - PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X)

# Feature importance from Random Forest
rf = RandomForestClassifier().fit(X, y)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importance
plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.show()`,
        complexity: "Filter: O(n), Wrapper: O(n²), PCA: O(min(n²p, p²n))"
      }
    },
    {
      title: "🛠️ Feature Extraction",
      id: "extraction",
      description: "Transforming raw data into more meaningful representations for machine learning.",
      keyPoints: [
        "Text features: Bag-of-words, TF-IDF, word embeddings",
        "Image features: SIFT, HOG, CNN activations",
        "Time series: Fourier transforms, wavelets",
        "Automated feature learning: Autoencoders"
      ],
      detailedExplanation: [
        "Text Feature Extraction:",
        "- Bag-of-words and n-grams",
        "- TF-IDF for document importance",
        "- Word2Vec, GloVe for semantic embeddings",
        "- BERT contextual embeddings",
        "",
        "Image Feature Extraction:",
        "- Traditional: SIFT, SURF, HOG",
        "- CNN-based: Pretrained model features",
        "- Autoencoder latent representations",
        "- Attention-based feature maps",
        "",
        "Time Series Features:",
        "- Statistical features (mean, variance)",
        "- Spectral analysis (FFT, wavelets)",
        "- Shape-based features (DTW)",
        "- Recurrent network embeddings",
        "",
        "Advanced Techniques:",
        "- Feature learning with autoencoders",
        "- Graph neural networks for relational data",
        "- Multimodal feature fusion",
        "- Self-supervised representation learning"
      ],
      code: {
        python: `# Feature Extraction Examples
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tensorflow.keras.applications import VGG16

# Text feature extraction
corpus = ["This is document one", "Another document here"]
vectorizer = TfidfVectorizer(max_features=1000)
X_text = vectorizer.fit_transform(corpus)

# Dimensionality reduction for text
svd = TruncatedSVD(n_components=100)
X_text_reduced = svd.fit_transform(X_text)

# Image feature extraction with CNN
model = VGG16(weights='imagenet', include_top=False, pooling='avg')
image_features = model.predict(images)

# Time series feature extraction
def extract_ts_features(series):
    features = {
        'mean': np.mean(series),
        'std': np.std(series),
        'max': np.max(series),
        'min': np.min(series),
        'zero_crossings': np.sum(np.diff(np.sign(series)) != 0)
    }
    return features

# Autoencoder for feature learning
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=50, batch_size=256)

# Extract learned features
encoded_features = encoder.predict(X)`,
        complexity: "Text: O(n*m), Image CNN: O(n*h*w*c), Autoencoder: O(n*e*d)"
      }
    },
    {
      title: "🧩 Feature Construction",
      id: "construction",
      description: "Creating new features from existing ones to improve model performance.",
      keyPoints: [
        "Numerical transformations: Log, square, binning",
        "Interaction features: Products, ratios",
        "Date/time features: Day of week, holidays",
        "Domain-specific features: Business metrics"
      ],
      detailedExplanation: [
        "Numerical Transformations:",
        "- Log transforms for skewed distributions",
        "- Polynomial features for non-linearities",
        "- Binning for non-linear relationships",
        "- Scaling and normalization",
        "",
        "Categorical Feature Engineering:",
        "- One-hot and target encoding",
        "- Frequency encoding",
        "- Embedding layers for high-cardinality",
        "- Feature hashing",
        "",
        "Temporal Features:",
        "- Cyclical encoding (sine/cosine)",
        "- Time since events",
        "- Rolling statistics",
        "- Seasonality indicators",
        "",
        "Domain-Specific Features:",
        "- Business KPIs and ratios",
        "- Physical/engineering formulas",
        "- Aggregated statistics by groups",
        "- Distance metrics for spatial data"
      ],
      code: {
        python: `# Feature Construction Examples
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Numerical transformations
df['log_income'] = np.log1p(df['income'])
df['age_squared'] = df['age'] ** 2

# Binning numerical features
df['age_bin'] = pd.cut(df['age'], bins=[0,18,35,50,100], labels=['child','adult','middle','senior'])

# Interaction features
df['income_per_age'] = df['income'] / df['age']
df['height_weight_ratio'] = df['height'] * df['weight']

# Date/time features
df['purchase_date'] = pd.to_datetime(df['purchase_date'])
df['purchase_dayofweek'] = df['purchase_date'].dt.dayofweek
df['is_weekend'] = df['purchase_dayofweek'].isin([5,6]).astype(int)

# Polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(df[['age', 'income']])

# Target encoding
target_mean = df.groupby('category')['target'].mean()
df['category_encoded'] = df['category'].map(target_mean)

# Aggregated features
df['avg_income_by_zip'] = df.groupby('zip_code')['income'].transform('mean')
df['max_income_by_education'] = df.groupby('education')['income'].transform('max')`,
        complexity: "Basic transforms: O(n), Aggregations: O(n log n)"
      }
    },
    {
      title: "⚖️ Feature Scaling",
      id: "scaling",
      description: "Normalizing feature ranges to improve model convergence and performance.",
      keyPoints: [
        "Standardization: Mean=0, variance=1",
        "Normalization: Min-max scaling",
        "Robust scaling: Median/IQR based",
        "Target-aware scaling: Quantile transforms"
      ],
      detailedExplanation: [
        "Standardization (Z-score):",
        "- Suitable for most linear models",
        "- Assumes roughly Gaussian distribution",
        "- Sensitive to outliers",
        "- Formula: (x - mean) / std",
        "",
        "Normalization (Min-Max):",
        "- Bounds features to [0,1] range",
        "- Useful for neural networks",
        "- Preserves zero entries in sparse data",
        "- Formula: (x - min) / (max - min)",
        "",
        "Robust Scaling:",
        "- Uses median and IQR",
        "- Resistant to outliers",
        "- Good for skewed distributions",
        "- Formula: (x - median) / IQR",
        "",
        "Special Cases:",
        "- Quantile transforms for non-linear",
        "- Power transforms (Box-Cox, Yeo-Johnson)",
        "- Unit norm scaling for text/SVMs",
        "- Custom scaling for domain needs"
      ],
      code: {
        python: `# Feature Scaling Techniques
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer

# Standardization
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Min-Max scaling
minmax = MinMaxScaler(feature_range=(0, 1))
X_normalized = minmax.fit_transform(X)

# Robust scaling
robust = RobustScaler()
X_robust = robust.fit_transform(X)

# Quantile transformation (Gaussian output)
quantile = QuantileTransformer(output_distribution='normal')
X_quantile = quantile.fit_transform(X)

# Power transformation (Yeo-Johnson)
power = PowerTransformer(method='yeo-johnson')
X_power = power.fit_transform(X)

# Custom scaling example
class LogScaler:
    def fit(self, X):
        self.min = np.log1p(X.min())
        self.max = np.log1p(X.max())
        
    def transform(self, X):
        X_log = np.log1p(X)
        return (X_log - self.min) / (self.max - self.min)
        
log_scaler = LogScaler()
log_scaler.fit(X)
X_log_scaled = log_scaler.transform(X)`,
        complexity: "All scalers: O(n) for fit, O(1) for transform"
      }
    },
    {
      title: "🔗 Feature Pipelines",
      id: "pipelines",
      description: "Building reproducible workflows for feature engineering and transformation.",
      keyPoints: [
        "Scikit-learn pipelines for workflow chaining",
        "Custom transformers for domain-specific processing",
        "Feature unions for parallel processing",
        "Persisting and reusing feature pipelines"
      ],
      detailedExplanation: [
        "Pipeline Construction:",
        "- Sequential application of transforms",
        "- Avoiding data leakage with proper ordering",
        "- Combining feature selection and scaling",
        "- Model stacking within pipelines",
        "",
        "Custom Transformers:",
        "- Implementing fit/transform methods",
        "- Hyperparameter tuning integration",
        "- Stateless vs. stateful transforms",
        "- Column-specific transformations",
        "",
        "Advanced Patterns:",
        "- Feature unions for heterogeneous data",
        "- Conditional processing paths",
        "- Memory caching for expensive steps",
        "- Parallel feature processing",
        "",
        "Production Considerations:",
        "- Persisting trained pipelines",
        "- Versioning feature engineering code",
        "- Monitoring feature distributions",
        "- Handling missing data in production"
      ],
      code: {
        python: `# Feature Engineering Pipelines
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib

# Custom transformer example
class TextLengthTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return np.array([len(text) for text in X]).reshape(-1, 1)

# Numeric preprocessing pipeline
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2))
])

# Text preprocessing pipeline
text_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=1000)),
    ('dim_reduce', TruncatedSVD(n_components=100))
])

# Combined feature union
preprocessor = FeatureUnion([
    ('numeric', numeric_pipeline, ['age', 'income']),
    ('text', text_pipeline, 'description'),
    ('length', TextLengthTransformer(), 'description')
])

# Full pipeline with model
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selector', SelectKBest(k=50)),
    ('classifier', RandomForestClassifier())
])

# Train and persist pipeline
full_pipeline.fit(X_train, y_train)
joblib.dump(full_pipeline, 'feature_pipeline.pkl')

# Custom column selector
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X[self.columns]

# Complex pipeline with conditionals
from sklearn.pipeline import make_union

numeric_features = ['age', 'income']
categorical_features = ['gender', 'education']
text_features = ['description']

preprocessor = make_union(
    Pipeline([
        ('selector', ColumnSelector(numeric_features)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ]),
    Pipeline([
        ('selector', ColumnSelector(categorical_features)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]),
    Pipeline([
        ('selector', ColumnSelector(text_features)),
        ('vectorizer', TfidfVectorizer())
    ])
)`,
        complexity: "Pipeline: O(sum of component complexities)"
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
        Feature Engineering for Machine Learning
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
        }}>Data Preprocessing → Feature Engineering</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Feature engineering is the process of transforming raw data into features that better represent 
          the underlying problem to predictive models, resulting in improved model accuracy on unseen data.
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

      {/* Best Practices */}
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
        }}>Feature Engineering Best Practices</h2>
        <div style={{ display: 'grid', gap: '1.5rem' }}>
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
            }}>Domain Knowledge Integration</h3>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.5rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Incorporate business-specific metrics and ratios
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Create features that reflect known causal relationships
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Engineer time-based features for temporal patterns
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Use hierarchical aggregations where appropriate
              </li>
            </ul>
          </div>

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
            }}>Technical Considerations</h3>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.5rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Always split data before feature engineering to avoid leakage
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Document all feature transformations for reproducibility
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Monitor feature distributions over time in production
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Balance feature richness with computational constraints
              </li>
            </ul>
          </div>

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
            }}>Evaluation Strategies</h3>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.5rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Use feature importance metrics to guide engineering efforts
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Evaluate feature sets with ablation studies
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Track feature quality metrics alongside model performance
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Validate feature stability across different data samples
              </li>
            </ul>
          </div>
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
        }}>Feature Engineering Insights</h3>
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
            }}>Impact on Model Performance</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              Well-engineered features can often improve model performance more than algorithm selection:
              <br/><br/>
              - Simple models with great features outperform complex models with poor features<br/>
              - Feature engineering reduces the need for large amounts of training data<br/>
              - Good features make patterns more easily learnable by models<br/>
              - Feature quality directly impacts model interpretability
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
            }}>Future Trends</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              Emerging directions in feature engineering:
              <br/><br/>
              - Automated feature engineering with ML (AutoML, featuretools)<br/>
              - Self-supervised feature learning from unlabeled data<br/>
              - Neural feature synthesis with generative models<br/>
              - Dynamic feature engineering that adapts to data drift
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default FeatureEngineering;