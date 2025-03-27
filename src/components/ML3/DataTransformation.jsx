import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function DataTransformation() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "🧹 Data Cleaning",
      id: "cleaning",
      description: "Preparing raw data for analysis by handling inconsistencies and missing values.",
      keyPoints: [
        "Handling missing values (imputation, deletion)",
        "Outlier detection (IQR, Z-score methods)",
        "Dealing with duplicate data entries",
        "Inconsistent formatting correction"
      ],
      detailedExplanation: [
        "Missing Data Strategies:",
        "- Deletion: Remove rows/columns with missing values (listwise deletion)",
        "- Imputation: Fill missing values (mean, median, mode, predictive)",
        "- Flagging: Add indicator variables for missingness",
        "",
        "Outlier Treatment:",
        "- Statistical methods: Z-score (|Z| > 3), IQR (1.5*IQR rule)",
        "- Visualization methods: Box plots, scatter plots",
        "- Domain-specific thresholds",
        "- Winsorization (capping extreme values)",
        "",
        "Duplicate Handling:",
        "- Exact duplicates: Remove all copies keeping one",
        "- Fuzzy duplicates: Use similarity measures (Levenshtein distance)",
        "- Contextual duplicates: Business rules for uniqueness",
        "",
        "Format Standardization:",
        "- Date/time formats",
        "- Categorical encoding",
        "- Units of measurement",
        "- String normalization (lowercase, trimming)"
      ],
      code: {
        python: `# Data Cleaning with Pandas
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Sample dataframe with missing values and outliers
data = pd.DataFrame({
    'age': [25, 30, np.nan, 35, 150, 40, np.nan],
    'income': [50000, 60000, 70000, np.nan, 80000, 90000, 1000000],
    'department': ['Sales', 'Sales', 'IT', 'IT', 'IT', 'HR', 'HR']
})

# Handle missing values
# Option 1: Drop rows with missing values
clean_data = data.dropna()

# Option 2: Impute missing values
imputer = SimpleImputer(strategy='median')
data['age'] = imputer.fit_transform(data[['age']])

# Handle outliers
def cap_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    df[column] = np.where(df[column] < lower_bound, lower_bound,
                         np.where(df[column] > upper_bound, upper_bound, df[column]))
    return df

data = cap_outliers(data, 'income')

# Handle duplicates
data = data.drop_duplicates()

# Standardize formats
data['department'] = data['department'].str.upper().str.strip()`,
        complexity: "Imputation: O(n), Outlier detection: O(n log n), Deduplication: O(n)"
      }
    },
    {
      title: "🔄 Data Transformation",
      id: "transformation",
      description: "Converting data into forms more suitable for modeling through scaling and encoding.",
      keyPoints: [
        "Scaling: Min-Max, Standard (Z-score), Robust",
        "Encoding: One-Hot, Label, Target Encoding",
        "Log and power transformations",
        "Binning continuous variables"
      ],
      detailedExplanation: [
        "Feature Scaling Methods:",
        "- StandardScaler: (x - μ) / σ (mean=0, std=1)",
        "- MinMaxScaler: (x - min) / (max - min) (range [0,1])",
        "- RobustScaler: Uses median and IQR (resistant to outliers)",
        "- MaxAbsScaler: Scales by maximum absolute value (range [-1,1])",
        "",
        "Encoding Techniques:",
        "- One-Hot: Binary columns for each category (sparse)",
        "- Label: Numeric codes for categories (ordinal)",
        "- Target: Mean target value per category (supervised)",
        "- Frequency: Category frequency as feature",
        "",
        "Non-linear Transformations:",
        "- Log transform: Reduces right skew (log1p for zeros)",
        "- Box-Cox: Power transform (requires positive values)",
        "- Yeo-Johnson: Generalized power transform",
        "- Quantile transform: Uniform distribution",
        "",
        "Binning Strategies:",
        "- Equal width: Fixed range bins",
        "- Equal frequency: Same count per bin",
        "- Decision tree bins: Optimal splits",
        "- Domain-specific bins (age groups)"
      ],
      code: {
        python: `# Data Transformation Examples
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, 
                                 PowerTransformer, KBinsDiscretizer)
import pandas as pd
import numpy as np

# Sample data
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50],
    'income': [50000, 60000, 70000, 80000, 90000, 100000],
    'department': ['Sales', 'IT', 'IT', 'HR', 'Sales', 'HR'],
    'purchases': [0, 2, 15, 3, 7, 30]
})

# Scaling
scaler = StandardScaler()
data[['age_scaled', 'income_scaled']] = scaler.fit_transform(data[['age', 'income']])

# Encoding
encoder = OneHotEncoder(sparse=False)
dept_encoded = encoder.fit_transform(data[['department']])
data = pd.concat([data, pd.DataFrame(dept_encoded, 
                    columns=encoder.get_feature_names(['department']))], axis=1)

# Log transform
data['log_purchases'] = np.log1p(data['purchases'])

# Binning
binner = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
data['age_binned'] = binner.fit_transform(data[['age']])

# Power transform
pt = PowerTransformer(method='yeo-johnson')
data['income_transformed'] = pt.fit_transform(data[['income']])`,
        complexity: "Scaling: O(n), Encoding: O(n), Binning: O(n log n)"
      }
    },
    {
      title: "🛠️ Feature Engineering",
      id: "engineering",
      description: "Creating new features from raw data to improve model performance.",
      keyPoints: [
        "Feature selection (Filter, Wrapper, Embedded methods)",
        "Feature extraction (PCA, LDA, t-SNE)",
        "Interaction features and polynomial terms",
        "Domain-specific feature creation"
      ],
      detailedExplanation: [
        "Feature Selection Approaches:",
        "- Filter methods: Select based on statistical measures (correlation)",
        "- Wrapper methods: Use model performance (recursive feature elimination)",
        "- Embedded methods: Built into model training (L1 regularization)",
        "- Hybrid approaches",
        "",
        "Dimensionality Reduction:",
        "- PCA: Linear projection to orthogonal components",
        "- LDA: Supervised dimensionality reduction",
        "- t-SNE: Non-linear visualization of high-dim data",
        "- UMAP: Preserves both local and global structure",
        "",
        "Feature Creation:",
        "- Mathematical transformations (ratios, differences)",
        "- Time-based features (lags, rolling stats)",
        "- Text features (TF-IDF, word counts, embeddings)",
        "- Image features (HOG, SIFT, CNN embeddings)",
        "",
        "Temporal Features:",
        "- Time since last event",
        "- Rolling averages/windows",
        "- Seasonality indicators",
        "- Time-based aggregations"
      ],
      code: {
        python: `# Feature Engineering Techniques
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np

# Sample data
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50],
    'income': [50000, 60000, 70000, 80000, 90000, 100000],
    'tenure': [1, 3, 5, 7, 9, 11],
    'target': [0, 1, 0, 1, 0, 1]
})

# Feature Selection
selector = SelectKBest(score_func=f_classif, k=2)
selected_features = selector.fit_transform(data[['age', 'income', 'tenure']], data['target'])

# Dimensionality Reduction
pca = PCA(n_components=2)
pca_features = pca.fit_transform(data[['age', 'income', 'tenure']])

# Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=True)
interaction_features = poly.fit_transform(data[['age', 'income']])

# Domain-Specific Features
data['income_to_age_ratio'] = data['income'] / data['age']
data['tenure_squared'] = data['tenure'] ** 2

# Time-Based Features (example with datetime)
dates = pd.date_range('2020-01-01', periods=6, freq='M')
data['month'] = dates.month
data['is_q1'] = data['month'].isin([1, 2, 3]).astype(int)`,
        complexity: "PCA: O(n³), Feature selection: O(n²), Polynomial features: O(n^k)"
      }
    },
    {
      title: "✂️ Data Splitting",
      id: "splitting",
      description: "Dividing data into training, validation, and test sets for model development.",
      keyPoints: [
        "Train-test split (simple holdout)",
        "Cross-validation (k-fold, stratified)",
        "Time-based splitting",
        "Group-based splitting"
      ],
      detailedExplanation: [
        "Basic Splitting Strategies:",
        "- Simple holdout: Fixed split (e.g., 70-30)",
        "- Random sampling: Shuffle then split",
        "- Stratified sampling: Preserve class distribution",
        "- Time-based: Older data for train, newer for test",
        "",
        "Cross-Validation Techniques:",
        "- k-fold: Divide into k equal parts, rotate test set",
        "- Stratified k-fold: Preserve class ratios in folds",
        "- Leave-one-out: Extreme k-fold (k=n)",
        "- Repeated: Multiple random k-fold iterations",
        "",
        "Special Cases:",
        "- Group k-fold: Keep groups together (same subject)",
        "- Time series CV: Expanding window approach",
        "- Nested CV: Inner loop for hyperparameter tuning",
        "- Bootstrapping: Sampling with replacement",
        "",
        "Best Practices:",
        "- Maintain distribution consistency across splits",
        "- Avoid leakage between train and test",
        "- Consider temporal dependencies",
        "- Account for hierarchical data structures"
      ],
      code: {
        python: `# Data Splitting Strategies
from sklearn.model_selection import (train_test_split, KFold, 
                                   StratifiedKFold, TimeSeriesSplit)
import pandas as pd
import numpy as np

# Sample data
data = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'target': np.random.randint(0, 2, 100),
    'group': np.repeat([1,2,3,4,5], 20),
    'date': pd.date_range('2020-01-01', periods=100)
})

# Simple train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data[['feature1', 'feature2']], 
    data['target'],
    test_size=0.2,
    random_state=42,
    stratify=data['target']
)

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in kf.split(data):
    X_train, X_test = data.iloc[train_index], data.iloc[test_index]

# Stratified K-Fold (for imbalanced data)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in skf.split(data[['feature1', 'feature2']], data['target']):
    X_train, X_test = data.iloc[train_index], data.iloc[test_index]

# Time Series Split
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(data):
    X_train, X_test = data.iloc[train_index], data.iloc[test_index]

# Group K-Fold (keep groups together)
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for train_index, test_index in gkf.split(data, groups=data['group']):
    X_train, X_test = data.iloc[train_index], data.iloc[test_index]`,
        complexity: "Train-test split: O(n), k-fold CV: O(kn)"
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
        Data Transformation for ML
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
        }}>Data Preprocessing → Data Transformation</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Transforming raw data into a format suitable for machine learning models is crucial for model performance.
          This section covers the essential techniques for cleaning, transforming, and preparing data.
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
                backgroundColor: '#bae6fd',
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
                backgroundColor: '#e0f2fe',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#0ea5e9',
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

      {/* Best Practices */}
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
        }}>Data Transformation Best Practices</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '1.5rem' }}>
          {[
            {
              title: "Reproducibility",
              content: "Always save transformation parameters (fitted scalers, encoders) to apply consistently to new data"
            },
            {
              title: "Pipeline Integration",
              content: "Use sklearn Pipelines to bundle transformations with model steps for cleaner code"
            },
            {
              title: "Data Leakage Prevention",
              content: "Fit transformers on training data only, then transform test data to avoid information leakage"
            },
            {
              title: "Monitoring",
              content: "Track distributions before/after transformations to ensure they behave as expected"
            },
            {
              title: "Iterative Process",
              content: "Feature engineering is often iterative - analyze model errors to create better features"
            },
            {
              title: "Documentation",
              content: "Document all transformations applied for future reference and model interpretability"
            }
          ].map((item, index) => (
            <div key={index} style={{
              backgroundColor: '#e0f2fe',
              padding: '1.5rem',
              borderRadius: '12px',
              borderLeft: '4px solid #0ea5e9'
            }}>
              <h3 style={{
                fontSize: '1.3rem',
                fontWeight: '600',
                color: '#0ea5e9',
                marginBottom: '0.75rem'
              }}>{item.title}</h3>
              <p style={{ color: '#374151' }}>{item.content}</p>
            </div>
          ))}
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
          color: '#0ea5e9',
          marginBottom: '1.5rem'
        }}>ML Practitioner Insights</h3>
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
            }}>Common Pitfalls</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Applying transformations to entire dataset before train-test split
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Not considering the impact of scaling on interpretability
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Over-engineering features that don't generalize
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Ignoring the computational cost of complex transformations
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
            }}>Advanced Techniques</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Automated Feature Engineering:</strong> Libraries like FeatureTools<br/>
              <strong>Differential Privacy:</strong> For privacy-preserving transformations<br/>
              <strong>Neural Feature Extractors:</strong> Using pretrained models as feature generators<br/>
              <strong>Feature Stores:</strong> Managing and sharing features across projects
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
            }}>Domain-Specific Considerations</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>NLP:</strong> Text cleaning, tokenization, embedding strategies<br/>
              <strong>Time Series:</strong> Lag features, rolling statistics, seasonality<br/>
              <strong>Computer Vision:</strong> Normalization, augmentation, pretrained features<br/>
              <strong>Tabular Data:</strong> Feature interactions, aggregations, binning
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DataTransformation;