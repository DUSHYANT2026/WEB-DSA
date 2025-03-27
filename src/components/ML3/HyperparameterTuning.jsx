import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function HyperparameterTuning() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "🔍 Grid Search",
      id: "grid-search",
      description: "Exhaustive search over specified parameter values to find the optimal combination.",
      keyPoints: [
        "Tests all possible combinations in parameter grid",
        "Guarantees finding best combination within grid",
        "Computationally expensive for large spaces",
        "Parallelizable across parameter combinations"
      ],
      detailedExplanation: [
        "Implementation Process:",
        "1. Define parameter grid with discrete values",
        "2. Create all possible combinations",
        "3. Evaluate model for each combination",
        "4. Select combination with best performance",
        "",
        "Key Considerations:",
        "- Parameter ranges should be carefully chosen",
        "- Can use with cross-validation (GridSearchCV)",
        "- Performance metrics must be clearly defined",
        "- Early stopping can save computation",
        "",
        "When to Use:",
        "- Small parameter spaces (few parameters with limited values)",
        "- When exhaustive search is computationally feasible",
        "- When parameter interactions are important",
        "- For final model tuning after narrowing ranges"
      ],
      code: {
        python: `# Grid Search with scikit-learn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create model
rf = RandomForestClassifier(random_state=42)

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # Use all available cores
    verbose=1
)

# Fit to data
grid_search.fit(X, y)

# Results
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Access all results
results_df = pd.DataFrame(grid_search.cv_results_)
print(results_df[['params', 'mean_test_score', 'std_test_score']])`,
        complexity: "O(n^k) where n=parameter values, k=parameters"
      }
    },
    {
      title: "🎲 Random Search",
      id: "random-search",
      description: "Samples parameter combinations randomly from specified distributions.",
      keyPoints: [
        "More efficient than grid search for high-dimensional spaces",
        "Can use continuous distributions for parameters",
        "Better at finding good combinations with fewer trials",
        "Doesn't guarantee optimal solution"
      ],
      detailedExplanation: [
        "Implementation Process:",
        "1. Define parameter distributions (discrete or continuous)",
        "2. Set number of iterations (budget)",
        "3. Randomly sample combinations",
        "4. Evaluate and select best performer",
        "",
        "Key Advantages:",
        "- More efficient coverage of large parameter spaces",
        "- Can focus sampling on promising regions",
        "- Works well with early stopping",
        "- Easier to parallelize than grid search",
        "",
        "When to Use:",
        "- Large parameter spaces (many parameters)",
        "- When some parameters matter more than others",
        "- Initial exploration of parameter space",
        "- When computational budget is limited",
        "",
        "Best Practices:",
        "- Use appropriate distributions (log-uniform for learning rates)",
        "- Allocate sufficient iterations (10-100x number of parameters)",
        "- Consider adaptive random search variants"
      ],
      code: {
        python: `# Random Search with scikit-learn
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform, randint
from sklearn.svm import SVC

# Define parameter distributions
param_dist = {
    'C': loguniform(1e-3, 1e3),  # Log-uniform between 0.001 and 1000
    'gamma': loguniform(1e-4, 1e1),
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': randint(1, 5)  # Uniform integer between 1 and 4
}

# Create model
svm = SVC()

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=svm,
    param_distributions=param_dist,
    n_iter=50,  # Number of parameter combinations to try
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# Fit to data
random_search.fit(X, y)

# Results
print("Best parameters:", random_search.best_params_)
print("Best score:", random_search.best_score_)

# Plotting parameter vs score
import matplotlib.pyplot as plt
plt.scatter(
    [params['C'] for params in random_search.cv_results_['params']],
    random_search.cv_results_['mean_test_score']
)
plt.xscale('log')
plt.xlabel('C parameter (log scale)')
plt.ylabel('Mean CV score')
plt.title('Random Search Results')
plt.show()`,
        complexity: "O(n) where n=number of iterations"
      }
    },
    {
      title: "🧠 Bayesian Optimization",
      id: "bayesian-opt",
      description: "Builds probabilistic model of objective function to guide search for optimal parameters.",
      keyPoints: [
        "Uses surrogate model (often Gaussian Process) to approximate objective",
        "Balances exploration and exploitation",
        "More efficient than random/grid search",
        "Works well with expensive-to-evaluate functions"
      ],
      detailedExplanation: [
        "Key Components:",
        "- Surrogate model: Approximates true objective function",
        "- Acquisition function: Determines next point to evaluate",
        "- History of evaluations: Guides model updates",
        "",
        "Implementation Process:",
        "1. Define parameter space and ranges",
        "2. Initialize with random points",
        "3. Build surrogate model from evaluations",
        "4. Select next point using acquisition function",
        "5. Evaluate and update model",
        "6. Repeat until convergence or budget exhausted",
        "",
        "Advantages:",
        "- Requires fewer evaluations than random search",
        "- Handles noisy objectives well",
        "- Can incorporate prior knowledge",
        "- Works with continuous and discrete parameters",
        "",
        "Popular Libraries:",
        "- scikit-optimize",
        "- Hyperopt",
        "- BayesianOptimization",
        "- Optuna"
      ],
      code: {
        python: `# Bayesian Optimization with scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from xgboost import XGBClassifier

# Define search spaces
search_spaces = {
    'learning_rate': Real(0.01, 1.0, 'log-uniform'),
    'max_depth': Integer(3, 10),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'gamma': Real(0, 5),
    'n_estimators': Integer(50, 200),
    'reg_alpha': Real(1e-4, 10, 'log-uniform'),
    'reg_lambda': Real(1e-4, 10, 'log-uniform')
}

# Create model
xgb = XGBClassifier(n_jobs=-1, random_state=42)

# Setup Bayesian Optimization
bayes_search = BayesSearchCV(
    estimator=xgb,
    search_spaces=search_spaces,
    n_iter=32,  # Number of evaluations
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# Fit to data
bayes_search.fit(X, y)

# Results
print("Best parameters:", bayes_search.best_params_)
print("Best score:", bayes_search.best_score_)

# Plotting optimization progress
from skopt.plots import plot_convergence
plot_convergence(bayes_search.optimizer_results_[0])
plt.show()

# Alternative with Optuna
import optuna
from optuna.samplers import TPESampler

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200)
    }
    model = XGBClassifier(**params)
    return cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()

study = optuna.create_study(direction='maximize', sampler=TPESampler())
study.optimize(objective, n_trials=50)

print("Best trial:", study.best_trial.params)`,
        complexity: "O(n²) to O(n³) per iteration (depends on surrogate model)"
      }
    },
    {
      title: "⚙️ Advanced Tuning Methods",
      id: "advanced",
      description: "Sophisticated techniques for hyperparameter optimization beyond basic approaches.",
      keyPoints: [
        "Genetic algorithms: Evolutionary optimization",
        "Hyperband: Bandit-based resource allocation",
        "BOHB: Combines Bayesian optimization with Hyperband",
        "Meta-learning: Transfer tuning knowledge"
      ],
      detailedExplanation: [
        "Genetic Algorithms:",
        "- Maintain population of parameter sets",
        "- Evolve through selection, crossover, mutation",
        "- Good for combinatorial/discrete spaces",
        "",
        "Hyperband:",
        "- Adaptive resource allocation",
        "- Early stopping of poorly performing configurations",
        "- More efficient than random search",
        "",
        "BOHB (Bayesian Optimization + Hyperband):",
        "- Uses Bayesian optimization to guide Hyperband",
        "- Combines strengths of both approaches",
        "- State-of-the-art for many problems",
        "",
        "Meta-Learning Approaches:",
        "- Learn from previous tuning experiments",
        "- Warm-start optimization",
        "- Transfer learning across datasets",
        "",
        "Other Methods:",
        "- Particle Swarm Optimization",
        "- Gradient-based optimization (for differentiable hyperparameters)",
        "- Multi-fidelity optimization",
        "- Neural Architecture Search (for deep learning)"
      ],
      code: {
        python: `# Advanced Tuning Methods Example
# Using Optuna with Hyperband pruning
import optuna
from optuna.pruners import HyperbandPruner

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500)
    }
    
    model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
    
    # Early stopping callback
    pruning_callback = optuna.integration.XGBoostPruningCallback(
        trial, 'validation_0-logloss'
    )
    
    return cross_val_score(
        model, 
        X, 
        y, 
        cv=5, 
        scoring='accuracy',
        fit_params={'callbacks': [pruning_callback]}
    ).mean()

# Create study with Hyperband pruner
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(),
    pruner=HyperbandPruner(
        min_resource=1,
        max_resource=100,
        reduction_factor=3
    )
)

study.optimize(objective, n_trials=100)

# Using Genetic Algorithms with TPOT
from tpot import TPOTClassifier

tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    cv=5,
    random_state=42,
    verbosity=2,
    n_jobs=-1
)

tpot.fit(X, y)
print(tpot.fitted_pipeline_)

# Save best pipeline
tpot.export('best_pipeline.py')`,
        complexity: "Varies by method (typically between O(n) and O(n²))"
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
        Hyperparameter Tuning
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
        }}>Model Evaluation and Optimization → Hyperparameter Tuning</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Hyperparameter tuning is crucial for maximizing model performance. This section covers various
          optimization strategies from basic grid search to advanced Bayesian methods, with practical
          implementations for machine learning workflows.
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
                }}>Implementation</h3>
                <p style={{
                  color: '#374151',
                  fontWeight: '600',
                  marginBottom: '1rem',
                  fontSize: '1.1rem'
                }}>Computational Complexity: {section.code.complexity}</p>
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
        }}>Hyperparameter Tuning Methods Comparison</h2>
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
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Best For</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Pros</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Cons</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Grid Search", "Small parameter spaces", "Guaranteed to find best in grid", "Exponential complexity"],
                ["Random Search", "Medium/large spaces", "More efficient than grid", "No guarantee of optimality"],
                ["Bayesian Optimization", "Expensive evaluations", "Sample efficient", "Overhead of surrogate model"],
                ["Genetic Algorithms", "Combinatorial spaces", "Good for discrete params", "Many hyperparameters itself"],
                ["Hyperband/BOHB", "Resource allocation", "Automated early stopping", "Complex to implement"]
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
        }}>Tuning Best Practices</h3>
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
            }}>Workflow Recommendations</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Start with broad random search to identify promising regions
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Refine with Bayesian optimization in promising areas
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Use early stopping to save computation time
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Consider parameter importance for focused tuning
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
            }}>Parameter Space Design</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Learning rates:</strong> Log-uniform distribution (e.g., 1e-5 to 1e-1)<br/>
              <strong>Layer sizes:</strong> Geometric progression (e.g., 32, 64, 128, 256)<br/>
              <strong>Regularization:</strong> Mixture of linear and log scales<br/>
              <strong>Discrete choices:</strong> Limit to most promising options
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
            }}>Advanced Considerations</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Multi-fidelity:</strong> Low-fidelity approximations first<br/>
              <strong>Warm-starting:</strong> Initialize with known good configurations<br/>
              <strong>Parallelization:</strong> Distributed tuning across machines<br/>
              <strong>Meta-learning:</strong> Transfer tuning knowledge between datasets
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default HyperparameterTuning;