import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function TimeSeriesForecasting() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "⏳ ARIMA Models",
      id: "arima",
      description: "Autoregressive Integrated Moving Average models for stationary time series.",
      keyPoints: [
        "AR (Autoregressive): Model future values based on past values",
        "I (Integrated): Differencing to make series stationary",
        "MA (Moving Average): Model future values based on past errors",
        "Seasonal ARIMA (SARIMA) for periodic patterns"
      ],
      detailedExplanation: [
        "Components of ARIMA(p,d,q):",
        "- p: Number of autoregressive terms",
        "- d: Degree of differencing needed for stationarity",
        "- q: Number of moving average terms",
        "",
        "Model Selection Process:",
        "1. Check stationarity (ADF test)",
        "2. Determine differencing order (d)",
        "3. Identify AR/MA terms (ACF/PACF plots)",
        "4. Estimate parameters (MLE)",
        "5. Validate residuals (Ljung-Box test)",
        "",
        "Applications:",
        "- Economic forecasting",
        "- Inventory management",
        "- Energy demand prediction",
        "- Stock price analysis"
      ],
      code: {
        python: `# ARIMA Implementation
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Load time series data
data = pd.read_csv('timeseries.csv', parse_dates=['date'], index_col='date')

# Check stationarity and difference if needed
def check_stationarity(series):
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(series)
    return result[1] < 0.05  # p-value < 0.05 indicates stationarity

if not check_stationarity(data['value']):
    data['value'] = data['value'].diff().dropna()

# Plot ACF/PACF to identify p and q
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))
plot_acf(data['value'], ax=ax1)
plot_pacf(data['value'], ax=ax2)
plt.show()

# Fit ARIMA model
model = ARIMA(data['value'], order=(2,1,1))  # (p,d,q)
results = model.fit()

# Summary of model
print(results.summary())

# Forecast next 10 periods
forecast = results.get_forecast(steps=10)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

# Plot results
data['value'].plot(figsize=(12,6), label='Observed')
forecast_mean.plot(label='Forecast')
plt.fill_between(conf_int.index,
                conf_int.iloc[:, 0],
                conf_int.iloc[:, 1], color='k', alpha=0.1)
plt.legend()
plt.title('ARIMA Forecast')
plt.show()`,
        complexity: "Fitting: O(n²), Forecasting: O(1) per step"
      }
    },
    {
      title: "📈 Exponential Smoothing",
      id: "smoothing",
      description: "Weighted average methods that give more importance to recent observations.",
      keyPoints: [
        "Simple Exponential Smoothing (no trend/seasonality)",
        "Holt's method (captures trend)",
        "Holt-Winters (captures trend and seasonality)",
        "ETS models (Error, Trend, Seasonal components)"
      ],
      detailedExplanation: [
        "Types of Exponential Smoothing:",
        "- Single (SES): Level only",
        "- Double: Level + Trend",
        "- Triple: Level + Trend + Seasonality",
        "",
        "Smoothing Parameters:",
        "- α (level): Closer to 1 weights recent obs more",
        "- β (trend): Controls trend component",
        "- γ (seasonal): Controls seasonal adjustment",
        "",
        "Model Selection:",
        "- AIC/BIC for parameter selection",
        "- Box-Cox transformation for variance stabilization",
        "- Automated model selection with ets()",
        "",
        "Applications:",
        "- Short-term demand forecasting",
        "- Inventory control systems",
        "- Financial market analysis",
        "- Web traffic prediction"
      ],
      code: {
        python: `# Exponential Smoothing Implementation
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import pandas as pd

# Load data
data = pd.read_csv('sales.csv', parse_dates=['month'], index_col='month')

# Fit Holt-Winters seasonal model
model = ExponentialSmoothing(data['sales'],
                            trend='add',
                            seasonal='mul',
                            seasonal_periods=12)
results = model.fit()

# Print parameters
print(f"Smoothing parameters: alpha={results.params['smoothing_level']:.3f}, "
      f"beta={results.params['smoothing_trend']:.3f}, "
      f"gamma={results.params['smoothing_seasonal']:.3f}")

# Forecast next year
forecast = results.forecast(12)

# Plot results
fig, ax = plt.subplots(figsize=(12,6))
data['sales'].plot(ax=ax, label='Observed')
forecast.plot(ax=ax, label='Forecast', color='red')
ax.fill_between(forecast.index,
               results.predict(start=data.index[-24])[-12:],
               forecast,
               color='red', alpha=0.1)
plt.title('Holt-Winters Seasonal Forecast')
plt.legend()
plt.show()

# Automated model selection
from statsmodels.tsa.api import ETSModel
best_aic = np.inf
best_model = None

# Test different combinations
for trend in ['add', 'mul', None]:
    for seasonal in ['add', 'mul', None]:
        try:
            model = ETSModel(data['sales'], trend=trend, seasonal=seasonal, seasonal_periods=12)
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_model = results
        except:
            continue

print(f"Best model: AIC={best_aic:.1f}")
print(best_model.summary())`,
        complexity: "Fitting: O(n), Forecasting: O(1) per step"
      }
    },
    {
      title: "🧠 LSTM for Time Series",
      id: "lstm",
      description: "Long Short-Term Memory networks for complex temporal patterns.",
      keyPoints: [
        "Special RNN architecture for long-term dependencies",
        "Memory cells with input, forget, and output gates",
        "Handles non-linear and multivariate relationships",
        "Requires careful hyperparameter tuning"
      ],
      detailedExplanation: [
        "LSTM Architecture Components:",
        "- Forget gate: Decides what information to discard",
        "- Input gate: Updates cell state with new information",
        "- Output gate: Determines next hidden state",
        "- Cell state: Carries information across time steps",
        "",
        "Implementation Considerations:",
        "- Sequence length selection",
        "- Normalization/scaling of inputs",
        "- Bidirectional LSTMs for richer context",
        "- Attention mechanisms for long sequences",
        "",
        "Training Process:",
        "1. Prepare sequential training samples",
        "2. Define network architecture",
        "3. Train with backpropagation through time",
        "4. Validate on holdout period",
        "5. Tune hyperparameters (epochs, units, etc.)",
        "",
        "Applications:",
        "- Multivariate financial forecasting",
        "- Energy load prediction",
        "- Weather forecasting",
        "- Anomaly detection in temporal data"
      ],
      code: {
        python: `# LSTM for Time Series Forecasting
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv('energy.csv', parse_dates=['timestamp'], index_col='timestamp')
values = data['consumption'].values.reshape(-1,1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length-1):
        X.append(data[i:(i+seq_length), 0])
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y)

seq_length = 24  # 24 hours lookback
X, y = create_sequences(scaled, seq_length)

# Split train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape for LSTM [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(X_train, y_train, 
                   epochs=50, 
                   batch_size=32,
                   validation_data=(X_test, y_test),
                   verbose=1)

# Plot training history
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.legend()
plt.show()

# Make predictions
test_predict = model.predict(X_test)
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test.reshape(-1,1))

# Plot predictions
plt.figure(figsize=(12,6))
plt.plot(y_test, label='Actual')
plt.plot(test_predict, label='Predicted')
plt.title('LSTM Time Series Forecasting')
plt.legend()
plt.show()`,
        complexity: "Training: O(n × L × H²) where L=sequence length, H=hidden units"
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
        Time Series Forecasting
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
        }}>Advanced Machine Learning Algorithms → Time Series Forecasting</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Time series forecasting involves predicting future values based on previously observed values.
          This section covers traditional statistical methods and modern deep learning approaches
          for analyzing and forecasting temporal data.
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
                backgroundColor: '#f0f9ff',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#0ea5e9',
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
                backgroundColor: '#f0f9ff',
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
          color: '#0ea5e9',
          marginBottom: '2rem'
        }}>Time Series Methods Comparison</h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            textAlign: 'left'
          }}>
            <thead style={{
              backgroundColor: '#0ea5e9',
              color: 'white'
            }}>
              <tr>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Method</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Strengths</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Weaknesses</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Best For</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["ARIMA", "Interpretable, handles trends/seasonality", "Requires stationary data, manual tuning", "Univariate, medium-term forecasts"],
                ["Exponential Smoothing", "Simple, handles seasonality well", "Limited to additive patterns", "Short-term, seasonal data"],
                ["LSTM", "Learns complex patterns, multivariate", "Computationally expensive, black-box", "Multivariate, long sequences"]
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
          color: '#0ea5e9',
          marginBottom: '1.5rem'
        }}>Forecasting Best Practices</h3>
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
            }}>Model Selection Guidelines</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Start simple (exponential smoothing) before trying complex models
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Use ARIMA for interpretable, stationary series
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Consider LSTMs for complex, multivariate patterns
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Ensemble methods can combine strengths of different approaches
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
            }}>Evaluation Metrics</h4>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
              gap: '1rem'
            }}>
              {[
                ["MAE", "Mean Absolute Error", "Robust to outliers"],
                ["RMSE", "Root Mean Squared Error", "Punishes large errors"],
                ["MAPE", "Mean Absolute % Error", "Relative error measure"],
                ["MASE", "Mean Abs Scaled Error", "Compares to naive forecast"]
              ].map(([metric, name, desc], index) => (
                <div key={index} style={{
                  backgroundColor: '#f0f9ff',
                  padding: '1rem',
                  borderRadius: '8px'
                }}>
                  <div style={{
                    fontWeight: '700',
                    color: '#0ea5e9',
                    marginBottom: '0.5rem'
                  }}>{metric}</div>
                  <div style={{ fontWeight: '600' }}>{name}</div>
                  <div style={{ color: '#64748b', fontSize: '0.9rem' }}>{desc}</div>
                </div>
              ))}
            </div>
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
              <strong>Prophet:</strong> Facebook's additive regression model<br/>
              <strong>N-BEATS:</strong> Neural basis expansion analysis<br/>
              <strong>DeepAR:</strong> Probabilistic forecasting with RNNs<br/>
              <strong>Temporal Fusion Transformers:</strong> Attention-based models
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default TimeSeriesForecasting;