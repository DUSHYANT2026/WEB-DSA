import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function NeuralNetworks() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "🧠 Perceptrons and MLPs",
      id: "perceptrons",
      description: "The building blocks of neural networks, from single neurons to multi-layer architectures.",
      keyPoints: [
        "Perceptron: Single neuron with learnable weights",
        "Multi-Layer Perceptron (MLP): Stacked layers of neurons",
        "Activation functions (Sigmoid, ReLU, Tanh)",
        "Universal approximation theorem"
      ],
      detailedExplanation: [
        "Key concepts:",
        "- Input layer, hidden layers, output layer architecture",
        "- Feedforward computation: Wx + b → activation",
        "- Decision boundaries and linear separability",
        "- Capacity vs. overfitting tradeoff",
        "",
        "Implementation considerations:",
        "- Weight initialization strategies",
        "- Bias terms and their role",
        "- Choosing appropriate activation functions",
        "- Hidden layer sizing and depth",
        "",
        "Mathematical formulation:",
        "- Forward pass: a⁽ˡ⁾ = f(W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾)",
        "- Activation functions:",
        "  • Sigmoid: 1/(1 + e⁻ˣ)",
        "  • ReLU: max(0, x)",
        "  • Tanh: (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)"
      ],
      code: {
        python: `# Implementing MLP from scratch
import numpy as np

class MLP:
    def __init__(self, layer_sizes):
        self.weights = [np.random.randn(y, x) * 0.1 
                       for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((y, 1)) for y in layer_sizes[1:]]
    
    def forward(self, x):
        a = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.relu(z)  # Using ReLU activation
        return a
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

# Example usage
mlp = MLP([784, 128, 64, 10])  # For MNIST classification
output = mlp.forward(input_image)

# Using PyTorch
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)`,
        complexity: "Forward pass: O(∑(l=1 to L) nˡnˡ⁻¹) where nˡ is layer size"
      }
    },
    {
      title: "🔄 Backpropagation",
      id: "backprop",
      description: "The fundamental algorithm for training neural networks through gradient computation.",
      keyPoints: [
        "Chain rule applied to computational graphs",
        "Gradient descent optimization",
        "Loss functions (Cross-Entropy, MSE)",
        "Vanishing/exploding gradients"
      ],
      detailedExplanation: [
        "Backpropagation steps:",
        "1. Forward pass: Compute loss",
        "2. Backward pass: Compute gradients",
        "3. Parameter update: Adjust weights",
        "",
        "Mathematical derivation:",
        "- Output layer gradients: ∂L/∂z⁽ᴸ⁾",
        "- Hidden layer gradients: ∂L/∂z⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ ∂L/∂z⁽ˡ⁺¹⁾ ⊙ f'(z⁽ˡ⁾)",
        "- Parameter gradients: ∂L/∂W⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ (a⁽ˡ⁻¹⁾)ᵀ",
        "",
        "Practical considerations:",
        "- Numerical stability issues",
        "- Gradient checking for verification",
        "- Mini-batch processing",
        "- Learning rate selection",
        "",
        "Advanced variants:",
        "- Nesterov momentum",
        "- Adagrad/RMSprop/Adam",
        "- Second-order methods"
      ],
      code: {
        python: `# Backpropagation Implementation
def backward(self, x, y_true):
    # Forward pass
    activations = [x]
    zs = []
    a = x
    for w, b in zip(self.weights, self.biases):
        z = np.dot(w, a) + b
        zs.append(z)
        a = self.relu(z)
        activations.append(a)
    
    # Backward pass
    dL_dz = (activations[-1] - y_true)  # MSE derivative
    gradients = []
    
    for l in range(len(self.weights)-1, -1, -1):
        # Gradient for weights
        dL_dW = np.dot(dL_dz, activations[l].T)
        # Gradient for biases
        dL_db = dL_dz
        # Gradient for previous layer
        if l > 0:
            dL_dz = np.dot(self.weights[l].T, dL_dz) * self.relu_derivative(zs[l-1])
        
        gradients.append((dL_dW, dL_db))
    
    return gradients[::-1]  # Reverse to match layer order

def relu_derivative(self, z):
    return (z > 0).astype(float)

# PyTorch does this automatically
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()  # Backpropagation
    optimizer.step()`,
        complexity: "Backward pass: ~2-3x forward pass complexity"
      }
    },
    {
      title: "🖼️ Convolutional Neural Networks",
      id: "cnns",
      description: "Specialized architectures for processing grid-like data (images, time series).",
      keyPoints: [
        "Convolutional layers: Local receptive fields",
        "Pooling layers: Dimensionality reduction",
        "Architectural patterns (LeNet, AlexNet, ResNet)",
        "Transfer learning with pretrained models"
      ],
      detailedExplanation: [
        "CNN building blocks:",
        "- Convolution: Filter application with shared weights",
        "- Padding and stride controls",
        "- Pooling (Max, Average) for translation invariance",
        "- 1x1 convolutions for channel mixing",
        "",
        "Modern architectures:",
        "- LeNet-5 (1998): Early success on MNIST",
        "- AlexNet (2012): Deep learning breakthrough",
        "- VGG (2014): Uniform architecture",
        "- ResNet (2015): Residual connections",
        "- EfficientNet (2019): Scalable architecture",
        "",
        "Implementation details:",
        "- Kernel size selection (3x3, 5x5, etc.)",
        "- Channel depth progression",
        "- Batch normalization layers",
        "- Dropout for regularization"
      ],
      code: {
        python: `# CNN Implementation Examples
import torch
import torch.nn as nn

# Basic CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)

# Using pretrained model
from torchvision import models
resnet = models.resnet18(pretrained=True)
# Replace final layer
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # For 100-class problem

# Modern architecture with skip connections
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return nn.functional.relu(out))`,
        complexity: "O(n²k²c_in c_out) per conv layer (n=spatial size, k=kernel size)"
      }
    },
    {
      title: "⏳ Recurrent Neural Networks",
      id: "rnns",
      description: "Networks with internal state for processing sequential data.",
      keyPoints: [
        "Recurrent connections for temporal processing",
        "Long Short-Term Memory (LSTM) units",
        "Gated Recurrent Units (GRUs)",
        "Sequence-to-sequence models"
      ],
      detailedExplanation: [
        "RNN fundamentals:",
        "- Hidden state carries temporal information",
        "- Unfolding through time for backpropagation",
        "- Challenges with long-term dependencies",
        "",
        "Advanced architectures:",
        "- LSTM: Input, forget, output gates",
        "- GRU: Simplified gating mechanism",
        "- Bidirectional RNNs: Context from both directions",
        "- Attention mechanisms for sequence alignment",
        "",
        "Applications:",
        "- Time series forecasting",
        "- Natural language processing",
        "- Speech recognition",
        "- Video analysis",
        "",
        "Implementation considerations:",
        "- Truncated backpropagation through time",
        "- Gradient clipping for stability",
        "- Teacher forcing for training",
        "- Beam search for decoding"
      ],
      code: {
        python: `# RNN Implementations
import torch
import torch.nn as nn

# Basic RNN
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)  # out: (batch, seq_len, hidden_size)
        return self.fc(out[:, -1, :])  # Last timestep

# LSTM Network
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return self.fc(out[:, -1, :])

# Sequence-to-sequence with attention
class Seq2SeqAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.decoder = nn.LSTM(output_size, hidden_size)
        self.attention = nn.Linear(2*hidden_size + hidden_size, 1)
        self.fc = nn.Linear(2*hidden_size + hidden_size, output_size)
    
    def forward(self, src, trg):
        # Encoder
        enc_output, (h, c) = self.encoder(src)
        h = h.view(2, -1).unsqueeze(0)  # Combine bidirectional
        c = c.view(2, -1).unsqueeze(0)
        
        # Decoder with attention
        outputs = []
        for t in range(trg.size(1)):
            # Attention weights
            energy = torch.tanh(self.attention(torch.cat((h.repeat(trg.size(0), 1, 1), 
                                              enc_output), dim=2))
            attention = torch.softmax(energy, dim=1)
            
            # Context vector
            context = (attention * enc_output).sum(dim=1)
            
            # Decoder step
            out, (h, c) = self.decoder(trg[:, t:t+1], (h, c))
            out = torch.cat((out.squeeze(1), context), dim=1)
            out = self.fc(out)
            outputs.append(out)
        
        return torch.stack(outputs, dim=1)`,
        complexity: "LSTM: O(nh²) per timestep (n=sequence length, h=hidden size)"
      }
    },
    {
      title: "🔄 Transformers",
      id: "transformers",
      description: "Attention-based architectures that have revolutionized NLP and beyond.",
      keyPoints: [
        "Self-attention mechanism",
        "Multi-head attention",
        "Positional encoding",
        "Encoder-decoder architecture"
      ],
      detailedExplanation: [
        "Transformer components:",
        "- Query-Key-Value attention computation",
        "- Scaled dot-product attention",
        "- Layer normalization and residual connections",
        "- Feed-forward sublayers",
        "",
        "Key architectures:",
        "- Original Transformer (Vaswani et al.)",
        "- BERT: Bidirectional pretraining",
        "- GPT: Autoregressive language modeling",
        "- Vision Transformers (ViT)",
        "",
        "Implementation details:",
        "- Masking for sequence processing",
        "- Positional encoding schemes",
        "- Multi-head attention splitting",
        "- Learning rate scheduling",
        "",
        "Applications:",
        "- Machine translation",
        "- Text generation",
        "- Image recognition",
        "- Multimodal learning"
      ],
      code: {
        python: `# Transformer Implementation
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        return self.norm2(x + self.dropout(ff_output))

# Using Hugging Face transformers
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)`,
        complexity: "O(n²d + nd²) where n=sequence length, d=embedding size"
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
        Neural Networks Fundamentals
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
        }}>Neural Networks</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Neural networks are the foundation of modern deep learning, capable of learning complex patterns
          through hierarchical feature extraction. This section covers architectures from basic perceptrons
          to cutting-edge transformer models.
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
                backgroundColor: '#ecfdf5',
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
                backgroundColor: '#f0f9ff',
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
        }}>Neural Network Architectures Comparison</h2>
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
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Type</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Best For</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Key Features</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Popular Libraries</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["MLP", "Tabular data, simple patterns", "Fully connected layers", "PyTorch, Keras"],
                ["CNN", "Images, grid-like data", "Convolutional filters, pooling", "TensorFlow, FastAI"],
                ["RNN/LSTM", "Sequences, time series", "Recurrent connections, memory", "PyTorch Lightning"],
                ["Transformer", "Text, long-range dependencies", "Self-attention, positional encoding", "Hugging Face"]
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
        backgroundColor: '#ecfdf5',
        borderRadius: '16px',
        boxShadow: '0 5px 15px rgba(0,0,0,0.05)',
        border: '1px solid #a7f3d0'
      }}>
        <h3 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#0369a1',
          marginBottom: '1.5rem'
        }}>Neural Network Best Practices</h3>
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
            }}>Architecture Selection</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                MLPs for simple structured data
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                CNNs for images and spatial data
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                RNNs/LSTMs for time series and sequences
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Transformers for text and long-range dependencies
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
            }}>Training Tips</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Initialization:</strong> Use He/Kaiming for ReLU networks<br/>
              <strong>Normalization:</strong> BatchNorm/LayerNorm for deep networks<br/>
              <strong>Regularization:</strong> Dropout, weight decay, early stopping<br/>
              <strong>Optimization:</strong> Adam is usually a safe choice
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
            }}>Emerging Trends</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Self-supervised learning:</strong> Pretraining on unlabeled data<br/>
              <strong>Neural Architecture Search:</strong> Automating model design<br/>
              <strong>Graph Neural Networks:</strong> For relational data<br/>
              <strong>Diffusion Models:</strong> State-of-the-art generation
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default NeuralNetworks;