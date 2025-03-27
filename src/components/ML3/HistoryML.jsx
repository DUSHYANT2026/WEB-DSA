import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function HistoryML() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "🕰️ Origins of Machine Learning",
      id: "origins",
      description: "The foundational ideas and early developments that shaped modern machine learning.",
      keyPoints: [
        "1940s-50s: Birth of neural networks and cybernetics",
        "1950: Alan Turing's 'Computing Machinery and Intelligence'",
        "1956: Dartmouth Workshop - birth of AI as a field",
        "1957: Frank Rosenblatt's Perceptron"
      ],
      detailedExplanation: [
        "Early Pioneers:",
        "- Warren McCulloch & Walter Pitts (1943): First mathematical model of neural networks",
        "- Alan Turing (1950): Proposed the Turing Test and learning machines",
        "- Arthur Samuel (1959): Coined 'machine learning' while working on checkers program",
        "",
        "Key Breakthroughs:",
        "- Perceptron (1957): First trainable neural network model",
        "- Adaline/Madaline (1960): Practical neural network for real-world problems",
        "- Nearest Neighbor (1967): Early instance-based learning algorithm",
        "",
        "Theoretical Foundations:",
        "- Norbert Wiener's Cybernetics (1948)",
        "- Claude Shannon's Information Theory",
        "- Frank Rosenblatt's Perceptron Convergence Theorem",
        "- Marvin Minsky's work on AI foundations"
      ],
      timeline: [
        ["1943", "McCulloch-Pitts neuron model"],
        ["1950", "Turing's seminal paper on machine intelligence"],
        ["1956", "Dartmouth Conference - AI founding event"],
        ["1957", "Rosenblatt's Perceptron"],
        ["1959", "Arthur Samuel defines machine learning"]
      ]
    },
    {
      title: "📉 AI Winters and Resurgences",
      id: "winters",
      description: "Periods of reduced funding and interest followed by renewed excitement in AI/ML.",
      keyPoints: [
        "1974-80: First AI winter (Perceptron limitations)",
        "1987-93: Second AI winter (expert systems plateau)",
        "1990s: Resurgence with statistical approaches",
        "2000s: Emergence of practical applications"
      ],
      detailedExplanation: [
        "First AI Winter (1974-1980):",
        "- Minsky & Papert's 'Perceptrons' (1969) highlighted limitations",
        "- Reduced funding for neural network research",
        "- Shift to symbolic AI and expert systems",
        "",
        "Interim Progress:",
        "- Backpropagation algorithm (1974, rediscovered 1986)",
        "- Hopfield Networks (1982)",
        "- Boltzmann Machines (1985)",
        "",
        "Second AI Winter (1987-1993):",
        "- Expert systems failed to scale",
        "- LISP machine market collapse",
        "- DARPA cuts AI funding",
        "",
        "Factors in Resurgence:",
        "- Increased computational power",
        "- Availability of large datasets",
        "- Improved algorithms and theoretical understanding"
      ],
      timeline: [
        ["1969", "Minsky & Papert expose Perceptron limitations"],
        ["1974", "First AI winter begins"],
        ["1986", "Backpropagation rediscovered"],
        ["1987", "Second AI winter begins"],
        ["1995", "Support Vector Machines introduced"]
      ]
    },
    {
      title: "🚀 Modern Machine Learning Era",
      id: "modern",
      description: "The explosion of machine learning in the 21st century and current state of the field.",
      keyPoints: [
        "2006: Deep learning breakthrough (Hinton et al.)",
        "2012: AlexNet dominates ImageNet competition",
        "2015: ResNet enables very deep networks",
        "2017: Transformer architecture revolutionizes NLP"
      ],
      detailedExplanation: [
        "Key Developments:",
        "- Deep Belief Networks (2006): Enabled training of deep architectures",
        "- AlexNet (2012): Demonstrated power of GPUs for deep learning",
        "- Word2Vec (2013): Effective word embeddings",
        "- GANs (2014): Generative Adversarial Networks",
        "",
        "Architectural Advances:",
        "- ResNet (2015): Solved vanishing gradient problem",
        "- Transformer (2017): Self-attention mechanisms",
        "- BERT (2018): Bidirectional language models",
        "- GPT models (2018-2023): Large language models",
        "",
        "Current Landscape:",
        "- Widespread industry adoption",
        "- Ethical concerns and responsible AI",
        "- Hardware specialization (TPUs, neuromorphic chips)",
        "- Multimodal models and AGI research"
      ],
      timeline: [
        ["2006", "Deep learning renaissance begins"],
        ["2012", "AlexNet wins ImageNet"],
        ["2015", "ResNet enables 100+ layer networks"],
        ["2017", "Transformer architecture introduced"],
        ["2020", "GPT-3 demonstrates few-shot learning"]
      ]
    },
    {
      title: "🔮 Future Directions",
      id: "future",
      description: "Emerging trends and potential future developments in machine learning.",
      keyPoints: [
        "Self-supervised and unsupervised learning",
        "Neuromorphic computing and brain-inspired architectures",
        "Explainable AI (XAI) and interpretability",
        "AI safety and alignment research"
      ],
      detailedExplanation: [
        "Technical Frontiers:",
        "- Few-shot and zero-shot learning",
        "- Neural-symbolic integration",
        "- Continual/lifelong learning",
        "- Energy-efficient AI",
        "",
        "Societal Impact Areas:",
        "- AI for scientific discovery",
        "- Personalized medicine",
        "- Climate change modeling",
        "- Education and accessibility",
        "",
        "Challenges to Address:",
        "- Bias and fairness in ML systems",
        "- Privacy-preserving learning",
        "- Robustness to adversarial attacks",
        "- Sustainable AI development",
        "",
        "Potential Paradigm Shifts:",
        "- Quantum machine learning",
        "- Biological learning systems",
        "- Artificial general intelligence",
        "- Human-AI collaboration frameworks"
      ],
      timeline: [
        ["2022", "Large language models become mainstream"],
        ["2025", "Projected growth in edge AI"],
        ["2030", "Potential AGI milestones"],
        ["2040", "Speculative brain-computer interfaces"]
      ]
    }
  ];

  return (
    <div style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '2rem',
      background: 'linear-gradient(to bottom right, #f5f3ff, #ede9fe)',
      borderRadius: '20px',
      boxShadow: '0 10px 30px rgba(0,0,0,0.1)'
    }}>
      <h1 style={{
        fontSize: '3.5rem',
        fontWeight: '800',
        textAlign: 'center',
        background: 'linear-gradient(to right, #7c3aed, #8b5cf6)',
        WebkitBackgroundClip: 'text',
        backgroundClip: 'text',
        color: 'transparent',
        marginBottom: '3rem'
      }}>
        History of Machine Learning
      </h1>

      <div style={{
        backgroundColor: 'rgba(124, 58, 237, 0.1)',
        padding: '2rem',
        borderRadius: '12px',
        marginBottom: '3rem',
        borderLeft: '4px solid #7c3aed'
      }}>
        <h2 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#7c3aed',
          marginBottom: '1rem'
        }}>Introduction to Machine Learning → History</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          Machine learning has evolved through decades of research, setbacks, and breakthroughs. 
          Understanding this history provides context for current techniques and insights into 
          future developments in artificial intelligence.
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
            border: '1px solid #ddd6fe',
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
              color: '#7c3aed'
            }}>{section.title}</h2>
            <button
              onClick={() => toggleSection(section.id)}
              style={{
                background: 'linear-gradient(to right, #7c3aed, #8b5cf6)',
                color: 'white',
                padding: '0.75rem 1.5rem',
                borderRadius: '8px',
                border: 'none',
                cursor: 'pointer',
                fontWeight: '600',
                transition: 'all 0.2s ease',
                ':hover': {
                  transform: 'scale(1.05)',
                  boxShadow: '0 5px 15px rgba(124, 58, 237, 0.4)'
                }
              }}
            >
              {visibleSection === section.id ? "Collapse Section" : "Expand Section"}
            </button>
          </div>

          {visibleSection === section.id && (
            <div style={{ display: 'grid', gap: '2rem' }}>
              <div style={{
                backgroundColor: '#f5f3ff',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#7c3aed',
                  marginBottom: '1rem'
                }}>Key Developments</h3>
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
                backgroundColor: '#ede9fe',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#7c3aed',
                  marginBottom: '1rem'
                }}>Historical Context</h3>
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
                backgroundColor: '#e9d5ff',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#7c3aed',
                  marginBottom: '1rem'
                }}>Timeline</h3>
                <div style={{
                  display: 'grid',
                  gap: '0.5rem',
                  gridTemplateColumns: 'auto 1fr',
                  alignItems: 'center'
                }}>
                  {section.timeline.map(([year, event], index) => (
                    <React.Fragment key={index}>
                      <div style={{
                        backgroundColor: '#7c3aed',
                        color: 'white',
                        padding: '0.5rem 1rem',
                        borderRadius: '20px',
                        fontWeight: '600',
                        justifySelf: 'start'
                      }}>
                        {year}
                      </div>
                      <div style={{ color: '#374151' }}>{event}</div>
                    </React.Fragment>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      ))}

      {/* Key Figures */}
      <div style={{
        marginTop: '3rem',
        padding: '2rem',
        backgroundColor: 'white',
        borderRadius: '16px',
        boxShadow: '0 5px 15px rgba(0,0,0,0.05)',
        border: '1px solid #ddd6fe'
      }}>
        <h2 style={{
          fontSize: '2rem',
          fontWeight: '700',
          color: '#7c3aed',
          marginBottom: '2rem'
        }}>Pioneers of Machine Learning</h2>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
          gap: '1.5rem'
        }}>
          {[
            {
              name: "Alan Turing",
              contribution: "Theoretical foundations of computation and learning",
              period: "1930s-1950s"
            },
            {
              name: "Frank Rosenblatt",
              contribution: "Invented the Perceptron (early neural network)",
              period: "1950s-1960s"
            },
            {
              name: "Geoffrey Hinton",
              contribution: "Backpropagation, Deep Learning revival",
              period: "1980s-present"
            },
            {
              name: "Yann LeCun",
              contribution: "Convolutional Neural Networks",
              period: "1980s-present"
            },
            {
              name: "Yoshua Bengio",
              contribution: "Probabilistic models, sequence learning",
              period: "1990s-present"
            },
            {
              name: "Andrew Ng",
              contribution: "Popularizing ML education, practical applications",
              period: "2000s-present"
            }
          ].map((person, index) => (
            <div key={index} style={{
              backgroundColor: '#f5f3ff',
              padding: '1.5rem',
              borderRadius: '12px',
              borderLeft: '4px solid #7c3aed'
            }}>
              <h3 style={{
                fontSize: '1.3rem',
                fontWeight: '700',
                color: '#7c3aed',
                marginBottom: '0.5rem'
              }}>{person.name}</h3>
              <p style={{ color: '#374151', marginBottom: '0.5rem' }}>{person.contribution}</p>
              <div style={{
                display: 'inline-block',
                backgroundColor: '#ede9fe',
                color: '#7c3aed',
                padding: '0.25rem 0.75rem',
                borderRadius: '20px',
                fontSize: '0.9rem',
                fontWeight: '600'
              }}>
                {person.period}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Key Takeaways */}
      <div style={{
        marginTop: '3rem',
        padding: '2rem',
        backgroundColor: '#f5f3ff',
        borderRadius: '16px',
        boxShadow: '0 5px 15px rgba(0,0,0,0.05)',
        border: '1px solid #ddd6fe'
      }}>
        <h3 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#7c3aed',
          marginBottom: '1.5rem'
        }}>Lessons from ML History</h3>
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
              color: '#7c3aed',
              marginBottom: '0.75rem'
            }}>Patterns of Progress</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Alternating cycles of hype and disillusionment (AI winters)
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Theoretical breakthroughs often precede practical applications by decades
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Hardware advances frequently enable algorithmic breakthroughs
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Interdisciplinary cross-pollination drives innovation
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
              color: '#7c3aed',
              marginBottom: '0.75rem'
            }}>Historical Context for Current ML</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              Many "new" concepts in machine learning have deep historical roots:
              <br/><br/>
              - Modern neural networks build on 1940s neurobiological models<br/>
              - Attention mechanisms relate to 1990s memory networks<br/>
              - GANs extend earlier work on adversarial training<br/>
              - Transfer learning concepts date to 1970s psychological theories
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
              color: '#7c3aed',
              marginBottom: '0.75rem'
            }}>Future Outlook</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              Based on historical patterns, we can anticipate:
              <br/><br/>
              - Continued cycles of hype and consolidation<br/>
              - Gradual progress toward more general AI capabilities<br/>
              - Increasing focus on ethical and societal impacts<br/>
              - Convergence of symbolic and connectionist approaches
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default HistoryML;