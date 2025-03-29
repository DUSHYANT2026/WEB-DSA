import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function RESTAPI() {
  const [visibleSection, setVisibleSection] = useState(null);

  const toggleSection = (section) => {
    setVisibleSection(visibleSection === section ? null : section);
  };

  const content = [
    {
      title: "📡 REST Fundamentals",
      id: "fundamentals",
      description: "Core principles and architectural constraints of RESTful APIs.",
      keyPoints: [
        "Client-server architecture",
        "Stateless communication",
        "Resource-based endpoints",
        "Standard HTTP methods (GET, POST, PUT, DELETE)"
      ],
      detailedExplanation: [
        "Key REST principles:",
        "- Uniform interface: Consistent resource identification and manipulation",
        "- Cacheability: Responses define cacheability",
        "- Layered system: Intermediary servers improve scalability",
        "- Code-on-demand (optional): Servers can extend client functionality",
        "",
        "HTTP Methods in REST:",
        "- GET: Retrieve resource representation",
        "- POST: Create new resource",
        "- PUT: Update existing resource",
        "- DELETE: Remove resource",
        "- PATCH: Partial resource updates",
        "",
        "Status Codes:",
        "- 2xx: Success (200 OK, 201 Created)",
        "- 3xx: Redirection (301 Moved Permanently)",
        "- 4xx: Client errors (400 Bad Request, 404 Not Found)",
        "- 5xx: Server errors (500 Internal Server Error)"
      ],
      code: {
        python: `# Example REST API endpoint with Flask
from flask import Flask, request, jsonify

app = Flask(__name__)

# Sample in-memory database
books = [
    {"id": 1, "title": "Clean Code", "author": "Robert Martin"},
    {"id": 2, "title": "Design Patterns", "author": "GoF"}
]

# GET all books
@app.route('/api/books', methods=['GET'])
def get_books():
    return jsonify(books)

# GET single book
@app.route('/api/books/<int:book_id>', methods=['GET'])
def get_book(book_id):
    book = next((b for b in books if b['id'] == book_id), None)
    if book is None:
        return jsonify({"error": "Book not found"}), 404
    return jsonify(book)

# POST new book
@app.route('/api/books', methods=['POST'])
def add_book():
    if not request.json or 'title' not in request.json:
        return jsonify({"error": "Bad request"}), 400
    
    new_book = {
        'id': books[-1]['id'] + 1,
        'title': request.json['title'],
        'author': request.json.get('author', '')
    }
    books.append(new_book)
    return jsonify(new_book), 201

if __name__ == '__main__':
    app.run(debug=True)`,
        complexity: "Basic CRUD operations: O(1) to O(n) depending on implementation"
      }
    },
    {
      title: "🔐 Authentication & Security",
      id: "auth",
      description: "Securing REST APIs and managing user authentication.",
      keyPoints: [
        "Token-based authentication (JWT)",
        "OAuth 2.0 flows",
        "API keys and rate limiting",
        "HTTPS and security best practices"
      ],
      detailedExplanation: [
        "Authentication Methods:",
        "- Basic Auth: Simple username/password (not recommended for production)",
        "- API Keys: Simple but less secure",
        "- JWT (JSON Web Tokens): Stateless tokens with expiration",
        "- OAuth 2.0: Delegated authorization framework",
        "",
        "Security Considerations:",
        "- Always use HTTPS (TLS encryption)",
        "- Implement proper CORS policies",
        "- Input validation and sanitization",
        "- Rate limiting to prevent abuse",
        "- Regular security audits",
        "",
        "Best Practices:",
        "- Store sensitive data securely (never in code)",
        "- Use environment variables for configuration",
        "- Implement proper error handling",
        "- Regular dependency updates",
        "- Security headers (CSP, XSS protection)"
      ],
      code: {
        python: `# JWT Authentication with Flask
from flask import Flask, request, jsonify
import jwt
import datetime
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Mock user database
users = {
    'admin': {'password': 'securepassword', 'role': 'admin'}
}

# Login endpoint
@app.route('/api/login', methods=['POST'])
def login():
    auth = request.authorization
    
    if not auth or not auth.username or not auth.password:
        return jsonify({"error": "Basic auth required"}), 401
    
    user = users.get(auth.username)
    if not user or user['password'] != auth.password:
        return jsonify({"error": "Invalid credentials"}), 401
    
    token = jwt.encode({
        'user': auth.username,
        'role': user['role'],
        'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
    }, app.config['SECRET_KEY'])
    
    return jsonify({'token': token})

# Protected route decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({"error": "Token is missing"}), 401
        
        try:
            data = jwt.decode(token.split()[1], app.config['SECRET_KEY'], algorithms=["HS256"])
        except:
            return jsonify({"error": "Token is invalid"}), 401
        
        return f(*args, **kwargs)
    return decorated

# Protected endpoint
@app.route('/api/protected', methods=['GET'])
@token_required
def protected():
    return jsonify({"message": "This is a protected endpoint"})

if __name__ == '__main__':
    app.run(ssl_context='adhoc')  # Enable HTTPS`,
        complexity: "JWT operations: O(1), Auth checks: O(1)"
      }
    },
    {
      title: "📦 API Design Best Practices",
      id: "design",
      description: "Principles for designing clean, maintainable, and scalable REST APIs.",
      keyPoints: [
        "Resource naming conventions",
        "Versioning strategies",
        "Pagination and filtering",
        "HATEOAS and discoverability"
      ],
      detailedExplanation: [
        "Naming Conventions:",
        "- Use nouns for resources (not verbs)",
        "- Plural resource names (/users not /user)",
        "- Lowercase with hyphens for multi-word resources",
        "- Consistent naming across endpoints",
        "",
        "API Versioning:",
        "- URL path versioning (/v1/users)",
        "- Header versioning (Accept: application/vnd.api.v1+json)",
        "- Query parameter versioning (/users?version=1)",
        "- Deprecation policies and sunset headers",
        "",
        "Advanced Features:",
        "- Pagination (limit/offset or cursor-based)",
        "- Filtering, sorting, and field selection",
        "- Hypermedia controls (HATEOAS)",
        "- Bulk operations",
        "- Async operations for long-running tasks"
      ],
      code: {
        python: `# Well-designed API with Flask
from flask import Flask, request, jsonify, url_for

app = Flask(__name__)

# Sample database
users = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"}
]

# GET users with pagination and filtering
@app.route('/api/v1/users', methods=['GET'])
def get_users():
    # Pagination
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    
    # Filtering
    name_filter = request.args.get('name')
    filtered_users = [u for u in users if not name_filter or name_filter.lower() in u['name'].lower()]
    
    # Pagination logic
    start = (page - 1) * per_page
    end = start + per_page
    paginated_users = filtered_users[start:end]
    
    # Build response with HATEOAS links
    response = {
        'data': paginated_users,
        'links': {
            'self': url_for('get_users', page=page, per_page=per_page, _external=True),
            'next': url_for('get_users', page=page+1, per_page=per_page, _external=True) if end < len(filtered_users) else None,
            'prev': url_for('get_users', page=page-1, per_page=per_page, _external=True) if start > 0 else None
        },
        'meta': {
            'total': len(filtered_users),
            'page': page,
            'per_page': per_page
        }
    }
    
    return jsonify(response)

# GET single user
@app.route('/api/v1/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = next((u for u in users if u['id'] == user_id), None)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    response = {
        'data': user,
        'links': {
            'self': url_for('get_user', user_id=user_id, _external=True),
            'users': url_for('get_users', _external=True)
        }
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run()`,
        complexity: "Pagination: O(n), Filtering: O(n)"
      }
    },
    {
      title: "🧪 Testing & Documentation",
      id: "testing",
      description: "Ensuring API reliability through testing and comprehensive documentation.",
      keyPoints: [
        "Unit and integration testing",
        "Automated API testing tools",
        "OpenAPI/Swagger documentation",
        "Mock servers for development"
      ],
      detailedExplanation: [
        "Testing Strategies:",
        "- Unit tests for individual components",
        "- Integration tests for endpoint behavior",
        "- Contract tests for API stability",
        "- Load testing for performance",
        "",
        "Documentation Standards:",
        "- OpenAPI/Swagger for machine-readable docs",
        "- Interactive API explorers",
        "- Code samples in multiple languages",
        "- Change logs and version diffs",
        "",
        "Testing Tools:",
        "- pytest for Python APIs",
        "- Postman/Newman for collection testing",
        "- Locust for load testing",
        "- WireMock for API mocking",
        "",
        "CI/CD Integration:",
        "- Automated testing in pipelines",
        "- Documentation generation on deploy",
        "- Canary deployments for APIs",
        "- Monitoring and alerting"
      ],
      code: {
        python: `# API Testing with pytest
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_get_users(client):
    response = client.get('/api/v1/users')
    assert response.status_code == 200
    data = response.get_json()
    assert 'data' in data
    assert 'links' in data
    assert 'meta' in data

def test_create_user(client):
    new_user = {'name': 'Charlie', 'email': 'charlie@example.com'}
    response = client.post('/api/v1/users', json=new_user)
    assert response.status_code == 201
    data = response.get_json()
    assert data['name'] == new_user['name']
    assert 'id' in data

# OpenAPI Documentation with Flask
from flask_swagger_ui import get_swaggerui_blueprint

SWAGGER_URL = '/api/docs'
API_URL = '/api/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "User API"}
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route(API_URL)
def swagger():
    return jsonify({
        "openapi": "3.0.0",
        "info": {
            "title": "User API",
            "version": "1.0.0"
        },
        "paths": {
            "/users": {
                "get": {
                    "summary": "Get all users",
                    "responses": {
                        "200": {
                            "description": "List of users",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/UserList"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "User": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                        "email": {"type": "string"}
                    }
                },
                "UserList": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/User"}
                        }
                    }
                }
            }
        }
    })`,
        complexity: "Unit tests: O(1), Integration tests: O(n)"
      }
    }
  ];

  return (
    <div style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '2rem',
      background: 'linear-gradient(to bottom right, #eff6ff, #e0e7ff)',
      borderRadius: '20px',
      boxShadow: '0 10px 30px rgba(0,0,0,0.1)'
    }}>
      <h1 style={{
        fontSize: '3.5rem',
        fontWeight: '800',
        textAlign: 'center',
        background: 'linear-gradient(to right, #3b82f6, #6366f1)',
        WebkitBackgroundClip: 'text',
        backgroundClip: 'text',
        color: 'transparent',
        marginBottom: '3rem'
      }}>
        REST API Development for ML
      </h1>

      <div style={{
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        padding: '2rem',
        borderRadius: '12px',
        marginBottom: '3rem',
        borderLeft: '4px solid #3b82f6'
      }}>
        <h2 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#3b82f6',
          marginBottom: '1rem'
        }}>Deployment and Real-World Projects → REST API Development</h2>
        <p style={{
          color: '#374151',
          fontSize: '1.1rem',
          lineHeight: '1.6'
        }}>
          REST APIs provide the interface between machine learning models and client applications.
          This section covers building, securing, and maintaining production-grade APIs for ML systems.
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
            border: '1px solid #bfdbfe',
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
              color: '#3b82f6'
            }}>{section.title}</h2>
            <button
              onClick={() => toggleSection(section.id)}
              style={{
                background: 'linear-gradient(to right, #3b82f6, #6366f1)',
                color: 'white',
                padding: '0.75rem 1.5rem',
                borderRadius: '8px',
                border: 'none',
                cursor: 'pointer',
                fontWeight: '600',
                transition: 'all 0.2s ease',
                ':hover': {
                  transform: 'scale(1.05)',
                  boxShadow: '0 5px 15px rgba(59, 130, 246, 0.4)'
                }
              }}
            >
              {visibleSection === section.id ? "Collapse Section" : "Expand Section"}
            </button>
          </div>

          {visibleSection === section.id && (
            <div style={{ display: 'grid', gap: '2rem' }}>
              <div style={{
                backgroundColor: '#eff6ff',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#3b82f6',
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
                backgroundColor: '#e0e7ff',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#3b82f6',
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
                backgroundColor: '#dbeafe',
                padding: '1.5rem',
                borderRadius: '12px'
              }}>
                <h3 style={{
                  fontSize: '1.5rem',
                  fontWeight: '600',
                  color: '#3b82f6',
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
                  border: '2px solid #93c5fd'
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
        border: '1px solid #bfdbfe'
      }}>
        <h2 style={{
          fontSize: '2rem',
          fontWeight: '700',
          color: '#3b82f6',
          marginBottom: '2rem'
        }}>REST API Technologies for ML</h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            textAlign: 'left'
          }}>
            <thead style={{
              backgroundColor: '#3b82f6',
              color: 'white'
            }}>
              <tr>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Technology</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Use Case</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>ML Integration</th>
                <th style={{ padding: '1rem', fontSize: '1.1rem', fontWeight: '600' }}>Performance</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Flask", "Lightweight Python API", "Quick ML model deployment", "Good for small-medium loads"],
                ["FastAPI", "Modern async Python API", "Built-in data validation", "Excellent performance"],
                ["Django REST", "Full-featured Python API", "Admin interface for models", "Good for complex apps"],
                ["Express.js", "Node.js API framework", "JS ecosystem integration", "High performance"],
                ["Spring Boot", "Java enterprise API", "Large-scale ML systems", "Excellent performance"]
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
        backgroundColor: '#eff6ff',
        borderRadius: '16px',
        boxShadow: '0 5px 15px rgba(0,0,0,0.05)',
        border: '1px solid #bfdbfe'
      }}>
        <h3 style={{
          fontSize: '1.8rem',
          fontWeight: '700',
          color: '#3b82f6',
          marginBottom: '1.5rem'
        }}>ML API Best Practices</h3>
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
              color: '#3b82f6',
              marginBottom: '0.75rem'
            }}>API Design for ML</h4>
            <ul style={{
              listStyleType: 'disc',
              paddingLeft: '1.5rem',
              display: 'grid',
              gap: '0.75rem'
            }}>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Use batch endpoints for model predictions
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Implement async processing for long-running tasks
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Version APIs to allow model updates without breaking clients
              </li>
              <li style={{ color: '#374151', fontSize: '1.1rem' }}>
                Include model metadata in responses (version, confidence scores)
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
              color: '#3b82f6',
              marginBottom: '0.75rem'
            }}>Performance Optimization</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Caching:</strong> Cache model predictions when appropriate<br/>
              <strong>Batching:</strong> Process multiple requests together<br/>
              <strong>Load Balancing:</strong> Distribute prediction requests<br/>
              <strong>Model Warmup:</strong> Keep frequently used models loaded
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
              color: '#3b82f6',
              marginBottom: '0.75rem'
            }}>Monitoring & Maintenance</h4>
            <p style={{
              color: '#374151',
              fontSize: '1.1rem',
              lineHeight: '1.6'
            }}>
              <strong>Logging:</strong> Track prediction requests and performance<br/>
              <strong>Metrics:</strong> Monitor latency, throughput, errors<br/>
              <strong>Alerting:</strong> Set up alerts for anomalies<br/>
              <strong>Canary Deployments:</strong> Test new models with subset of traffic
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default RESTAPI;