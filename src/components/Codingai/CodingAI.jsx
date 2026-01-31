import React, { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { atomDark } from "react-syntax-highlighter/dist/esm/styles/prism";

export default function CodingAI() {
  const [input, setInput] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [copyStatus, setCopyStatus] = useState("Copy Code");

  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        handleGenerate();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [input, loading]);

  const handleGenerate = async () => {
    if (!input.trim() || loading) return;
    setLoading(true);
    setResponse("");

    try {
      const res = await fetch("http://localhost:5000/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: input }),
      });
      const data = await res.json();
      setResponse(data.result || data.text || "No response received.");
    } catch (err) {
      setResponse("❌ **Connection Error**: Ensure your backend is running on port 5000.");
    } finally {
      setLoading(false);
    }
  };

  const copyCodeOnly = () => {
    const codeBlockRegex = /```(?:[a-z]*)\n([\s\S]*?)\n```/g;
    let matches = [];
    let match;

    while ((match = codeBlockRegex.exec(response)) !== null) {
      matches.push(match[1]);
    }

    const textToCopy = matches.length > 0 ? matches.join("\n\n") : response;
    
    navigator.clipboard.writeText(textToCopy);
    setCopyStatus("✅ Copied!");
    setTimeout(() => setCopyStatus("Copy Code"), 2000);
  };

  const handleClear = () => {
    setInput("");
    setResponse("");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0a0a0c] via-[#0c0c0e] to-[#0e0e12] text-zinc-200 selection:bg-blue-500/30 font-sans">
      {/* Background Grid Pattern */}
      <div className="fixed inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:64px_64px] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000,transparent)]" />
      
      <div className="relative max-w-5xl mx-auto px-4 sm:px-6 py-8 sm:py-16 lg:py-20">
        
        {/* Header */}
        <div className="mb-12 space-y-3 text-center">
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="relative">
              <div className="absolute inset-0 bg-blue-500 blur-xl opacity-50" />
              <div className="relative bg-gradient-to-br from-blue-500 to-blue-600 p-3 rounded-2xl shadow-2xl">
                <span className="text-2xl">🤖</span>
              </div>
            </div>
          </div>
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight bg-gradient-to-r from-white via-zinc-100 to-zinc-400 bg-clip-text text-transparent">
            Coding AI Assistant
          </h1>
          <p className="text-zinc-400 text-base sm:text-lg max-w-2xl mx-auto">
            Generate clean, optimized code snippets with AI-powered assistance
          </p>
        </div>

        {/* Input Card */}
        <div className="mb-6 group">
          <div className="relative bg-gradient-to-b from-zinc-900/80 to-zinc-900/40 backdrop-blur-xl border border-zinc-800/50 rounded-3xl transition-all duration-300 hover:border-zinc-700/50 shadow-2xl shadow-black/20">
            {/* Glow Effect on Focus */}
            <div className="absolute -inset-[1px] bg-gradient-to-r from-blue-500/0 via-blue-500/50 to-blue-500/0 rounded-3xl opacity-0 group-focus-within:opacity-100 blur transition-opacity duration-500" />
            
            <div className="relative">
              <textarea
                className="w-full bg-transparent p-6 sm:p-8 pb-20 rounded-3xl resize-none focus:outline-none min-h-[160px] sm:min-h-[180px] text-zinc-100 placeholder:text-zinc-500 text-base leading-relaxed"
                placeholder="Describe what you want to build... (e.g., 'Create a debounce function in TypeScript' or 'React hook for fetching data')"
                value={input}
                onChange={(e) => setInput(e.target.value)}
              />
              
              {/* Bottom Actions Bar */}
              <div className="absolute bottom-0 left-0 right-0 p-4 sm:p-6 border-t border-zinc-800/50 bg-zinc-900/60 backdrop-blur-sm rounded-b-3xl">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <kbd className="hidden sm:inline-flex items-center gap-1 px-2 py-1 text-[10px] font-semibold text-zinc-500 bg-zinc-800/50 border border-zinc-700/50 rounded-lg">
                      <span>⌘</span>
                      <span>↵</span>
                    </kbd>
                    <span className="text-xs text-zinc-600 hidden sm:block">
                      Quick submit
                    </span>
                    {input.trim() && (
                      <button
                        onClick={handleClear}
                        className="text-xs text-zinc-500 hover:text-zinc-300 px-3 py-1 rounded-lg hover:bg-zinc-800/50 transition-all"
                      >
                        Clear
                      </button>
                    )}
                  </div>
                  
                  <button
                    onClick={handleGenerate}
                    disabled={loading || !input.trim()}
                    className={`relative px-6 sm:px-8 py-3 rounded-xl font-semibold text-sm transition-all duration-200
                      ${loading || !input.trim()
                        ? "bg-zinc-800 text-zinc-500 cursor-not-allowed" 
                        : "bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white shadow-lg shadow-blue-900/30 hover:shadow-blue-900/50 active:scale-95"}`}
                  >
                    {loading && (
                      <span className="absolute inset-0 flex items-center justify-center">
                        <span className="w-4 h-4 border-2 border-zinc-600 border-t-transparent rounded-full animate-spin" />
                      </span>
                    )}
                    <span className={loading ? "opacity-0" : ""}>
                      {loading ? "Processing..." : "Generate Code"}
                    </span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Output Section */}
        {(response || loading) && (
          <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex justify-between items-center px-1">
              <div className="flex items-center gap-2">
                <div className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-pulse" />
                <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider">
                  Generated Solution
                </h3>
              </div>
              {response && !loading && (
                <button
                  onClick={copyCodeOnly}
                  className="group flex items-center gap-2 text-xs font-semibold py-2 px-4 rounded-xl bg-zinc-800/50 border border-zinc-700/50 hover:bg-zinc-700/50 hover:border-zinc-600 transition-all duration-200 text-blue-400 hover:text-blue-300"
                >
                  <svg 
                    className="w-3.5 h-3.5 transition-transform group-hover:scale-110" 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  {copyStatus}
                </button>
              )}
            </div>

            <div className="bg-gradient-to-b from-zinc-900/80 to-zinc-900/40 backdrop-blur-xl border border-zinc-800/50 rounded-3xl overflow-hidden shadow-2xl shadow-black/20">
              <div className="p-6 sm:p-8 overflow-auto max-h-[600px] leading-relaxed custom-scrollbar">
                {loading ? (
                  <div className="space-y-6">
                    <div className="space-y-3">
                      <div className="h-4 bg-gradient-to-r from-zinc-800 via-zinc-700 to-zinc-800 rounded-lg w-3/4 animate-pulse" />
                      <div className="h-4 bg-gradient-to-r from-zinc-800 via-zinc-700 to-zinc-800 rounded-lg w-1/2 animate-pulse" />
                    </div>
                    <div className="h-48 bg-gradient-to-br from-zinc-800/50 via-zinc-800/30 to-zinc-800/50 rounded-2xl animate-pulse border border-zinc-700/30" />
                    <div className="space-y-3">
                      <div className="h-4 bg-gradient-to-r from-zinc-800 via-zinc-700 to-zinc-800 rounded-lg w-2/3 animate-pulse" />
                      <div className="h-4 bg-gradient-to-r from-zinc-800 via-zinc-700 to-zinc-800 rounded-lg w-3/4 animate-pulse" />
                    </div>
                  </div>
                ) : (
                  <div className="prose prose-invert prose-pre:p-0 max-w-none prose-headings:text-zinc-100 prose-p:text-zinc-300 prose-strong:text-zinc-200">
                    <ReactMarkdown
                      components={{
                        code({ node, inline, className, children, ...props }) {
                          const match = /language-(\w+)/.exec(className || "");
                          return !inline && match ? (
                            <div className="relative group/code">
                              <SyntaxHighlighter
                                style={atomDark}
                                language={match[1]}
                                PreTag="div"
                                className="!rounded-2xl !my-6 !bg-[#0d0d0f] !border !border-zinc-800/50 shadow-xl !text-sm"
                                customStyle={{
                                  padding: '1.5rem',
                                  background: 'linear-gradient(135deg, #0d0d0f 0%, #121214 100%)',
                                }}
                                {...props}
                              >
                                {String(children).replace(/\n$/, "")}
                              </SyntaxHighlighter>
                            </div>
                          ) : (
                            <code className="bg-zinc-800/70 px-2 py-1 rounded-lg text-blue-300 font-mono text-sm border border-zinc-700/30" {...props}>
                              {children}
                            </code>
                          );
                        },
                        h1: ({ children }) => (
                          <h1 className="text-2xl font-bold mt-8 mb-4 text-zinc-100">{children}</h1>
                        ),
                        h2: ({ children }) => (
                          <h2 className="text-xl font-bold mt-6 mb-3 text-zinc-200">{children}</h2>
                        ),
                        h3: ({ children }) => (
                          <h3 className="text-lg font-semibold mt-4 mb-2 text-zinc-300">{children}</h3>
                        ),
                        ul: ({ children }) => (
                          <ul className="space-y-2 my-4">{children}</ul>
                        ),
                        ol: ({ children }) => (
                          <ol className="space-y-2 my-4">{children}</ol>
                        ),
                        li: ({ children }) => (
                          <li className="text-zinc-300 leading-relaxed">{children}</li>
                        ),
                      }}
                    >
                      {response}
                    </ReactMarkdown>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Empty State */}
        {!response && !loading && (
          <div className="mt-16 text-center space-y-6 opacity-40">
            <div className="flex justify-center gap-8 text-4xl">
              <span className="animate-bounce" style={{ animationDelay: '0s' }}>💡</span>
              <span className="animate-bounce" style={{ animationDelay: '0.1s' }}>⚡</span>
              <span className="animate-bounce" style={{ animationDelay: '0.2s' }}>🚀</span>
            </div>
            <p className="text-zinc-600 text-sm">
              Start by describing what you want to build above
            </p>
          </div>
        )}
      </div>

      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(39, 39, 42, 0.3);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(63, 63, 70, 0.8);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(82, 82, 91, 0.9);
        }
        @keyframes slide-in-from-bottom-4 {
          from {
            transform: translateY(1rem);
            opacity: 0;
          }
          to {
            transform: translateY(0);
            opacity: 1;
          }
        }
        .animate-in {
          animation: slide-in-from-bottom-4 0.5s ease-out;
        }
      `}</style>
    </div>
  );
}