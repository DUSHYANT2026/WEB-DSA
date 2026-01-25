import React, { useState } from "react";

export default function CodingAI() {
  const [input, setInput] = useState("");
  const [response, setResponse] = useState("");

  const handleGenerate = async () => {
    // 1. Send input to your backend
    const res = await fetch("/api/chat", {
      method: "POST",
      body: JSON.stringify({ prompt: input }),
      headers: { "Content-Type": "application/json" }
    });
    const data = await res.json();
    // 2. Set the result to display
    setResponse(data.text);
  };

  return (
    <div className="pt-24 min-h-screen bg-zinc-900 text-white p-8">
      <div className="max-w-4xl mx-auto space-y-6">
        <h1 className="text-3xl font-bold">Coding AI Assistant</h1>
        
        {/* Prompt Input */}
        <div className="flex gap-4">
          <textarea 
            className="w-full bg-zinc-800 border border-zinc-700 p-4 rounded-xl"
            placeholder="Ask for code (e.g., 'Write a binary search in Python')"
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button onClick={handleGenerate} className="bg-blue-600 px-6 rounded-xl">Generate</button>
        </div>

        {/* Result Display Area */}
        <div className="bg-zinc-950 border border-zinc-800 p-6 rounded-2xl min-h-[400px]">
          <pre className="whitespace-pre-wrap font-mono text-blue-300">
            {response || "AI output will appear here..."}
          </pre>
        </div>
      </div>
    </div>
  );
}