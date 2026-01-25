import React from "react";
import { NavLink } from "react-router-dom";
import { useTheme } from "../../ThemeContext.jsx";
import { Binary, Calculator, BrainCircuit, Terminal, ArrowRight, Circle } from "lucide-react";

function Bitm() {
  const { darkMode } = useTheme();

  const topics = [
    {
      title: "Bitwise Foundations",
      desc: "AND, OR, XOR, and bit-shifting logic for performance optimization.",
      path: "/bit1",
      color: "blue",
      difficulty: "Intermediate",
      icon: <Binary />,
      stats: "15 Concepts"
    },
    {
      title: "Algorithmic Maths",
      desc: "Prime numbers, GCD, Sieve of Eratosthenes, and Modular Arithmetic.",
      path: "/bit2",
      color: "amber",
      difficulty: "Intermediate",
      icon: <Calculator />,
      stats: "10 Lessons"
    },
    {
      title: "Leetcode Patterns",
      desc: "Common bit-masking and mathematical patterns in interview problems.",
      path: "/bit3",
      color: "purple",
      difficulty: "Advanced",
      icon: <BrainCircuit />,
      stats: "40 Questions"
    },
    {
      title: "MAANG Bit-Magic",
      desc: "Extremely optimized solutions for low-level system design questions.",
      path: "/bit4",
      color: "rose",
      difficulty: "Expert",
      icon: <Terminal />,
      stats: "12 Challenges"
    }
  ];

  const colorMap = {
    blue: "from-blue-500/20 to-cyan-500/20 border-blue-500/20 text-blue-400",
    amber: "from-amber-500/20 to-orange-500/20 border-amber-500/20 text-amber-400",
    purple: "from-purple-500/20 to-indigo-500/20 border-purple-500/20 text-purple-400",
    rose: "from-rose-500/20 to-red-500/20 border-rose-500/20 text-rose-400",
  };

  return (
    <div className={`min-h-screen py-16 px-6 transition-colors duration-500 ${darkMode ? "bg-[#0b0f1a]" : "bg-slate-50"}`}>
      <div className="max-w-6xl mx-auto">
        
        {/* Header Section */}
        <div className="flex flex-col items-center mb-16 text-center">
          <span className={`px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-widest mb-4 ${darkMode ? "bg-blue-500/10 text-blue-400" : "bg-blue-100 text-blue-600"}`}>
            Efficiency Module
          </span>
          <h1 className={`text-5xl font-black tracking-tight mb-4 ${darkMode ? "text-white" : "text-slate-900"}`}>
            Bit-Magic & <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500">Discrete Maths.</span>
          </h1>
          <p className={`text-lg max-w-xl ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
            Go beyond standard logic by manipulating raw data and mastering the mathematical foundations of CS.
          </p>
        </div>

        {/* Bento Grid - 2x2 for Bitm */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {topics.map((topic, index) => (
            <NavLink 
              to={topic.path} 
              key={index}
              className="group relative"
            >
              <div className={`
                relative overflow-hidden rounded-[2.5rem] border p-8 h-full flex flex-col transition-all duration-500
                ${darkMode 
                  ? "bg-zinc-900/40 border-zinc-800/50 backdrop-blur-xl hover:bg-zinc-800/60 hover:border-blue-500/50 shadow-2xl shadow-black/50" 
                  : "bg-white border-slate-200 hover:border-blue-400 hover:shadow-xl hover:shadow-blue-500/10"}
                group-hover:-translate-y-2
              `}>
                
                {/* Visual Accent */}
                <div className={`absolute top-0 right-0 -mr-16 -mt-16 w-48 h-48 bg-gradient-to-br ${colorMap[topic.color]} blur-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />

                <div className="relative z-10 flex flex-col h-full">
                  <div className="flex justify-between items-start mb-8">
                    <div className={`p-4 rounded-2xl transition-all duration-300 ${darkMode ? "bg-zinc-800 text-blue-400 group-hover:bg-blue-500 group-hover:text-white" : "bg-slate-100 text-slate-700 group-hover:bg-blue-600 group-hover:text-white"}`}>
                      {React.cloneElement(topic.icon, { size: 28 })}
                    </div>
                    <span className={`text-[10px] font-bold uppercase tracking-wider px-3 py-1 rounded-full ${darkMode ? "bg-zinc-800 text-slate-400" : "bg-slate-100 text-slate-500"}`}>
                      {topic.difficulty}
                    </span>
                  </div>

                  <h3 className={`text-2xl font-bold mb-3 transition-colors ${darkMode ? "text-white group-hover:text-blue-400" : "text-slate-900 group-hover:text-blue-600"}`}>
                    {topic.title}
                  </h3>
                  <p className={`text-base leading-relaxed mb-8 ${darkMode ? "text-slate-400" : "text-slate-500"}`}>
                    {topic.desc}
                  </p>

                  <div className={`mt-auto flex items-center justify-between pt-6 border-t ${darkMode ? "border-zinc-800" : "border-slate-100"}`}>
                    <span className={`flex items-center text-xs font-medium ${darkMode ? "text-slate-500" : "text-slate-400"}`}>
                      <Circle className="w-2 h-2 mr-2 fill-blue-500 animate-pulse stroke-none" />
                      {topic.stats}
                    </span>
                    <div className={`flex items-center text-sm font-bold transition-all ${darkMode ? "text-white group-hover:text-blue-400" : "text-slate-900 group-hover:text-blue-600"} group-hover:translate-x-2`}>
                      Analyze <ArrowRight className="ml-2 w-4 h-4" />
                    </div>
                  </div>
                </div>
              </div>
            </NavLink>
          ))}
        </div>
      </div>
    </div>
  );
}

export default Bitm;