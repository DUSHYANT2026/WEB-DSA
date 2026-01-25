import React from "react";
import { NavLink } from "react-router-dom";
import { useTheme } from "../../ThemeContext.jsx";
import { 
  Infinity, 
  Combine, 
  ListChecks, 
  Trophy, 
  Layers, 
  ArrowRight, 
  Circle 
} from "lucide-react";

function Recursion() {
  const { darkMode } = useTheme();

  const topics = [
    {
      title: "Recursion Basics",
      desc: "Base cases, stack frames, and the anatomy of a recursive call.",
      path: "/recursion1",
      color: "blue",
      difficulty: "Beginner",
      icon: <Infinity />,
      stats: "12 Lessons"
    },
    {
      title: "Backtracking Core",
      desc: "Mastering the 'Explore, Undo, Repeat' pattern. Permutations & subsets.",
      path: "/recursion2",
      color: "emerald",
      difficulty: "Intermediate",
      icon: <Combine />,
      stats: "10 Patterns"
    },
    {
      title: "Leetcode Top 40",
      desc: "Classic problems: N-Queens, Sudoku Solver, and Combination Sum.",
      path: "/recursion3",
      color: "amber",
      difficulty: "Advanced",
      icon: <ListChecks />,
      stats: "40 Questions"
    },
    {
      title: "MAANG Tier Hard",
      desc: "Complex recursion with pruning and state optimization for elite roles.",
      path: "/recursion4",
      color: "rose",
      difficulty: "Expert",
      icon: <Trophy />,
      stats: "15 Challenges"
    },
    {
      title: "Advanced Patterns",
      desc: "Divide & Conquer, Recursive Tree pruning, and tailored state management.",
      path: "/recursion5",
      color: "indigo",
      difficulty: "Expert",
      icon: <Layers />,
      stats: "8 Deep Dives"
    }
  ];

  const colorMap = {
    blue: "from-blue-500/20 to-indigo-500/20 border-blue-500/20 text-blue-400",
    emerald: "from-emerald-500/20 to-teal-500/20 border-emerald-500/20 text-emerald-400",
    amber: "from-amber-500/20 to-orange-500/20 border-amber-500/20 text-amber-400",
    rose: "from-rose-500/20 to-pink-500/20 border-rose-500/20 text-rose-400",
    indigo: "from-indigo-500/20 to-blue-500/20 border-indigo-500/20 text-indigo-400",
  };

  return (
    <div className={`min-h-screen py-16 px-6 transition-colors duration-500 ${darkMode ? "bg-[#0b0f1a]" : "bg-slate-50"}`}>
      <div className="max-w-6xl mx-auto">
        
        {/* Header Section */}
        <div className="flex flex-col items-center mb-16 text-center">
          <span className={`px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-widest mb-4 ${darkMode ? "bg-pink-500/10 text-pink-400" : "bg-pink-100 text-pink-600"}`}>
            Deep Logic
          </span>
          <h1 className={`text-5xl font-black tracking-tight mb-4 ${darkMode ? "text-white" : "text-slate-900"}`}>
            Recursion & <span className="text-transparent bg-clip-text bg-gradient-to-r from-pink-500 to-violet-600">Backtracking.</span>
          </h1>
          <p className={`text-lg max-w-xl ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
            Master the art of solving problems by breaking them into smaller versions of themselves and exploring all possibilities.
          </p>
        </div>

        {/* Bento Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-8">
          {topics.map((topic, index) => (
            <NavLink 
              to={topic.path} 
              key={index} 
              className={`group relative flex flex-col ${
                index < 2 ? "lg:col-span-3" : // First row: 2 items
                index < 4 ? "lg:col-span-3" : // Second row: 2 items
                "lg:col-span-6"              // Last item: Featured Full-width
              }`}
            >
              <div className={`
                relative overflow-hidden rounded-[2.5rem] border p-8 h-full flex flex-col transition-all duration-500
                ${darkMode 
                  ? "bg-zinc-900/40 border-zinc-800/50 backdrop-blur-xl hover:bg-zinc-800/60 hover:border-pink-500/50 shadow-2xl shadow-black/50" 
                  : "bg-white border-slate-200 hover:border-pink-400 hover:shadow-xl hover:shadow-pink-500/10"}
                group-hover:-translate-y-2
              `}>
                
                {/* Visual Glow */}
                <div className={`absolute top-0 right-0 -mr-16 -mt-16 w-48 h-48 bg-gradient-to-br ${colorMap[topic.color]} blur-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />

                <div className="relative z-10 flex flex-col h-full">
                  <div className="flex justify-between items-start mb-8">
                    <div className={`p-4 rounded-2xl transition-colors duration-300 ${darkMode ? "bg-zinc-800 text-pink-400 group-hover:bg-pink-500 group-hover:text-white" : "bg-slate-100 text-slate-700 group-hover:bg-pink-600 group-hover:text-white"}`}>
                      {React.cloneElement(topic.icon, { size: 24 })}
                    </div>
                    <span className={`text-[10px] font-bold uppercase tracking-wider px-3 py-1 rounded-full ${darkMode ? "bg-zinc-800 text-slate-400" : "bg-slate-100 text-slate-500"}`}>
                      {topic.difficulty}
                    </span>
                  </div>

                  <h3 className={`text-xl font-bold mb-3 transition-colors ${darkMode ? "text-white group-hover:text-pink-400" : "text-slate-900 group-hover:text-pink-600"}`}>
                    {topic.title}
                  </h3>
                  <p className={`text-sm leading-relaxed mb-8 ${darkMode ? "text-slate-400" : "text-slate-500"}`}>
                    {topic.desc}
                  </p>

                  <div className={`mt-auto flex items-center justify-between pt-6 border-t ${darkMode ? "border-zinc-800" : "border-slate-100"}`}>
                    <span className={`flex items-center text-xs font-medium ${darkMode ? "text-slate-500" : "text-slate-400"}`}>
                      <Circle className={`w-2 h-2 mr-2 fill-pink-500 stroke-none ${index === 1 ? 'animate-bounce' : ''}`} />
                      {topic.stats}
                    </span>
                    <div className={`flex items-center text-sm font-bold transition-all ${darkMode ? "text-white group-hover:text-pink-400" : "text-slate-900 group-hover:text-pink-600"} group-hover:translate-x-1`}>
                      Explore <ArrowRight className="ml-2 w-4 h-4" />
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

export default Recursion;