import React from "react";
import { NavLink } from "react-router-dom";
import { useTheme } from "../../ThemeContext.jsx";
import { BookOpen, Code, Search, Grid, Award, Bolt, ArrowRight, Circle } from "lucide-react";

function Arrays() {
  const { darkMode } = useTheme();

  const topics = [
    {
      title: "Array Essentials",
      desc: "Foundational logic and memory structures.",
      path: "/Narray1",
      color: "blue",
      difficulty: "Beginner",
      icon: <BookOpen />,
      stats: "12 Lessons"
    },
    {
      title: "String Patterns",
      desc: "Manipulations, Regex, and Palindromes.",
      path: "/Narray2",
      color: "emerald",
      difficulty: "Beginner",
      icon: <Code />,
      stats: "8 Lessons"
    },
    {
      title: "Binary Search",
      desc: "Divide and conquer optimization techniques.",
      path: "/Narray3",
      color: "amber",
      difficulty: "Intermediate",
      icon: <Search />,
      stats: "10 Lessons"
    },
    {
      title: "Matrix Mastery",
      desc: "2D traversals and spiral transformations.",
      path: "/Narray4",
      color: "rose",
      difficulty: "Intermediate",
      icon: <Grid />,
      stats: "15 Lessons"
    },
    {
      title: "Leetcode Top 100",
      desc: "Patterns most likely to appear in interviews.",
      path: "/Narray5",
      color: "violet",
      difficulty: "Advanced",
      icon: <Award />,
      stats: "100 Questions"
    },
    {
      title: "MAANG Architect",
      desc: "Hard-level logic for top-tier companies.",
      path: "/Narray6",
      color: "slate",
      difficulty: "Expert",
      icon: <Bolt />,
      stats: "20 Challenges"
    }
  ];

  const colorMap = {
    blue: "from-blue-500/20 to-indigo-500/20 border-blue-500/20 text-blue-400",
    emerald: "from-emerald-500/20 to-teal-500/20 border-emerald-500/20 text-emerald-400",
    amber: "from-amber-500/20 to-orange-500/20 border-amber-500/20 text-amber-400",
    rose: "from-rose-500/20 to-pink-500/20 border-rose-500/20 text-rose-400",
    violet: "from-violet-500/20 to-purple-500/20 border-violet-500/20 text-violet-400",
    slate: "from-slate-600/20 to-slate-800/20 border-slate-500/20 text-slate-400",
  };

  return (
    <div className={`min-h-screen py-16 px-6 transition-colors duration-500 ${darkMode ? "bg-[#0b0f1a]" : "bg-slate-50"}`}>
      <div className="max-w-6xl mx-auto">
        
        {/* Modern Header */}
        <div className="flex flex-col items-center mb-16 text-center">
          <span className={`px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-widest mb-4 ${darkMode ? "bg-orange-500/10 text-orange-400" : "bg-orange-100 text-orange-600"}`}>
            Curriculum 2026
          </span>
          <h1 className={`text-5xl font-black tracking-tight mb-4 ${darkMode ? "text-white" : "text-slate-900"}`}>
            Master the <span className="text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-red-500">Fundamentals.</span>
          </h1>
          <p className={`text-lg max-w-xl ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
            Structured learning paths designed to take you from syntax basics to 
            architecting complex algorithmic solutions.
          </p>
        </div>

        {/* Bento Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {topics.map((topic, index) => (
            <NavLink 
              to={topic.path} 
              key={index}
              className="group relative"
            >
              <div className={`
                relative overflow-hidden rounded-[2.5rem] border p-8 h-full flex flex-col transition-all duration-500
                ${darkMode 
                  ? "bg-zinc-900/40 border-zinc-800/50 backdrop-blur-xl hover:bg-zinc-800/60 hover:border-orange-500/50 shadow-2xl shadow-black/50" 
                  : "bg-white border-slate-200 hover:border-orange-400 hover:shadow-xl hover:shadow-orange-500/10"}
                group-hover:-translate-y-2
              `}>
                
                {/* Decorative Background Glow (Hover only) */}
                <div className={`absolute top-0 right-0 -mr-16 -mt-16 w-40 h-40 bg-gradient-to-br ${colorMap[topic.color]} blur-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />

                <div className="relative z-10 flex flex-col h-full">
                  <div className="flex justify-between items-start mb-8">
                    <div className={`p-4 rounded-2xl transition-colors duration-300 ${darkMode ? "bg-zinc-800 text-orange-400 group-hover:bg-orange-500 group-hover:text-white" : "bg-slate-100 text-slate-700 group-hover:bg-orange-600 group-hover:text-white"}`}>
                      {React.cloneElement(topic.icon, { size: 24 })}
                    </div>
                    <span className={`text-[10px] font-bold uppercase tracking-wider px-3 py-1 rounded-full ${darkMode ? "bg-zinc-800 text-slate-400" : "bg-slate-100 text-slate-500"}`}>
                      {topic.difficulty}
                    </span>
                  </div>

                  <h3 className={`text-xl font-bold mb-3 transition-colors ${darkMode ? "text-white group-hover:text-orange-400" : "text-slate-900 group-hover:text-orange-600"}`}>
                    {topic.title}
                  </h3>
                  <p className={`text-sm leading-relaxed mb-8 ${darkMode ? "text-slate-400" : "text-slate-500"}`}>
                    {topic.desc}
                  </p>

                  <div className={`mt-auto flex items-center justify-between pt-6 border-t ${darkMode ? "border-zinc-800" : "border-slate-50"}`}>
                    <span className={`flex items-center text-xs font-medium ${darkMode ? "text-slate-500" : "text-slate-400"}`}>
                      <Circle className="w-2 h-2 mr-2 fill-emerald-500 stroke-none" />
                      {topic.stats}
                    </span>
                    <div className={`flex items-center text-sm font-bold transition-all ${darkMode ? "text-white group-hover:text-orange-400" : "text-slate-900 group-hover:text-orange-600"} group-hover:translate-x-1`}>
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

export default Arrays;