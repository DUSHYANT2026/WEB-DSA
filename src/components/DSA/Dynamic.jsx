import React from "react";
import { NavLink } from "react-router-dom";
import { useTheme } from "../../ThemeContext.jsx";
import { 
  Box, 
  Grid2x2, 
  ListOrdered, 
  Type, 
  GitCompare, 
  Trophy, 
  ArrowRight, 
  Circle 
} from "lucide-react";

function Dynamic() {
  const { darkMode } = useTheme();

  const topics = [
    {
      title: "1D DP Foundations",
      desc: "Memoization vs. Tabulation essentials. Climbing stairs to Fibonacci.",
      path: "/dynamic1",
      color: "blue",
      difficulty: "Intermediate",
      icon: <Box />,
      stats: "15 Lessons"
    },
    {
      title: "Multi-D & Grid DP",
      desc: "Pathfinding and optimization in 2D/3D grids and matrices.",
      path: "/dynamic2",
      color: "emerald",
      difficulty: "Advanced",
      icon: <Grid2x2 />,
      stats: "12 Patterns"
    },
    {
      title: "Sequence & LIS",
      desc: "Mastering Longest Increasing Subsequences and subset patterns.",
      path: "/dynamic3",
      color: "violet",
      difficulty: "Advanced",
      icon: <ListOrdered />,
      stats: "10 Lessons"
    },
    {
      title: "String DP",
      desc: "LCS, Edit Distance, and Palindromic partitioning logic.",
      path: "/dynamic4",
      color: "rose",
      difficulty: "Advanced",
      icon: <Type />,
      stats: "9 Problems"
    },
    {
      title: "Leetcode Hard 50",
      desc: "The most vital DP questions curated for top performance.",
      path: "/dynamic5",
      color: "indigo",
      difficulty: "Expert",
      icon: <GitCompare />,
      stats: "50 Questions"
    },
    {
      title: "MAANG Architect",
      desc: "The most brutal DP challenges asked by Google and Meta.",
      path: "/dynamic6",
      color: "slate",
      difficulty: "Expert",
      icon: <Trophy />,
      stats: "20 Challenges"
    }
  ];

  const colorMap = {
    blue: "from-blue-500/20 to-indigo-500/20 border-blue-500/20 text-blue-400",
    emerald: "from-emerald-500/20 to-teal-500/20 border-emerald-500/20 text-emerald-400",
    violet: "from-violet-500/20 to-purple-500/20 border-violet-500/20 text-violet-400",
    rose: "from-rose-500/20 to-pink-500/20 border-rose-500/20 text-rose-400",
    indigo: "from-indigo-500/20 to-blue-500/20 border-indigo-500/20 text-indigo-400",
    slate: "from-slate-600/20 to-slate-800/20 border-slate-500/20 text-slate-400",
  };

  return (
    <div className={`min-h-screen py-16 px-6 transition-colors duration-500 ${darkMode ? "bg-[#0b0f1a]" : "bg-slate-50"}`}>
      <div className="max-w-6xl mx-auto">
        
        {/* Header Section */}
        <div className="flex flex-col items-center mb-16 text-center">
          <span className={`px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-widest mb-4 ${darkMode ? "bg-violet-500/10 text-violet-400" : "bg-violet-100 text-violet-600"}`}>
            Optimization Mastery
          </span>
          <h1 className={`text-5xl font-black tracking-tight mb-4 ${darkMode ? "text-white" : "text-slate-900"}`}>
            Dynamic <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-500 to-fuchsia-600">Programming.</span>
          </h1>
          <p className={`text-lg max-w-xl ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
            Break down complex problems into overlapping subproblems and master the art of space-time trade-offs.
          </p>
        </div>

        {/* Bento Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {topics.map((topic, index) => (
            <NavLink to={topic.path} key={index} className="group relative">
              <div className={`
                relative overflow-hidden rounded-[2.5rem] border p-8 h-full flex flex-col transition-all duration-500
                ${darkMode 
                  ? "bg-zinc-900/40 border-zinc-800/50 backdrop-blur-xl hover:bg-zinc-800/60 hover:border-violet-500/50 shadow-2xl shadow-black/50" 
                  : "bg-white border-slate-200 hover:border-violet-400 hover:shadow-xl hover:shadow-violet-500/10"}
                group-hover:-translate-y-2
              `}>
                
                {/* Visual Glow Accent */}
                <div className={`absolute top-0 right-0 -mr-16 -mt-16 w-40 h-40 bg-gradient-to-br ${colorMap[topic.color]} blur-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />

                <div className="relative z-10 flex flex-col h-full">
                  <div className="flex justify-between items-start mb-8">
                    <div className={`p-4 rounded-2xl transition-all duration-300 ${darkMode ? "bg-zinc-800 text-violet-400 group-hover:bg-violet-500 group-hover:text-white" : "bg-slate-100 text-slate-700 group-hover:bg-violet-600 group-hover:text-white"}`}>
                      {React.cloneElement(topic.icon, { size: 24 })}
                    </div>
                    <span className={`text-[10px] font-bold uppercase tracking-wider px-3 py-1 rounded-full ${darkMode ? "bg-zinc-800 text-slate-400" : "bg-slate-100 text-slate-500"}`}>
                      {topic.difficulty}
                    </span>
                  </div>

                  <h3 className={`text-xl font-bold mb-3 transition-colors ${darkMode ? "text-white group-hover:text-violet-400" : "text-slate-900 group-hover:text-violet-600"}`}>
                    {topic.title}
                  </h3>
                  <p className={`text-sm leading-relaxed mb-8 ${darkMode ? "text-slate-400" : "text-slate-500"}`}>
                    {topic.desc}
                  </p>

                  <div className={`mt-auto flex items-center justify-between pt-6 border-t ${darkMode ? "border-zinc-800" : "border-slate-100"}`}>
                    <span className={`flex items-center text-xs font-medium ${darkMode ? "text-slate-500" : "text-slate-400"}`}>
                      <Circle className="w-2 h-2 mr-2 fill-fuchsia-500 stroke-none animate-pulse" />
                      {topic.stats}
                    </span>
                    <div className={`flex items-center text-sm font-bold transition-all ${darkMode ? "text-white group-hover:text-violet-400" : "text-slate-900 group-hover:text-violet-600"} group-hover:translate-x-1`}>
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

export default Dynamic;