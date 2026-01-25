import React from "react";
import { NavLink } from "react-router-dom";
import { useTheme } from "../../ThemeContext.jsx";
import { Map, Layers, LayoutList, List, Hash, BarChart4, ArrowRight, Circle } from "lucide-react";

function STL() {
  const { darkMode } = useTheme();

  const topics = [
    {
      title: "Maps & Dictionaries",
      desc: "Ordered and Unordered Key-Value storage logic.",
      path: "/stl1",
      color: "indigo",
      difficulty: "Intermediate",
      icon: <Map />,
      stats: "10 Lessons"
    },
    {
      title: "Set Operations",
      desc: "Multi-set, Ordered, and Unordered unique collections.",
      path: "/Stl2",
      color: "cyan",
      difficulty: "Intermediate",
      icon: <Layers />,
      stats: "7 Lessons"
    },
    {
      title: "Linear Containers",
      desc: "Stack, Queue, and Deque (Double-ended queue) mastery.",
      path: "/Stl3",
      color: "violet",
      difficulty: "Beginner",
      icon: <LayoutList />,
      stats: "12 Lessons"
    },
    {
      title: "Dynamic Sequences",
      desc: "Deep dive into Vectors and Linked Lists internals.",
      path: "/Stl4",
      color: "fuchsia",
      difficulty: "Beginner",
      icon: <List />,
      stats: "9 Lessons"
    },
    {
      title: "Hashing Algorithms",
      desc: "Most asked Hash Table problems from Leetcode.",
      path: "/Stl5",
      color: "blue",
      difficulty: "Advanced",
      icon: <Hash />,
      stats: "45 Questions"
    },
    {
      title: "MAANG STL Patterns",
      desc: "Complex container nesting for top-tier interviews.",
      path: "/Stl6",
      color: "slate",
      difficulty: "Expert",
      icon: <BarChart4 />,
      stats: "15 Challenges"
    }
  ];

  const colorMap = {
    indigo: "from-indigo-500/20 to-blue-500/20 border-indigo-500/20 text-indigo-400",
    cyan: "from-cyan-500/20 to-teal-500/20 border-cyan-500/20 text-cyan-400",
    violet: "from-violet-500/20 to-purple-500/20 border-violet-500/20 text-violet-400",
    fuchsia: "from-fuchsia-500/20 to-pink-500/20 border-fuchsia-500/20 text-fuchsia-400",
    blue: "from-blue-500/20 to-sky-500/20 border-blue-500/20 text-blue-400",
    slate: "from-slate-600/20 to-slate-800/20 border-slate-500/20 text-slate-400",
  };

  return (
    <div className={`min-h-screen py-16 px-6 transition-colors duration-500 ${darkMode ? "bg-[#0b0f1a]" : "bg-slate-50"}`}>
      <div className="max-w-6xl mx-auto">
        
        {/* Header Section */}
        <div className="flex flex-col items-center mb-16 text-center">
          <span className={`px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-widest mb-4 ${darkMode ? "bg-indigo-500/10 text-indigo-400" : "bg-indigo-100 text-indigo-600"}`}>
            C++ Toolkit
          </span>
          <h1 className={`text-5xl font-black tracking-tight mb-4 ${darkMode ? "text-white" : "text-slate-900"}`}>
            Standard <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-500">Template Library.</span>
          </h1>
          <p className={`text-lg max-w-xl ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
            Optimize your workflow with pre-built data structures and powerful 
            algorithmic containers.
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
                  ? "bg-zinc-900/40 border-zinc-800/50 backdrop-blur-xl hover:bg-zinc-800/60 hover:border-indigo-500/50 shadow-2xl shadow-black/50" 
                  : "bg-white border-slate-200 hover:border-indigo-400 hover:shadow-xl hover:shadow-indigo-500/10"}
                group-hover:-translate-y-2
              `}>
                
                {/* Subtle Glow Effect */}
                <div className={`absolute top-0 right-0 -mr-16 -mt-16 w-40 h-40 bg-gradient-to-br ${colorMap[topic.color]} blur-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />

                <div className="relative z-10 flex flex-col h-full">
                  <div className="flex justify-between items-start mb-8">
                    <div className={`p-4 rounded-2xl transition-colors duration-300 ${darkMode ? "bg-zinc-800 text-indigo-400 group-hover:bg-indigo-500 group-hover:text-white" : "bg-slate-100 text-slate-700 group-hover:bg-indigo-600 group-hover:text-white"}`}>
                      {React.cloneElement(topic.icon, { size: 24 })}
                    </div>
                    <span className={`text-[10px] font-bold uppercase tracking-wider px-3 py-1 rounded-full ${darkMode ? "bg-zinc-800 text-slate-400" : "bg-slate-100 text-slate-500"}`}>
                      {topic.difficulty}
                    </span>
                  </div>

                  <h3 className={`text-xl font-bold mb-3 transition-colors ${darkMode ? "text-white group-hover:text-indigo-400" : "text-slate-900 group-hover:text-indigo-600"}`}>
                    {topic.title}
                  </h3>
                  <p className={`text-sm leading-relaxed mb-8 ${darkMode ? "text-slate-400" : "text-slate-500"}`}>
                    {topic.desc}
                  </p>

                  <div className={`mt-auto flex items-center justify-between pt-6 border-t ${darkMode ? "border-zinc-800" : "border-slate-100"}`}>
                    <span className={`flex items-center text-xs font-medium ${darkMode ? "text-slate-500" : "text-slate-400"}`}>
                      <Circle className="w-2 h-2 mr-2 fill-cyan-500 stroke-none" />
                      {topic.stats}
                    </span>
                    <div className={`flex items-center text-sm font-bold transition-all ${darkMode ? "text-white group-hover:text-indigo-400" : "text-slate-900 group-hover:text-indigo-600"} group-hover:translate-x-1`}>
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

export default STL;