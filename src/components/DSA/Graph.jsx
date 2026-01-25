import React from "react";
import { NavLink } from "react-router-dom";
import { useTheme } from "../../ThemeContext.jsx";
import { 
  Network, 
  Grid3X3, 
  Share2, 
  Route, 
  Split, 
  Trophy, 
  Flame, 
  ArrowRight, 
  Circle 
} from "lucide-react";

function Graph() {
  const { darkMode } = useTheme();

  const topics = [
    {
      title: "BFS & DFS Core",
      desc: "Breadth-First and Depth-First search fundamentals on adjacency lists.",
      path: "/graph1",
      color: "blue",
      difficulty: "Intermediate",
      icon: <Network />,
      stats: "15 Lessons"
    },
    {
      title: "Matrix & Grid Traversal",
      desc: "Flood fill, island counting, and searching in 2D coordinate spaces.",
      path: "/graph2",
      color: "emerald",
      difficulty: "Intermediate",
      icon: <Grid3X3 />,
      stats: "10 Patterns"
    },
    {
      title: "MST & Disjoint Sets",
      desc: "Kruskal’s, Prim’s, and Union-Find with path compression logic.",
      path: "/graph3",
      color: "amber",
      difficulty: "Advanced",
      icon: <Share2 />,
      stats: "8 Algorithms"
    },
    {
      title: "Shortest Path Mastery",
      desc: "Dijkstra, Bellman-Ford, and Floyd-Warshall for weighted graphs.",
      path: "/graph4",
      color: "rose",
      difficulty: "Advanced",
      icon: <Route />,
      stats: "12 Lessons"
    },
    {
      title: "Strong Connectivity",
      desc: "Kosaraju's algorithm, Bridges, and finding Articulation Points.",
      path: "/graph5",
      color: "indigo",
      difficulty: "Expert",
      icon: <Split />,
      stats: "6 Concepts"
    },
    {
      title: "Leetcode Top Graphs",
      desc: "The most vital graph problems curated for technical interviews.",
      path: "/graph6",
      color: "violet",
      difficulty: "Advanced",
      icon: <Trophy />,
      stats: "60 Questions"
    },
    {
      title: "MAANG Tier Challenges",
      desc: "Hardest graph-theory problems asked by Google, Meta, and Amazon.",
      path: "/graph7",
      color: "slate",
      difficulty: "Expert",
      icon: <Flame />,
      stats: "25 Challenges"
    }
  ];

  const colorMap = {
    blue: "from-blue-500/20 to-indigo-500/20 border-blue-500/20 text-blue-400",
    emerald: "from-emerald-500/20 to-teal-500/20 border-emerald-500/20 text-emerald-400",
    amber: "from-amber-500/20 to-orange-500/20 border-amber-500/20 text-amber-400",
    rose: "from-rose-500/20 to-pink-500/20 border-rose-500/20 text-rose-400",
    indigo: "from-indigo-500/20 to-purple-500/20 border-indigo-500/20 text-indigo-400",
    violet: "from-violet-500/20 to-fuchsia-500/20 border-violet-500/20 text-violet-400",
    slate: "from-slate-600/20 to-slate-800/20 border-slate-500/20 text-slate-400",
  };

  return (
    <div className={`min-h-screen py-16 px-6 transition-colors duration-500 ${darkMode ? "bg-[#0b0f1a]" : "bg-slate-50"}`}>
      <div className="max-w-6xl mx-auto">
        
        {/* Header Section */}
        <div className="flex flex-col items-center mb-16 text-center">
          <span className={`px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-widest mb-4 ${darkMode ? "bg-violet-500/10 text-violet-400" : "bg-violet-100 text-violet-600"}`}>
            Network & Connectivity
          </span>
          <h1 className={`text-5xl font-black tracking-tight mb-4 ${darkMode ? "text-white" : "text-slate-900"}`}>
            Graph <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-500 to-indigo-600">Algorithms.</span>
          </h1>
          <p className={`text-lg max-w-xl ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
            Master the traversal and optimization of complex networks, from social grids to shortest path routing.
          </p>
        </div>

        {/* Bento Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {topics.map((topic, index) => (
            <NavLink 
              to={topic.path} 
              key={index} 
              className={`group relative ${index === 6 ? "lg:col-span-3" : ""}`}
            >
              <div className={`
                relative overflow-hidden rounded-[2.5rem] border p-8 h-full flex flex-col transition-all duration-500
                ${darkMode 
                  ? "bg-zinc-900/40 border-zinc-800/50 backdrop-blur-xl hover:bg-zinc-800/60 hover:border-violet-500/50 shadow-2xl shadow-black/50" 
                  : "bg-white border-slate-200 hover:border-violet-400 hover:shadow-xl hover:shadow-violet-500/10"}
                group-hover:-translate-y-2
              `}>
                
                {/* Visual Glow */}
                <div className={`absolute top-0 right-0 -mr-16 -mt-16 w-40 h-40 bg-gradient-to-br ${colorMap[topic.color]} blur-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />

                <div className="relative z-10 flex flex-col h-full">
                  <div className="flex justify-between items-start mb-8">
                    <div className={`p-4 rounded-2xl transition-colors duration-300 ${darkMode ? "bg-zinc-800 text-violet-400 group-hover:bg-violet-500 group-hover:text-white" : "bg-slate-100 text-slate-700 group-hover:bg-violet-600 group-hover:text-white"}`}>
                      {React.cloneElement(topic.icon, { size: 24 })}
                    </div>
                    <span className={`text-[10px] font-bold uppercase tracking-wider px-3 py-1 rounded-full ${darkMode ? "bg-zinc-800 text-slate-400" : "bg-slate-100 text-slate-500"}`}>
                      {topic.difficulty}
                    </span>
                  </div>

                  <h3 className={`text-xl font-bold mb-3 transition-colors ${darkMode ? "text-white group-hover:text-violet-400" : "text-slate-900 group-hover:text-violet-600"}`}>
                    {topic.title}
                  </h3>
                  <p className={`text-sm leading-relaxed mb-8 ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
                    {topic.desc}
                  </p>

                  <div className={`mt-auto flex items-center justify-between pt-6 border-t ${darkMode ? "border-zinc-800" : "border-slate-100"}`}>
                    <span className={`flex items-center text-xs font-medium ${darkMode ? "text-slate-500" : "text-slate-400"}`}>
                      <Circle className="w-2 h-2 mr-2 fill-indigo-500 stroke-none" />
                      {topic.stats}
                    </span>
                    <div className={`flex items-center text-sm font-bold transition-all ${darkMode ? "text-white group-hover:text-violet-400" : "text-slate-900 group-hover:text-violet-600"} group-hover:translate-x-1`}>
                      Explore Module <ArrowRight className="ml-2 w-4 h-4" />
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

export default Graph;