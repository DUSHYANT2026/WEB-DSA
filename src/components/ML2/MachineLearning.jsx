import React from "react";
import { NavLink } from "react-router-dom";
import { useTheme } from "../../ThemeContext.jsx";
import { 
  History, 
  Layers, 
  GitMerge, 
  ChevronRight, 
  ArrowRight, 
  Lightbulb, 
  Code, 
  Database, 
  Cpu 
} from "lucide-react";

const MachineLearning = () => {
  const { darkMode } = useTheme();

  const topics = [
    {
      title: "History of ML",
      description: "From the Turing Test to GPT-4: the evolution of algorithmic intelligence.",
      icon: <History size={28} />,
      gradient: "from-blue-600 to-indigo-600",
      path: "/HistoryML",
      tag: "Evolution"
    },
    {
      title: "AI vs ML vs DL",
      description: "Mapping the hierarchy of Artificial Intelligence, Machine Learning, and Deep Learning.",
      icon: <Layers size={28} />,
      gradient: "from-emerald-600 to-teal-600",
      path: "/AIvsMLvsDL",
      tag: "Hierarchy"
    },
    {
      title: "ML Pipeline",
      description: "The lifecycle: from data collection and cleaning to model deployment.",
      icon: <GitMerge size={28} />,
      gradient: "from-amber-500 to-orange-600",
      path: "/MLPipeline",
      tag: "Process"
    },
  ];

  return (
    <div className={`min-h-screen py-20 px-6 transition-colors duration-500 ${darkMode ? "bg-[#0b0f1a]" : "bg-slate-50"}`}>
      <div className="max-w-6xl mx-auto">
        
        {/* Header Section */}
        <div className="text-center mb-20">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-pink-500/10 text-pink-500 text-xs font-bold uppercase tracking-widest mb-6">
            <Lightbulb size={14} />
            <span>Phase 02: Core Concepts</span>
          </div>
          <h1 className={`text-5xl md:text-6xl font-black mb-6 tracking-tight ${darkMode ? "text-white" : "text-slate-900"}`}>
            Introduction to <span className="text-transparent bg-clip-text bg-gradient-to-r from-orange-500 via-pink-500 to-purple-600">Machine Learning.</span>
          </h1>
          <p className={`text-lg md:text-xl max-w-3xl mx-auto leading-relaxed ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
            Discover how data-driven algorithms are redefining the limits of software engineering and modern technology.
          </p>
        </div>

        {/* Learning Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-24">
          {topics.map((topic, index) => (
            <NavLink key={index} to={topic.path} className="group relative">
              <div className={`
                relative h-full overflow-hidden rounded-[2.5rem] border p-8 flex flex-col transition-all duration-500
                ${darkMode 
                  ? "bg-zinc-900/40 border-zinc-800/50 backdrop-blur-xl hover:bg-zinc-800/60 hover:border-pink-500/50 shadow-2xl shadow-black/50" 
                  : "bg-white border-slate-200 hover:border-pink-400 hover:shadow-xl hover:shadow-pink-500/10"}
                group-hover:-translate-y-2
              `}>
                <div className={`inline-flex p-4 rounded-2xl mb-6 bg-gradient-to-br ${topic.gradient} text-white shadow-lg shadow-indigo-500/20`}>
                  {topic.icon}
                </div>
                
                <span className="text-[10px] font-bold uppercase tracking-tighter text-pink-500 mb-2">{topic.tag}</span>
                <h3 className={`text-2xl font-bold mb-3 ${darkMode ? "text-white" : "text-slate-900"}`}>{topic.title}</h3>
                <p className={`text-sm leading-relaxed mb-8 flex-grow ${darkMode ? "text-slate-400" : "text-slate-500"}`}>{topic.description}</p>
                
                <div className={`flex items-center text-sm font-bold transition-all ${darkMode ? "text-white group-hover:text-pink-400" : "text-slate-900 group-hover:text-pink-600"} group-hover:translate-x-1`}>
                  Explore Module <ArrowRight size={16} className="ml-2" />
                </div>
              </div>
            </NavLink>
          ))}
        </div>

        {/* Feature Bento Section */}
        <div className={`rounded-[3rem] p-1 shadow-2xl bg-gradient-to-br from-gray-800 to-black`}>
          <div className={`rounded-[2.9rem] p-8 md:p-12 ${darkMode ? "bg-[#0b0f1a]/80" : "bg-white/10"} backdrop-blur-md`}>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
              <div>
                <h2 className="text-4xl font-black text-white mb-6">Ready to build?</h2>
                <p className="text-slate-300 text-lg mb-8 leading-relaxed">
                  Go beyond theory. Our platform provides the compute, the data, and the frameworks to help you deploy your first neural network.
                </p>
                <button className="group px-8 py-4 bg-white text-black font-bold rounded-2xl hover:bg-pink-500 hover:text-white transition-all flex items-center shadow-xl shadow-white/5 hover:shadow-pink-500/20">
                  Start Learning Path <ChevronRight size={18} className="ml-2 group-hover:translate-x-1 transition-transform" />
                </button>
              </div>

              <div className="grid grid-cols-2 gap-4">
                {[
                  { label: "Lessons", val: "50+", icon: <Code className="text-emerald-400" /> },
                  { label: "Projects", val: "20+", icon: <Database className="text-blue-400" /> },
                  { label: "Datasets", val: "12+", icon: <Cpu className="text-purple-400" /> },
                  { label: "Frameworks", val: "5+", icon: <Lightbulb className="text-amber-400" /> },
                ].map((stat, i) => (
                  <div key={i} className="bg-white/5 border border-white/10 p-6 rounded-3xl backdrop-blur-sm hover:bg-white/10 transition-colors">
                    <div className="mb-2">{stat.icon}</div>
                    <div className="text-3xl font-black text-white">{stat.val}</div>
                    <div className="text-xs font-bold text-slate-500 uppercase tracking-widest">{stat.label}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MachineLearning;