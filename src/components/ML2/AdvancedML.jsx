import React from "react";
import { NavLink } from "react-router-dom";
import { useTheme } from "../../ThemeContext.jsx";
import { 
  Trees, 
  BrainCircuit, 
  LineChart, 
  BookOpen, 
  PlayCircle, 
  ArrowRight, 
  ArrowLeft,
  Circle 
} from "lucide-react";

const AdvancedML = () => {
  const { darkMode } = useTheme();

  const topics = [
    {
      title: "Ensemble Learning",
      path: "/EnsembleLearning",
      desc: "Bagging, Boosting, and Stacking. Master Random Forests and XGBoost logic.",
      color: "emerald",
      difficulty: "Advanced",
      icon: <Trees />,
      stats: "15 Lessons"
    },
    {
      title: "Neural Networks",
      path: "/NeuralNetworks",
      desc: "Deep Learning basics: Backpropagation, Activation functions, and CNN/RNN architectures.",
      color: "amber",
      difficulty: "Expert",
      icon: <BrainCircuit />,
      stats: "20 Lessons"
    },
    {
      title: "Time Series",
      path: "/TimeSeriesForecasting",
      desc: "Predicting the future: ARIMA models, LSTMs, and handling seasonal data.",
      color: "rose",
      difficulty: "Advanced",
      icon: <LineChart />,
      stats: "10 Patterns"
    }
  ];

  const colorMap = {
    emerald: "from-emerald-500/20 to-teal-500/20 border-emerald-500/20 text-emerald-400",
    amber: "from-amber-500/20 to-orange-500/20 border-amber-500/20 text-amber-400",
    rose: "from-rose-500/20 to-pink-500/20 border-rose-500/20 text-rose-400",
  };

  return (
    <div className={`min-h-screen py-16 px-6 transition-colors duration-500 ${darkMode ? "bg-[#0b0f1a]" : "bg-slate-50"}`}>
      <div className="max-w-6xl mx-auto">
        
        {/* Breadcrumb Navigation */}
        <nav className="flex items-center space-x-2 mb-8 text-sm font-medium">
          <NavLink to="/" className={`${darkMode ? "text-slate-500 hover:text-white" : "text-slate-400 hover:text-slate-900"} transition-colors`}>Home</NavLink>
          <span className="text-slate-500">/</span>
          <span className="text-indigo-500 font-bold">Advanced ML</span>
        </nav>

        {/* Header Section */}
        <div className="mb-16">
          <h1 className={`text-5xl font-black tracking-tight mb-6 ${darkMode ? "text-white" : "text-slate-900"}`}>
            Advanced <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-500 to-purple-500">Machine Learning.</span>
          </h1>
          <p className={`text-xl max-w-3xl leading-relaxed ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
            Master the state-of-the-art algorithms that drive modern computer vision, 
            natural language processing, and predictive analytics.
          </p>
        </div>

        {/* Bento Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
          {topics.map((topic, index) => (
            <NavLink to={topic.path} key={index} className="group relative">
              <div className={`
                relative overflow-hidden rounded-[2.5rem] border p-8 h-full flex flex-col transition-all duration-500
                ${darkMode 
                  ? "bg-zinc-900/40 border-zinc-800/50 backdrop-blur-xl hover:bg-zinc-800/60 hover:border-indigo-500/50 shadow-2xl shadow-black/50" 
                  : "bg-white border-slate-200 hover:border-indigo-400 hover:shadow-xl hover:shadow-indigo-500/10"}
                group-hover:-translate-y-2
              `}>
                
                {/* Decorative Glow */}
                <div className={`absolute top-0 right-0 -mr-16 -mt-16 w-48 h-48 bg-gradient-to-br ${colorMap[topic.color]} blur-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />

                <div className="relative z-10 flex flex-col h-full">
                  <div className="flex justify-between items-start mb-8">
                    <div className={`p-4 rounded-2xl transition-colors duration-300 ${darkMode ? "bg-zinc-800 text-indigo-400 group-hover:bg-indigo-500 group-hover:text-white" : "bg-slate-100 text-slate-700 group-hover:bg-indigo-600 group-hover:text-white"}`}>
                      {React.cloneElement(topic.icon, { size: 28 })}
                    </div>
                    <span className={`text-[10px] font-bold uppercase tracking-wider px-3 py-1 rounded-full ${darkMode ? "bg-zinc-800 text-slate-400" : "bg-slate-100 text-slate-500"}`}>
                      {topic.difficulty}
                    </span>
                  </div>

                  <h3 className={`text-2xl font-bold mb-3 transition-colors ${darkMode ? "text-white group-hover:text-indigo-400" : "text-slate-900 group-hover:text-indigo-600"}`}>
                    {topic.title}
                  </h3>
                  <p className={`text-sm leading-relaxed mb-8 ${darkMode ? "text-slate-400" : "text-slate-500"}`}>
                    {topic.desc}
                  </p>

                  <div className={`mt-auto flex items-center justify-between pt-6 border-t ${darkMode ? "border-zinc-800" : "border-slate-100"}`}>
                    <span className={`flex items-center text-xs font-medium ${darkMode ? "text-slate-500" : "text-slate-400"}`}>
                      <Circle className="w-2 h-2 mr-2 fill-indigo-500 stroke-none" />
                      {topic.stats}
                    </span>
                    <div className={`flex items-center text-sm font-bold transition-all ${darkMode ? "text-white group-hover:text-indigo-400" : "text-slate-900 group-hover:text-indigo-600"} group-hover:translate-x-1`}>
                      Deep Dive <ArrowRight className="ml-2 w-4 h-4" />
                    </div>
                  </div>
                </div>
              </div>
            </NavLink>
          ))}
        </div>

        {/* Resource Section */}
        <div className={`rounded-[2.5rem] p-10 border ${darkMode ? 'bg-zinc-900/40 border-zinc-800' : 'bg-white border-slate-200 shadow-sm'} mb-16`}>
          <h2 className={`text-2xl font-bold mb-8 ${darkMode ? 'text-white' : 'text-slate-900'}`}>Learning Resources</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className={`p-6 rounded-2xl flex items-center space-x-4 transition-colors ${darkMode ? 'bg-zinc-800/50 hover:bg-zinc-800' : 'bg-slate-50 hover:bg-slate-100'}`}>
              <div className="bg-blue-500 p-3 rounded-xl text-white"><BookOpen size={24} /></div>
              <div>
                <h4 className={`font-bold ${darkMode ? 'text-white' : 'text-slate-900'}`}>Advanced Theory</h4>
                <p className="text-sm text-slate-500">Curated papers and specialized ML textbooks.</p>
              </div>
            </div>
            <div className={`p-6 rounded-2xl flex items-center space-x-4 transition-colors ${darkMode ? 'bg-zinc-800/50 hover:bg-zinc-800' : 'bg-slate-50 hover:bg-slate-100'}`}>
              <div className="bg-purple-500 p-3 rounded-xl text-white"><PlayCircle size={24} /></div>
              <div>
                <h4 className={`font-bold ${darkMode ? 'text-white' : 'text-slate-900'}`}>Architecting Nets</h4>
                <p className="text-sm text-slate-500">Visual guides to deep learning architectures.</p>
              </div>
            </div>
          </div>
        </div>

        {/* Global Navigation */}
        <div className="flex justify-between items-center">
          <NavLink to="/DataPreprocessing" className={`flex items-center px-6 py-3 rounded-2xl font-bold transition-all ${darkMode ? 'bg-zinc-800 text-white hover:bg-zinc-700' : 'bg-slate-200 text-slate-700 hover:bg-slate-300'}`}>
            <ArrowLeft className="mr-2" size={18} /> Preprocessing
          </NavLink>
          <NavLink to="/ReinforcementLearning" className="flex items-center px-8 py-4 rounded-2xl font-bold bg-indigo-600 text-white hover:bg-indigo-500 shadow-xl shadow-indigo-600/20 transition-all hover:-translate-y-1">
            Reinforcement Learning <ArrowRight className="ml-2" size={18} />
          </NavLink>
        </div>
      </div>
    </div>
  );
};

export default AdvancedML;