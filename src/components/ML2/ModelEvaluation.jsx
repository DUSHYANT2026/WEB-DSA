import React from "react";
import { NavLink } from "react-router-dom";
import { useTheme } from "../../ThemeContext.jsx";
import { 
  BarChart3, 
  Settings2, 
  ShieldCheck, 
  Scale, 
  ArrowRight, 
  Circle, 
  Target 
} from "lucide-react";

export const ModelEvaluation = () => {
  const { darkMode } = useTheme();

  const topics = [
    {
      title: "Evaluation Metrics",
      desc: "Precision, Recall, F1-Score, and ROC-AUC curves for classification and regression.",
      path: "/EvaluationMetrics",
      color: "emerald",
      difficulty: "Intermediate",
      icon: <BarChart3 />,
      stats: "10 Metrics"
    },
    {
      title: "Hyperparameter Tuning",
      desc: "Grid Search, Random Search, and Bayesian optimization to find the best model parameters.",
      path: "/HyperparameterTuning",
      color: "amber",
      difficulty: "Advanced",
      icon: <Settings2 />,
      stats: "4 Methods"
    },
    {
      title: "Regularization Basics",
      desc: "L1 (Lasso) and L2 (Ridge) techniques to prevent overfitting and improve generalization.",
      path: "/RegularizationBasics",
      color: "rose",
      difficulty: "Intermediate",
      icon: <ShieldCheck />,
      stats: "5 Lessons"
    },
    {
      title: "Bias-Variance & CV",
      desc: "Mastering the tradeoff between underfitting and overfitting using Cross-Validation.",
      path: "/BiasVarianceTradeoff",
      color: "blue",
      difficulty: "Advanced",
      icon: <Scale />,
      stats: "8 Concepts"
    }
  ];

  const colorMap = {
    emerald: "from-emerald-500/20 to-teal-500/20 border-emerald-500/20 text-emerald-400",
    amber: "from-amber-500/20 to-orange-500/20 border-amber-500/20 text-amber-400",
    rose: "from-rose-500/20 to-pink-500/20 border-rose-500/20 text-rose-400",
    blue: "from-blue-500/20 to-indigo-500/20 border-blue-500/20 text-blue-400",
  };

  return (
    <div className={`min-h-screen py-16 px-6 transition-colors duration-500 ${darkMode ? "bg-[#0b0f1a]" : "bg-slate-50"}`}>
      <div className="max-w-6xl mx-auto">
        
        {/* Header Section */}
        <div className="flex flex-col items-center mb-16 text-center">
          <span className={`px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-widest mb-4 ${darkMode ? "bg-indigo-500/10 text-indigo-400" : "bg-indigo-100 text-indigo-600"}`}>
            Performance Tuning
          </span>
          <h1 className={`text-5xl font-black tracking-tight mb-4 ${darkMode ? "text-white" : "text-slate-900"}`}>
            Model <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-indigo-600">Evaluation.</span>
          </h1>
          <p className={`text-lg max-w-xl ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
            Building a model is only half the battle. Learn to measure, validate, and optimize your algorithms for real-world reliability.
          </p>
        </div>

        {/* Bento Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {topics.map((topic, index) => (
            <NavLink 
              to={topic.path} 
              key={index} 
              className={`group relative flex flex-col ${index === 3 ? "md:col-span-2 lg:col-span-2" : "lg:col-span-2"}`}
            >
              <div className={`
                relative overflow-hidden rounded-[2.5rem] border p-8 h-full flex flex-col transition-all duration-500
                ${darkMode 
                  ? "bg-zinc-900/40 border-zinc-800/50 backdrop-blur-xl hover:bg-zinc-800/60 hover:border-indigo-500/50 shadow-2xl shadow-black/50" 
                  : "bg-white border-slate-200 hover:border-indigo-400 hover:shadow-xl hover:shadow-indigo-500/10"}
                group-hover:-translate-y-2
              `}>
                
                {/* Visual Glow */}
                <div className={`absolute top-0 right-0 -mr-16 -mt-16 w-48 h-48 bg-gradient-to-br ${colorMap[topic.color]} blur-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />

                <div className="relative z-10 flex flex-col h-full">
                  <div className="flex justify-between items-start mb-8">
                    <div className={`p-4 rounded-2xl transition-colors duration-300 ${darkMode ? "bg-zinc-800 text-indigo-400 group-hover:bg-indigo-500 group-hover:text-white" : "bg-slate-100 text-slate-700 group-hover:bg-indigo-600 group-hover:text-white"}`}>
                      {React.cloneElement(topic.icon, { size: 24 })}
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
                      Optimize <ArrowRight className="ml-2 w-4 h-4" />
                    </div>
                  </div>
                </div>
              </div>
            </NavLink>
          ))}
        </div>

        {/* Informational Callout */}
        <div className={`mt-16 p-8 rounded-[2.5rem] border ${darkMode ? 'bg-indigo-500/5 border-indigo-500/20' : 'bg-indigo-50 border-indigo-100'} flex flex-col md:flex-row items-center gap-6`}>
           <div className={`p-4 rounded-full ${darkMode ? 'bg-indigo-500/20 text-indigo-400' : 'bg-white text-indigo-600 shadow-sm'}`}>
              <Target size={32} />
           </div>
           <div>
              <h4 className={`text-lg font-bold ${darkMode ? 'text-white' : 'text-slate-900'}`}>The Goal of Optimization</h4>
              <p className={`text-sm ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                A model with 99% training accuracy but 50% test accuracy is useless. We focus on <strong>Generalization</strong>—ensuring your model performs on data it has never seen before.
              </p>
           </div>
        </div>
      </div>
    </div>
  );
};

export default ModelEvaluation;