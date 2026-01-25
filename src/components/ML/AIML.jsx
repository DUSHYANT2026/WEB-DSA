import React from "react";
import { NavLink } from "react-router-dom";
import { useTheme } from "../../ThemeContext.jsx";

// Specialized icons for AI/ML domains
const MLIcon = ({ title }) => {
  const iconProps = { className: "h-6 w-6 text-cyan-400 group-hover:scale-110 transition-transform duration-300" };
  if (title.includes("Foundations")) return <svg {...iconProps} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" /></svg>;
  if (title.includes("Introduction")) return <svg {...iconProps} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>;
  if (title.includes("Preprocessing")) return <svg {...iconProps} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.183.319l-3.08 1.913A2 2 0 001 19.105V20a2 2 0 002 2h18a2 2 0 002-2v-.895a2 2 0 00-1.104-1.789l-2.468-1.234z" /></svg>;
  if (title.includes("Supervised")) return <svg {...iconProps} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" /></svg>;
  if (title.includes("Unsupervised")) return <svg {...iconProps} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" /></svg>;
  if (title.includes("Reinforcement")) return <svg {...iconProps} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>;
  if (title.includes("Evaluation")) return <svg {...iconProps} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>;
  if (title.includes("Deployment")) return <svg {...iconProps} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>;
  return <svg {...iconProps} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>;
};

export const AIML = () => {
  const { darkMode } = useTheme();

  const roadmapItems = [
    { title: "Foundations of Machine Learning", description: "Linear Algebra, Probability & Statistics, Calculus, Python basics.", path: "/Foundation" },
    { title: "Introduction to Machine Learning", description: "History of ML, AI/ML/Deep Learning Differences, ML Pipeline.", path: "/MachineLearning" },
    { title: "Data Preprocessing", description: "Data Cleaning, Data Transformation, Feature Engineering, Data Splitting.", path: "/DataPreprocessing" },
    { title: "Supervised Learning", description: "Regression, Classification, Decision Trees, Support Vector Machines.", path: "/SupervisedLearning" },
    { title: "Unsupervised Learning", description: "Clustering, Dimensionality Reduction, Anomaly Detection.", path: "/UnsupervisedLearning" },
    { title: "Advanced ML Algorithms", description: "Ensemble Learning, Neural Networks, Time Series Forecasting.", path: "/AdvancedML" },
    { title: "Reinforcement Learning", description: "Basics of RL, MDP, Q-Learning, Policy Gradient Methods.", path: "/ReinforcementLearning" },
    { title: "Model Evaluation", description: "Evaluation Metrics, Hyperparameter Tuning, Regularization.", path: "/ModelEvaluation" },
    { title: "Deployment Basics", description: "Model saving/loading and simple Flask API deployment.", path: "/BasicDeployment" }
  ];

  return (
    <div className={`min-h-screen py-16 px-6 lg:px-8 transition-colors duration-500 ${darkMode ? "bg-[#0b0f1a]" : "bg-slate-50"}`}>
      <div className="max-w-6xl mx-auto">
        
        {/* Header Section */}
        <div className="mb-20 text-center">
          <span className="px-4 py-1 rounded-full text-xs font-bold tracking-widest uppercase bg-cyan-500/10 text-cyan-500 mb-4 inline-block">
            Step-by-Step Curriculum
          </span>
          <h1 className={`text-4xl md:text-5xl font-black mb-4 tracking-tight ${darkMode ? "text-white" : "text-slate-900"}`}>
            AI/ML <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-600">Learning Roadmap</span>
          </h1>
          <p className={`text-lg max-w-2xl mx-auto ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
            Master the mathematics and engineering behind artificial intelligence with our structured sequence.
          </p>
        </div>

        {/* Learning Pipeline (Cards Grid) */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 relative">
          {/* Subtle connecting line background for desktop */}
          <div className="hidden lg:block absolute top-1/2 left-0 w-full h-px bg-gradient-to-r from-transparent via-cyan-500/10 to-transparent -z-10" />

          {roadmapItems.map((item, index) => (
            <NavLink to={item.path} className="group" key={index}>
              <div className={`
                relative p-8 rounded-[2rem] border h-full flex flex-col transition-all duration-500
                ${darkMode 
                  ? "bg-zinc-900/40 border-zinc-800/50 backdrop-blur-xl hover:bg-zinc-800/60 hover:border-cyan-500/50 shadow-2xl shadow-black/50" 
                  : "bg-white border-slate-200 hover:border-cyan-400 hover:shadow-xl hover:shadow-cyan-500/10"}
                group-hover:-translate-y-2
              `}>
                {/* Step Number Badge */}
                <div className="absolute top-6 right-8 text-4xl font-black opacity-5 group-hover:opacity-10 transition-opacity">
                  {index + 1}
                </div>

                <div className={`w-12 h-12 rounded-2xl flex items-center justify-center mb-6 ${darkMode ? "bg-zinc-800" : "bg-slate-100"}`}>
                  <MLIcon title={item.title} />
                </div>

                <h3 className={`text-xl font-bold mb-3 tracking-tight ${darkMode ? "text-white" : "text-slate-900"}`}>
                  {item.title}
                </h3>
                
                <p className={`text-sm leading-relaxed mb-6 ${darkMode ? "text-slate-400" : "text-slate-500"}`}>
                  {item.description}
                </p>

                <div className="mt-auto flex items-center text-sm font-bold text-cyan-500">
                  <span className="group-hover:mr-2 transition-all">Explore Module</span>
                  <svg className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-all transform group-hover:translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </div>
              </div>
            </NavLink>
          ))}
        </div>

        {/* Footer CTA */}
        <div className="mt-20 text-center">
          <div className="inline-flex flex-col items-center">
            <p className={`text-sm font-medium mb-6 ${darkMode ? "text-slate-500" : "text-slate-400"}`}>
              Ready to start with the basics?
            </p>
            <NavLink to="/Foundation" className="group relative px-10 py-4 bg-cyan-600 text-white rounded-2xl font-bold text-lg shadow-xl shadow-cyan-600/20 hover:bg-cyan-500 transition-all active:scale-95">
              Begin Your Journey
              <div className="absolute inset-0 rounded-2xl bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity" />
            </NavLink>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIML;