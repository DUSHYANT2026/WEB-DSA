import React, { useState, useEffect } from "react";
import { NavLink } from "react-router-dom";
import { useTheme } from "../../ThemeContext.jsx";
import { ArrowLeft, ArrowRight, BookOpen, ChevronRight, Circle } from "lucide-react";

const Foundation = () => {
  const { darkMode } = useTheme();

  const topics = [
    {
      title: "Linear Algebra",
      path: "/LinearAlgebra",
      description: "Vectors, matrices, eigenvalues, and spatial transformations.",
      color: "from-blue-600 to-indigo-600",
      icon: "∑",
      step: "01"
    },
    {
      title: "Prob & Stats",
      path: "/Probability",
      description: "Distributions, hypothesis testing, and statistical inference.",
      color: "from-emerald-600 to-teal-600",
      icon: "σ",
      step: "02"
    },
    {
      title: "Multivariate Calculus",
      path: "/Calculus",
      description: "Partial derivatives, chain rule, and gradient descent logic.",
      color: "from-amber-500 to-orange-600",
      icon: "∫",
      step: "03"
    },
    {
      title: "Python for ML",
      path: "/Python",
      description: "Data manipulation with NumPy, Pandas, and SciKit-Learn.",
      color: "from-rose-500 to-pink-600",
      icon: "Py",
      step: "04"
    }
  ];

  return (
    <div className={`min-h-screen py-16 px-6 transition-colors duration-500 ${darkMode ? "bg-[#0b0f1a]" : "bg-slate-50"}`}>
      <div className="max-w-6xl mx-auto">
        
        {/* Breadcrumb & Meta */}
        <nav className="flex items-center space-x-2 mb-8 text-sm font-medium">
          <NavLink to="/" className={`${darkMode ? "text-slate-500 hover:text-white" : "text-slate-400 hover:text-slate-900"} transition-colors`}>Home</NavLink>
          <ChevronRight size={14} className="text-slate-500" />
          <span className="text-orange-500">Foundations</span>
        </nav>

        {/* Header Section */}
        <div className="mb-16">
          <div className="flex items-center space-x-3 mb-4">
            <div className="h-px w-12 bg-orange-500"></div>
            <span className="text-orange-500 uppercase tracking-widest font-bold text-xs">Phase 01</span>
          </div>
          <h1 className={`text-5xl font-black tracking-tight mb-6 ${darkMode ? "text-white" : "text-slate-900"}`}>
            The Mathematical <span className="text-transparent bg-clip-text bg-gradient-to-r from-orange-400 to-red-500">Bedrock.</span>
          </h1>
          <p className={`text-xl max-w-2xl leading-relaxed ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
            Before building models, you must understand the language they speak. 
            These modules cover the essential math required for modern AI engineering.
          </p>
        </div>

        {/* Bento Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-20">
          {topics.map((topic, index) => (
            <NavLink to={topic.path} key={index} className="group relative">
              <div className={`
                relative overflow-hidden rounded-[2.5rem] border p-10 h-full flex flex-col transition-all duration-500
                ${darkMode 
                  ? "bg-zinc-900/40 border-zinc-800/50 backdrop-blur-xl hover:bg-zinc-800/60 hover:border-orange-500/50 shadow-2xl shadow-black/50" 
                  : "bg-white border-slate-200 hover:border-orange-400 hover:shadow-xl hover:shadow-orange-500/10"}
                group-hover:-translate-y-2
              `}>
                
                {/* Step Indicator */}
                <div className="absolute top-8 right-10 text-5xl font-black opacity-5 group-hover:opacity-10 transition-opacity italic">
                  {topic.step}
                </div>

                <div className="relative z-10 flex flex-col h-full">
                  <div className="flex justify-between items-start mb-10">
                    <div className={`w-16 h-16 rounded-2xl flex items-center justify-center text-3xl font-serif italic shadow-lg transition-transform duration-500 group-hover:scale-110 group-hover:rotate-3 ${topic.color} text-white`}>
                      {topic.icon}
                    </div>
                  </div>

                  <h3 className={`text-2xl font-bold mb-4 ${darkMode ? "text-white" : "text-slate-900"}`}>
                    {topic.title}
                  </h3>
                  <p className={`text-base leading-relaxed mb-8 ${darkMode ? "text-slate-400" : "text-slate-600"}`}>
                    {topic.description}
                  </p>

                  <div className={`mt-auto flex items-center justify-between pt-8 border-t ${darkMode ? "border-zinc-800" : "border-slate-100"}`}>
                    <span className="flex items-center text-xs font-bold uppercase tracking-widest text-orange-500">
                      <Circle size={8} className="mr-2 fill-orange-500 animate-pulse" />
                      Core Module
                    </span>
                    <div className={`flex items-center text-sm font-bold transition-all ${darkMode ? "text-white group-hover:text-orange-400" : "text-slate-900 group-hover:text-orange-600"} group-hover:translate-x-2`}>
                      Begin <ArrowRight className="ml-2 w-4 h-4" />
                    </div>
                  </div>
                </div>
              </div>
            </NavLink>
          ))}
        </div>

        {/* Global Navigation */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-6 pt-12 border-t border-slate-200/10">
          <NavLink to="/" className={`flex items-center space-x-2 font-bold px-6 py-3 rounded-2xl transition-all ${darkMode ? "bg-zinc-800 text-white hover:bg-zinc-700" : "bg-slate-200 text-slate-900 hover:bg-slate-300"}`}>
            <ArrowLeft size={18} />
            <span>Curriculum Home</span>
          </NavLink>

          <NavLink to="/MachineLearning" className="flex items-center space-x-2 font-bold px-8 py-4 rounded-2xl bg-orange-600 text-white hover:bg-orange-500 shadow-xl shadow-orange-600/20 transition-all hover:-translate-y-1">
            <span>Next: Intro to ML</span>
            <ArrowRight size={18} />
          </NavLink>
        </div>
      </div>
    </div>
  );
};

export default Foundation;