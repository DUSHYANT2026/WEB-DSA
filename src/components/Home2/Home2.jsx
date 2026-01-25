import React from "react";
import { NavLink } from "react-router-dom";
import { useTheme } from "../../ThemeContext.jsx";

// Adding specific icons for better UX recognition
const CategoryIcon = ({ title }) => {
  const iconProps = { className: "h-6 w-6 mb-3 opacity-90 group-hover:scale-110 transition-transform duration-300" };
  if (title.includes("Array")) return <svg {...iconProps} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" /></svg>;
  if (title.includes("STL")) return <svg {...iconProps} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" /></svg>;
  if (title.includes("Linked")) return <svg {...iconProps} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" /></svg>;
  if (title.includes("Stack")) return <svg {...iconProps} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 14v6m-3-3h6M6 10h2a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v2a2 2 0 002 2zm10 0h2a2 2 0 002-2V6a2 2 0 00-2-2h-2a2 2 0 00-2 2v2a2 2 0 002 2zM6 20h2a2 2 0 002-2v-2a2 2 0 00-2-2H6a2 2 0 00-2 2v2a2 2 0 002 2z" /></svg>;
  if (title.includes("Dynamic")) return <svg {...iconProps} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>;
  if (title.includes("Recursion")) return <svg {...iconProps} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>;
  if (title.includes("Trees")) return <svg {...iconProps} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" /></svg>;
  if (title.includes("Graph")) return <svg {...iconProps} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 20l4-16m2 16l4-16M6 9h14M4 15h14" /></svg>;
  return <svg {...iconProps} fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" /></svg>;
};

function Home2() {
  const { darkMode } = useTheme();

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const categories = [
    { path: "/Arrays", title: "Array, String, Matrix", gradient: "from-blue-500 to-indigo-600", shadow: "shadow-blue-500/20" },
    { path: "/STL", title: "Standard Template Library", gradient: "from-emerald-500 to-teal-600", shadow: "shadow-emerald-500/20" },
    { path: "/Linkedlist", title: "Linked List", gradient: "from-amber-500 to-orange-600", shadow: "shadow-amber-500/20" },
    { path: "/Stack", title: "Stack, Queue & Heaps", gradient: "from-rose-500 to-red-600", shadow: "shadow-rose-500/20" },
    { path: "/Recusion", title: "Recursion & Backtracking", gradient: "from-violet-500 to-purple-600", shadow: "shadow-violet-500/20" },
    { path: "/Dynamic", title: "Dynamic Programming", gradient: "from-fuchsia-500 to-pink-600", shadow: "shadow-fuchsia-500/20" },
    { path: "/Tree", title: "Trees (Binary, BST, AVL)", gradient: "from-cyan-500 to-blue-600", shadow: "shadow-cyan-500/20" },
    { path: "/Graph", title: "Graph Algorithms", gradient: "from-orange-500 to-red-600", shadow: "shadow-orange-500/20" },
    { path: "/Bitm", title: "Bit Manipulation & Maths", gradient: "from-slate-600 to-slate-800", shadow: "shadow-slate-500/20" },
    { path: "/Algorithm", title: "All Algorithms", gradient: "from-blue-600 to-cyan-500", shadow: "shadow-blue-500/20" },
    { path: "/Trie", title: "Trie Data Structure", gradient: "from-pink-500 to-rose-500", shadow: "shadow-pink-500/20" },
  ];

  return (
    <div className={`min-h-screen transition-colors duration-500 ${darkMode ? "bg-[#0b0f1a] text-gray-100" : "bg-[#f8fafc] text-gray-900"}`}>
      <div className="mx-auto w-full max-w-6xl px-6 py-12">
        
        {/* Header Section */}
        <header className="relative mb-16 p-10 rounded-3xl overflow-hidden group">
          <div className={`absolute inset-0 transition-opacity duration-500 ${darkMode ? "bg-zinc-900/50" : "bg-indigo-600"}`} />
          
          {/* Animated Background Grids */}
          <div className="absolute inset-0 opacity-20 bg-[radial-gradient(#fff_1px,transparent_1px)] [background-size:20px_20px]" />
          
          <div className="relative z-10">
            <h1 className="text-4xl md:text-5xl font-black tracking-tight text-white mb-4">
              DSA <span className="text-transparent bg-clip-text bg-gradient-to-r from-orange-300 to-rose-300">Topic Vault</span>
            </h1>
            <p className="text-lg md:text-xl font-medium text-indigo-100 max-w-2xl mx-auto opacity-90">
              A comprehensive guide to master data structures and algorithms with interactive problems and theory.
            </p>
          </div>
        </header>

        {/* Categories Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {categories.map((category, index) => (
            <NavLink
              to={category.path}
              onClick={scrollToTop}
              key={index}
              className="group relative"
            >
              {/* Card Glow Background */}
              <div className={`absolute -inset-0.5 bg-gradient-to-r ${category.gradient} rounded-2xl blur opacity-0 group-hover:opacity-30 transition duration-500`} />
              
              <div className={`relative p-6 rounded-2xl border h-full flex flex-col justify-between transition-all duration-300 transform group-hover:-translate-y-2 ${
                  darkMode 
                    ? "bg-zinc-900/80 border-zinc-800/50 backdrop-blur-xl" 
                    : "bg-white border-slate-200 shadow-sm"
                }`}
              >
                <div>
                  <div className={`inline-flex items-center justify-center p-3 rounded-xl mb-4 bg-gradient-to-br ${category.gradient} text-white shadow-lg ${category.shadow}`}>
                    <CategoryIcon title={category.title} />
                  </div>
                  
                  <h3 className={`text-xl font-bold mb-2 tracking-tight ${darkMode ? "text-white" : "text-slate-800"}`}>
                    {category.title}
                  </h3>
                  <p className={`text-sm leading-relaxed ${darkMode ? "text-gray-400" : "text-slate-500"}`}>
                    Master the logic, complexity, and implementation of {category.title.split(' ')[0]} concepts.
                  </p>
                </div>

                <div className="mt-6 flex items-center text-sm font-bold uppercase tracking-widest text-indigo-500 group-hover:text-indigo-400 transition-colors">
                  Start Learning
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 ml-2 transform group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                </div>
              </div>
            </NavLink>
          ))}
        </div>
      </div>
    </div>
  );
}

export default Home2;