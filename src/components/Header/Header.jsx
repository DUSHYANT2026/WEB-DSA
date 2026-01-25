import React, { useState, useEffect } from "react";
import { Link, NavLink } from "react-router-dom";
import { useTheme } from "../../ThemeContext.jsx";

export default function Header() {
  const { darkMode, setDarkMode } = useTheme();
  const [navOpen, setNavOpen] = useState(false);

  useEffect(() => {
    if (darkMode) {
      document.body.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.body.classList.remove("dark");
      localStorage.setItem("theme", "light");
    }
  }, [darkMode]);

  const toggleDarkMode = () => setDarkMode(!darkMode);

  const navItems = [
    { path: "/", name: "Home" },
    { path: "/HOME2", name: "DSA" },
    { path: "/AIML", name: "AI-ML" },
    { path: "/Codeforces", name: "Codeforces" },
    { path: "/Leetcode", name: "LeetCode" },
    { path: "/Github", name: "GitHub" },
    { path: "/about", name: "About" },
    { path: "/contact", name: "Contact" },
    { path: "/CodingAI", name: "Coding-AI" },
  ];

  return (
    <header
      className={`fixed top-0 left-0 w-full z-50 border-b transition-colors duration-300 ${
        darkMode
          ? "bg-[#0f172a] border-slate-800 text-slate-200"
          : "bg-white border-slate-200 text-slate-900"
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          
          {/* Left: Logo & Full Brand Name */}
          <div className="flex items-center flex-shrink-0">
            <Link to="/" className="flex items-center space-x-3 group">
              <img
                src="./aac2.jpg"
                className="h-8 w-8 rounded-full border border-blue-500/50 group-hover:scale-105 transition-transform"
                alt="Logo"
              />
              <span className={`text-lg font-bold tracking-tight transition-colors ${
                darkMode ? "text-white" : "text-slate-900"
              }`}>
                All About Coding
              </span>
            </Link>
          </div>

          {/* Center: Main Navigation */}
          <nav className="hidden lg:flex items-center space-x-1">
            {navItems.map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `px-3 py-2 text-sm font-medium rounded-md transition-all relative ${
                    isActive
                      ? darkMode 
                        ? "text-blue-400 after:absolute after:bottom-0 after:left-0 after:h-0.5 after:w-full after:bg-blue-400" 
                        : "text-blue-600 after:absolute after:bottom-0 after:left-0 after:h-0.5 after:w-full after:bg-blue-600"
                      : "text-slate-500 hover:text-slate-900 dark:hover:text-white"
                  }`
                }
              >
                {item.name}
              </NavLink>
            ))}
          </nav>

          {/* Right: Theme Toggle & Mobile Button */}
          <div className="flex items-center space-x-4">
            <button
              onClick={toggleDarkMode}
              className={`p-2 rounded-md transition-colors ${
                darkMode ? "hover:bg-slate-800 text-slate-400" : "hover:bg-slate-100 text-slate-600"
              }`}
            >
              {darkMode ? (
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" /></svg>
              ) : (
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" /></svg>
              )}
            </button>

            <button
              className="lg:hidden p-2 rounded-md text-slate-500 hover:text-slate-900 dark:hover:text-white"
              onClick={() => setNavOpen(!navOpen)}
            >
              <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {navOpen ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                )}
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      <div className={`lg:hidden transition-all duration-300 ease-in-out ${
        navOpen ? "max-h-screen opacity-100" : "max-h-0 opacity-0 overflow-hidden"
      } ${darkMode ? "bg-[#1e293b]" : "bg-slate-50"}`}>
        <div className="px-4 pt-2 pb-6 space-y-1 shadow-inner">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              onClick={() => setNavOpen(false)}
              className={({ isActive }) =>
                `block px-3 py-3 text-base font-semibold rounded-md transition-colors ${
                  isActive
                    ? "text-blue-500 bg-blue-500/10"
                    : "text-slate-500 hover:bg-slate-200/50 dark:hover:bg-slate-800"
                }`
              }
            >
              {item.name}
            </NavLink>
          ))}
        </div>
      </div>
    </header>
  );
}