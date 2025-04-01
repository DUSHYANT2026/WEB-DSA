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

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  return (
    <header
      className={`fixed top-0 left-0 w-full z-50 ${
        darkMode
          ? "bg-zinc-900 text-blue-400"
          : "bg-gradient-to-r from-teal-500 to-purple-500"
      } shadow-lg`}
    >
      <div className="max-w-screen-xl mx-auto px-4 py-3 flex justify-between items-center md:px-6">
        {/* Logo on the left */}
        <Link
          to="/"
          className={`flex items-center font-bold text-xl ${
            darkMode
              ? "text-white hover:text-blue-300"
              : "text-white hover:text-orange-300"
          } transition duration-300`}
        >
          <img src="./aac2.jpg" className="mr-2 h-10 rounded-xl" alt="Logo" />
          <span className="hidden sm:inline">AllAboutCoding</span>
        </Link>

        {/* Desktop Navigation in center */}
        <nav className="hidden md:flex items-center space-x-6 mx-auto">
          {[
            { path: "/", name: "Home" },
            { path: "/about", name: "About" },
            { path: "/contact", name: "Contact" },
            { path: "/Leetcode", name: "LeetCode" },
            { path: "/Github", name: "GitHub" },
            { path: "/AIML", name: "AI-ML" },
          ].map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `px-3 py-2 text-sm font-medium transition duration-300 ${
                  isActive
                    ? "text-orange-300 underline font-bold"
                    : darkMode
                    ? "text-blue-400 hover:text-blue-300"
                    : "text-white hover:text-orange-300"
                }`
              }
            >
              {item.name}
            </NavLink>
          ))}
        </nav>

        {/* Right section - dark mode toggle and mobile menu button */}
        <div className="flex items-center space-x-4">
          {/* Dark mode toggle at far right (desktop) */}
          <div className="hidden md:flex items-center">
            <button
              onClick={toggleDarkMode}
              className={`relative inline-flex items-center justify-between h-8 rounded-full w-14 px-1 transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 ${
                darkMode
                  ? "bg-blue-600 focus:ring-blue-400"
                  : "bg-orange-400 focus:ring-orange-300"
              }`}
              aria-label={`Switch to ${darkMode ? "light" : "dark"} mode`}
            >
              {/* Icons on both sides (fade in/out based on mode) */}
              <svg
                className={`w-4 h-4 text-white transition-opacity duration-300 ${
                  darkMode ? "opacity-100" : "opacity-0"
                }`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
                />
              </svg>

              {/* Toggle circle */}
              <span
                className={`absolute inline-flex items-center justify-center w-6 h-6 transform transition-transform duration-300 rounded-full bg-white shadow-md ${
                  darkMode ? "translate-x-6" : "translate-x-0"
                }`}
              />

              <svg
                className={`w-4 h-4 text-white transition-opacity duration-300 ${
                  darkMode ? "opacity-0" : "opacity-100"
                }`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
                />
              </svg>

              <span className="sr-only">Toggle Dark Mode</span>
            </button>
          </div>

          {/* Mobile menu button */}
          {/* Mobile menu button */}
          <button
            className="md:hidden p-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all duration-200"
            onClick={() => setNavOpen(!navOpen)}
            aria-label={navOpen ? "Close menu" : "Open menu"}
            aria-expanded={navOpen}
            aria-controls="mobile-menu"
          >
            <svg
              className={`w-7 h-7 text-white transform transition-transform duration-300 ${
                navOpen ? "rotate-90" : "rotate-0"
              }`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              {navOpen ? (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2.5}
                  d="M6 18L18 6M6 6l12 12"
                  className="transition-opacity duration-300 opacity-100"
                />
              ) : (
                <>
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2.5}
                    d="M4 6h16"
                    className="transition-all duration-300 origin-center"
                    transform={navOpen ? "rotate(45) translate(5,-5)" : ""}
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2.5}
                    d="M4 12h16"
                    className="transition-opacity duration-300"
                    opacity={navOpen ? "0" : "1"}
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2.5}
                    d="M4 18h16"
                    className="transition-all duration-300 origin-center"
                    transform={navOpen ? "rotate(-45) translate(5,5)" : ""}
                  />
                </>
              )}
            </svg>
          </button>
        </div>
      </div>

      {/* Mobile Navigation - follows navbar gradient in light mode */}
      {navOpen && (
        <div
          className={`md:hidden absolute top-16 left-0 right-0 ${
            darkMode
              ? "bg-zinc-800"
              : "bg-gradient-to-r from-teal-500 to-purple-500"
          } shadow-lg`}
        >
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            {[
              { path: "/", name: "Home" },
              { path: "/about", name: "About" },
              { path: "/contact", name: "Contact" },
              { path: "/Leetcode", name: "LeetCode" },
              { path: "/Github", name: "GitHub" },
              { path: "/AIML", name: "AI-ML" },
            ].map((item) => (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `block px-3 py-2 rounded-md text-base font-medium ${
                    isActive
                      ? darkMode
                        ? "bg-zinc-700 text-orange-300"
                        : "bg-teal-700 text-orange-300"
                      : darkMode
                      ? "text-blue-400 hover:bg-zinc-700 hover:text-blue-300"
                      : "text-white hover:bg-teal-700 hover:text-orange-300"
                  }`
                }
                onClick={() => setNavOpen(false)}
              >
                {item.name}
              </NavLink>
            ))}

            {/* Dark mode toggle inside mobile menu */}
            <div className="px-3 py-2 flex items-center justify-between">
              <span className="text-base font-medium text-white mr-3">
                {darkMode ? "Light Mode" : "Dark Mode"}
              </span>

              <button
                onClick={toggleDarkMode}
                className={`relative inline-flex items-center h-7 rounded-full w-12 transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 ${
                  darkMode
                    ? "bg-blue-600 focus:ring-blue-400"
                    : "bg-orange-500 focus:ring-orange-300"
                }`}
                aria-label={`Switch to ${darkMode ? "light" : "dark"} mode`}
              >
                {/* Toggle handle with icon */}
                <span
                  className={`absolute inline-flex items-center justify-center w-5 h-5 transform transition-transform duration-300 rounded-full bg-white shadow-md ${
                    darkMode ? "translate-x-6" : "translate-x-1"
                  }`}
                >
                  {darkMode ? (
                    <svg
                      className="w-3 h-3 text-blue-600"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
                      />
                    </svg>
                  ) : (
                    <svg
                      className="w-3 h-3 text-orange-500"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
                      />
                    </svg>
                  )}
                </span>
                <span className="sr-only">Toggle Dark Mode</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
