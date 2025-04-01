import React from "react";
import { NavLink } from "react-router-dom";

function Arrays() {
  return (
    <div className="mx-auto w-full max-w-7xl">
      <div className="text-5xl font-extrabold text-center mb-12 text-gray-900 bg-gradient-to-r from-blue-500 to-purple-500 p-8 rounded-2xl shadow-xl hover:shadow-2xl transform hover:scale-105 transition-all duration-300 border border-gray-300 cursor-pointer sm:flex sm:items-center sm:justify-center sm:gap-4">
        <span className="text-gray-100 text-center text-3xl">
          Array, String, Matrix Problems
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        <NavLink to="/arrays-basic">
          <div className="bg-gradient-to-r from-blue-400 to-indigo-500 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer">
            <span className="text-white text-center text-2xl font-bold">
              Basic Array Operations
            </span>
          </div>
        </NavLink>

        <NavLink to="/strings-problems">
          <div className="bg-gradient-to-r from-green-400 to-teal-500 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer">
            <span className="text-white text-center text-2xl font-bold">
              String Manipulation Problems
            </span>
          </div>
        </NavLink>

        <NavLink to="/matrix-problems">
          <div className="bg-gradient-to-r from-purple-400 to-pink-500 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer">
            <span className="text-white text-center text-2xl font-bold">
              Matrix Traversal Problems
            </span>
          </div>
        </NavLink>

        <NavLink to="/advanced-array">
          <div className="bg-gradient-to-r from-yellow-400 to-orange-500 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer">
            <span className="text-white text-center text-2xl font-bold">
              Advanced Array Techniques
            </span>
          </div>
        </NavLink>
      </div>
    </div>
  );
}

export default Arrays;
