import React from "react";
import { NavLink } from "react-router-dom";

function Tree() {
  return (
    <div className="mx-auto w-full max-w-7xl p-4">
      <div className="text-5xl font-extrabold text-center mb-12 bg-gradient-to-r from-pink-500 to-red-500 p-4 sm:p-6 rounded-2xl shadow-xl hover:shadow-2xl transform hover:scale-105 transition-all duration-300 border border-gray-300 cursor-pointer">
        <span className="text-white text-2xl sm:text-3xl">
          Tree (Binary Tree, BST, AVL)
        </span>
      </div>

      {/* Responsive Grid Layout */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* BFS/DFS Card */}
        <NavLink to="/tree1" className="md:col-span-1">
          <div className="bg-gradient-to-r from-blue-500 to-purple-500 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer h-full flex items-center justify-center">
            <span className="text-white text-center text-2xl font-bold">
              BFS And DFS (Tree) Notes with Questions
            </span>
          </div>
        </NavLink>

        {/* Binary Tree Card */}
        <NavLink to="/tree2" className="md:col-span-1">
          <div className="bg-gradient-to-r from-green-500 to-teal-500 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer h-full flex items-center justify-center">
            <span className="text-white text-center text-2xl font-bold">
              Binary Tree And Binary Search Tree Notes
            </span>
          </div>
        </NavLink>

        {/* AVL Tree Card */}
        <NavLink to="/tree3" className="md:col-span-1">
          <div className="bg-gradient-to-r from-yellow-500 to-orange-500 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer h-full flex items-center justify-center">
            <span className="text-white text-center text-2xl font-bold">
              AVL Tree Notes with Questions
            </span>
          </div>
        </NavLink>

        {/* Leetcode Questions Card - spans 2 columns on medium screens */}
        <NavLink to="/tree4" className="md:col-span-2 lg:col-span-1">
          <div className="bg-gradient-to-r from-pink-500 to-red-500 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer h-full flex items-center justify-center">
            <span className="text-white text-center text-2xl font-bold">
              Most Asked Leetcode Questions (Tree)
            </span>
          </div>
        </NavLink>

        {/* Maang Questions Card */}
        <NavLink to="/tree5" className="md:col-span-1 lg:col-span-1">
          <div className="bg-gradient-to-r from-indigo-500 to-blue-500 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer h-full flex items-center justify-center">
            <span className="text-white text-center text-2xl font-bold">
              Hard Questions Asked in Maang Companies
            </span>
          </div>
        </NavLink>
      </div>
    </div>
  );
}

export default Tree;