import React from 'react';
import { NavLink } from 'react-router-dom'; // Import NavLink

export const DataPreprocessing = () => {
  return (
    <div className="mx-auto w-full max-w-7xl p-6">
      {/* Header Section */}
      <div className="text-5xl font-extrabold text-center mb-12 text-gray-900 bg-gradient-to-r from-orange-500 to-pink-500 p-8 rounded-2xl shadow-xl hover:shadow-2xl transform hover:scale-105 transition-all duration-300 border border-gray-300 cursor-pointer sm:flex sm:items-center sm:justify-center sm:gap-4">
        <span className="text-gray-100 sm:text-center hidden sm:block text-3xl">
          Data Preprocessing
        </span>
      </div>

      {/* Grid Layout for Topics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <NavLink to="/DataCleaning" aria-label="Learn about Data Cleaning">
          <div className="bg-gradient-to-r from-blue-500 to-purple-500 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer">
            <span className="text-white text-center text-2xl font-bold">Data Cleaning</span>
          </div>
        </NavLink>

        <NavLink to="/DataTransformation" aria-label="Learn about Data Transformation">
          <div className="bg-gradient-to-r from-green-500 to-teal-500 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer">
            <span className="text-white text-center text-2xl font-bold">Data Transformation</span>
          </div>
        </NavLink>

        <NavLink to="/FeatureEngineering" aria-label="Learn about Feature Engineering">
          <div className="bg-gradient-to-r from-yellow-500 to-orange-500 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer">
            <span className="text-white text-center text-2xl font-bold">Feature Engineering</span>
          </div>
        </NavLink>

        <NavLink to="/SplittingData" aria-label="Learn about Splitting Data">
          <div className="bg-gradient-to-r from-pink-500 to-red-500 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer">
            <span className="text-white text-center text-2xl font-bold">Splitting Data</span>
          </div>
        </NavLink>
      </div>
    </div>
  );
};

export default DataPreprocessing;