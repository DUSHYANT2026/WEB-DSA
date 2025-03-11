import React, { PureComponent } from "react";
import { NavLink } from "react-router-dom";

export class AIML extends PureComponent {
  render() {
    return (
      <div className="mt-16 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-4xl font-bold text-center mb-12 text-gray-800 bg-gradient-to-r from-blue-500 to-indigo-500 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer">
          AI/ML Roadmap - Step-by-Step Learning
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
          {/* Foundations */}
          <NavLink to="/Foundation" className="block">
            <div className="bg-gradient-to-r from-gray-800 to-gray-900 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer">
              <h3 className="text-2xl font-bold text-white mb-4">Foundations</h3>
              <p className="text-gray-300">
                Linear algebra, probability, statistics, and Python basics.
              </p>
            </div>
          </NavLink>

          {/* Machine Learning Basics */}
          <NavLink to="/MachineLearning" className="block">
            <div className="bg-gradient-to-r from-green-500 to-teal-500 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer">
              <h3 className="text-2xl font-bold text-white mb-4">ML Basics</h3>
              <p className="text-gray-100">
                Regression, classification, and core ML algorithms.
              </p>
            </div>
          </NavLink>

          {/* Data Preprocessing */}
          <NavLink to="/DataPreprocessing" className="block">
            <div className="bg-gradient-to-r from-purple-600 to-indigo-600 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer">
              <h3 className="text-2xl font-bold text-white mb-4">Data Preprocessing</h3>
              <p className="text-gray-100">
                Cleaning, transformation, and feature engineering.
              </p>
            </div>
          </NavLink>

          {/* Unsupervised Learning */}
          <NavLink to="/UnsupervisedLearning" className="block">
            <div className="bg-gradient-to-r from-orange-500 to-yellow-500 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer">
              <h3 className="text-2xl font-bold text-white mb-4">Unsupervised Learning</h3>
              <p className="text-gray-100">
                Clustering (K-Means) and dimensionality reduction (PCA).
              </p>
            </div>
          </NavLink>

          {/* Model Evaluation */}
          <NavLink to="/ModelEvaluation" className="block">
            <div className="bg-gradient-to-r from-blue-500 to-cyan-500 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer">
              <h3 className="text-2xl font-bold text-white mb-4">Model Evaluation</h3>
              <p className="text-gray-100">
                Metrics, hyperparameter tuning, and bias-variance tradeoff.
              </p>
            </div>
          </NavLink>

          {/* Basic Deployment */}
          <NavLink to="/BasicDeployment" className="block">
            <div className="bg-gradient-to-r from-pink-500 to-red-500 p-6 rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border border-gray-200 cursor-pointer">
              <h3 className="text-2xl font-bold text-white mb-4">Deployment Basics</h3>
              <p className="text-gray-100">
                Model saving/loading and simple Flask API deployment.
              </p>
            </div>
          </NavLink>
        </div>
      </div>
    );
  }
}

export default AIML;