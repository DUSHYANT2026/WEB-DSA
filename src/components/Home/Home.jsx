import React from "react";
import { NavLink } from "react-router-dom";
import { useTheme } from "../../ThemeContext.jsx";

// SVG Icons for the roadmap steps
const icons = {
  dsa: [
    <path strokeLinecap="round" strokeLinejoin="round" d="M14.25 9.75L16.5 12l-2.25 2.25m-4.5 0L7.5 12l2.25-2.25M6 20.25h12A2.25 2.25 0 0020.25 18V6A2.25 2.25 0 0018 3.75H6A2.25 2.25 0 003.75 6v12A2.25 2.25 0 006 20.25z" />,
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 9.75h16.5m-16.5 4.5h16.5m-16.5-1.5h16.5m-16.5-1.5h16.5" />,
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />,
    <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" />,
    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />,
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 4.5v15m6-15v15m-10.5-6h15m-15-3h15" />,
    <path strokeLinecap="round" strokeLinejoin="round" d="M15.042 21.042L15 21m-4.5-4.5L15 21M3 3l3.042 3.042m0 0L9 9m-3-3l3 3" />,
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />,
  ],
  aiml: [
    <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 21l5.25-11.25L21 21m-9-3.75h.008v.008H12v-.008zM12 15h.008v.008H12v-.008zm0-3.75h.008v.008H12v-.008zm0-3.75h.008v.008H12V9zm-3.75 0h.008v.008H8.25V9zm0 3.75h.008v.008H8.25v-.008zm0 3.75h.008v.008H8.25v-.008zM3 9h3.75v3.75H3V9zm0 3.75h3.75v3.75H3v-3.75z" />,
    <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 14.25v2.25m3-4.5v4.5m3-6.75v6.75m3-9v9M6 20.25h12A2.25 2.25 0 0020.25 18V6A2.25 2.25 0 0018 3.75H6A2.25 2.25 0 003.75 6v12A2.25 2.25 0 006 20.25z" />,
    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12a7.5 7.5 0 0015 0m-15 0a7.5 7.5 0 1115 0m-15 0H3m16.5 0H21m-9 6.75v-1.5m-6.364-3.546l-1.06-1.061m12.728 0l-1.061 1.06m-10.607-4.454l1.06-1.06m7.425 0l1.061 1.06" />,
    <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />,
    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />,
    <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 10.5-10.5 18" />,
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 5.25h16.5m-16.5 4.5h16.5m-16.5 4.5h16.5" />,
    <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.898 20.849L16.5 21.75l-.398-.901a2.25 2.25 0 00-1.634-1.634l-.901-.398 2.05-2.05.398.901a2.25 2.25 0 001.634 1.634l.901.398z" />,
  ]
};

export default function Home() {
  const { darkMode } = useTheme();
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const learningFeatures = [
    { text: "Data Structure & Algorithm", darkHoverColor: "hover:text-orange-400", lightHoverColor: "hover:text-orange-500" },
    { text: "Most Important LeetCode Questions", darkHoverColor: "hover:text-orange-400", lightHoverColor: "hover:text-orange-500" },
    { text: "Most Important MAANG Interview Questions", darkHoverColor: "hover:text-orange-400", lightHoverColor: "hover:text-orange-500" },
    { text: "Learn DSA and Get Placed in MAANG Companies", darkHoverColor: "hover:text-orange-400", lightHoverColor: "hover:text-orange-500" },
  ];

  const mlFeatures = [
    { text: "Machine Learning Fundamentals", darkHoverColor: "hover:text-orange-400", lightHoverColor: "hover:text-orange-500" },
    { text: "Most Important Machine Learning Algorithms", darkHoverColor: "hover:text-orange-400", lightHoverColor: "hover:text-orange-500" },
    { text: "Key Machine Learning Interview Questions for Top Tech Companies", darkHoverColor: "hover:text-orange-400", lightHoverColor: "hover:text-orange-500" },
    { text: "Learn Machine Learning and Get Hired by Top AI/ML Companies", darkHoverColor: "hover:text-orange-400", lightHoverColor: "hover:text-orange-500" },
  ];

  const dsaRoadmap = [
    { step: 1, title: "Programming Basics", description: "Master fundamentals of programming and problem-solving.", icon: icons.dsa[0] },
    { step: 2, title: "Arrays & Strings", description: "Learn array manipulation and string algorithms.", icon: icons.dsa[1] },
    { step: 3, title: "Sorting & Searching", description: "Understand essential algorithms for data organization.", icon: icons.dsa[2] },
    { step: 4, title: "Linked Lists", description: "Master pointer manipulation and linked data structures.", icon: icons.dsa[3] },
    { step: 5, title: "Stacks & Queues", description: "Implement LIFO and FIFO data structures.", icon: icons.dsa[4] },
    { step: 6, title: "Trees & BSTs", description: "Work with hierarchical data structures.", icon: icons.dsa[5] },
    { step: 7, title: "Graphs & Algorithms", description: "Solve complex problems with graph theory.", icon: icons.dsa[6] },
    { step: 8, title: "Dynamic Programming", description: "Optimize solutions with memoization techniques.", icon: icons.dsa[7] },
  ];

  const aimlRoadmap = [
    { step: 1, title: "Python & Mathematics", description: "Build foundation in programming and essential math.", icon: icons.aiml[0] },
    { step: 2, title: "Data Analysis", description: "Learn to process and visualize data.", icon: icons.aiml[1] },
    { step: 3, title: "Statistics & Probability", description: "Master statistical concepts for ML.", icon: icons.aiml[2] },
    { step: 4, title: "ML Fundamentals", description: "Understand core machine learning concepts.", icon: icons.aiml[3] },
    { step: 5, title: "Supervised Learning", description: "Implement classification and regression algorithms.", icon: icons.aiml[4] },
    { step: 6, title: "Unsupervised Learning", description: "Work with clustering and dimensionality reduction.", icon: icons.aiml[5] },
    { step: 7, title: "Deep Learning", description: "Build neural networks for complex tasks.", icon: icons.aiml[6] },
    { step: 8, title: "AI Projects", description: "Develop end-to-end AI applications.", icon: icons.aiml[7] },
  ];

  const FeatureText = ({ feature }) => (
    <p
      className={`font-semibold tracking-wide leading-relaxed transform hover:translate-x-4 transition duration-300 ease-in-out ${
        darkMode ? feature.darkHoverColor : feature.lightHoverColor
      }`}
    >
      {feature.text}
    </p>
  );
  
  // --- NEW, MORE BEAUTIFUL ROADMAP COMPONENT ---
  const RoadmapCard = ({ step, gradient }) => (
    <div
      className={`group relative p-5 rounded-xl shadow-lg transition-all duration-300 ease-in-out transform hover:scale-105 hover:shadow-2xl ${
        darkMode
          ? "bg-gray-800/60 backdrop-blur-sm border border-gray-700/50"
          : "bg-white/60 backdrop-blur-sm border border-gray-200/50"
      }`}
    >
      {/* Subtle hover glow effect */}
      <div
        className={`absolute -inset-px rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 ${gradient}`}
      />
      <div className="relative flex items-start space-x-4">
        <div className={`flex-shrink-0 w-12 h-12 rounded-lg ${gradient} flex items-center justify-center`}>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
            className="w-7 h-7 text-white"
          >
            {step.icon}
          </svg>
        </div>
        <div>
          <h3 className={`font-bold text-lg mb-1 ${darkMode ? "text-gray-100" : "text-gray-900"}`}>
            {step.title}
          </h3>
          <p className={`text-sm ${darkMode ? "text-gray-400" : "text-gray-600"}`}>
            {step.description}
          </p>
        </div>
      </div>
    </div>
  );

  const Roadmap = ({ title, steps, route, gradient, titleGradient }) => (
    <div className="mb-24">
      <NavLink to={route} onClick={scrollToTop}>
        <h2
          className={`text-3xl font-bold text-center mb-16 p-5 rounded-2xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition duration-300 border text-white ${titleGradient} ${
            darkMode ? "border-gray-700" : "border-transparent"
          }`}
        >
          {title}
        </h2>
      </NavLink>

      <div className="relative">
        <div
          className={`hidden lg:block absolute top-5 left-1/2 w-1.5 h-[calc(100%-2rem)] -translate-x-1/2 rounded-full ${gradient} opacity-50`}
        />

        <div className="space-y-12 lg:space-y-0 lg:grid lg:grid-cols-[1fr_auto_1fr] lg:gap-y-20 items-start">
          {steps.map((step, index) => {
            const isEven = index % 2 === 0;
            return (
              <React.Fragment key={index}>
                {/* --- MOBILE/TABLET VIEW (Single-sided timeline) --- */}
                <div className="lg:hidden flex items-start space-x-4">
                  <div className="flex flex-col items-center">
                    <div
                      className={`flex items-center justify-center w-12 h-12 rounded-full font-bold text-white z-10 shadow-lg ${gradient}`}
                    >
                      {step.step}
                    </div>
                    {index !== steps.length - 1 && (
                      <div className={`w-0.5 h-full min-h-[8rem] ${gradient}`} />
                    )}
                  </div>
                  <div className="pt-1 w-full">
                    <RoadmapCard step={step} gradient={gradient} />
                  </div>
                </div>

                {/* --- DESKTOP VIEW (Zigzag layout) --- */}
                {isEven ? (
                  <>
                    <div className="hidden lg:block pr-10 text-right">
                      <RoadmapCard step={step} gradient={gradient} />
                    </div>
                    <div className="hidden lg:flex justify-center">
                      <div
                        className={`relative flex items-center justify-center w-14 h-14 rounded-full font-bold text-xl text-white z-10 shadow-xl ${gradient} ${
                          darkMode ? "ring-8 ring-zinc-900" : "ring-8 ring-white"
                        }`}
                      >
                        {step.step}
                      </div>
                    </div>
                    <div />
                  </>
                ) : (
                  <>
                    <div />
                    <div className="hidden lg:flex justify-center">
                      <div
                        className={`relative flex items-center justify-center w-14 h-14 rounded-full font-bold text-xl text-white z-10 shadow-xl ${gradient} ${
                          darkMode ? "ring-8 ring-zinc-900" : "ring-8 ring-white"
                        }`}
                      >
                        {step.step}
                      </div>
                    </div>
                    <div className="hidden lg:block pl-10">
                      <RoadmapCard step={step} gradient={gradient} />
                    </div>
                  </>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      <div className="text-center mt-16">
        <NavLink to={route} onClick={scrollToTop}>
          <button
            className={`px-8 py-3 rounded-full font-semibold text-white transition-all duration-300 shadow-lg hover:shadow-xl transform hover:-translate-y-1 ${titleGradient}`}
          >
            Explore the {title}
          </button>
        </NavLink>
      </div>
    </div>
  );

  return (
    <div
      className={`min-h-screen flex flex-col items-center justify-center px-4 sm:px-6 lg:px-8 ${
        darkMode ? "bg-zinc-900 text-white" : "bg-gray-50 text-gray-900"
      } transition-colors duration-300`}
    >
      <div className="w-full max-w-6xl mx-auto">
        {/* Hero Section */}
        <section
          className={`relative overflow-hidden rounded-3xl mx-auto sm:mx-12 md:mx-16 my-8 py-10 shadow-2xl bg-gradient-to-r ${
            darkMode
              ? "from-gray-800 to-gray-900 border-gray-700"
              : "from-gray-50 to-gray-100 border-gray-200"
          } border hover:shadow-3xl transform hover:scale-[1.01] transition duration-500 ease-in-out`}
        >
          <div className="relative z-10 px-6 mx-auto grid grid-cols-1 lg:grid-cols-2 gap-10 items-center">
            <div className="space-y-6 text-center lg:text-left order-2 lg:order-1">
              <h1
                className={`text-4xl md:text-5xl font-bold pb-3 bg-clip-text text-transparent bg-gradient-to-r ${
                  darkMode
                    ? "from-orange-400 to-purple-500"
                    : "from-orange-500 to-purple-600"
                } transition duration-500 hover:scale-105`}
              >
                All About Coding
              </h1>
              <div
                className={`text-lg space-y-3 ${
                  darkMode ? "text-gray-300" : "text-gray-700"
                }`}
              >
                {learningFeatures.map((feature, index) => (
                  <FeatureText key={index} feature={feature} />
                ))}
              </div>
              <div
                className={`text-lg space-y-3 mt-6 ${
                  darkMode ? "text-gray-300" : "text-gray-700"
                }`}
              >
                {mlFeatures.map((feature, index) => (
                  <FeatureText key={index} feature={feature} />
                ))}
              </div>
            </div>
            <div className="flex justify-center lg:justify-end order-1 lg:order-2">
              <div className="relative overflow-hidden rounded-3xl shadow-2xl transform hover:rotate-2 transition duration-500 ease-in-out max-w-sm md:max-w-md">
                <img
                  className={`w-full h-auto max-h-96 object-cover rounded-3xl border-4 ${
                    darkMode ? "border-gray-600" : "border-gray-300"
                  } hover:scale-105 transition-all duration-500`}
                  src={"./aac2.jpg"}
                  alt="ALL ABOUT CODING"
                  loading="lazy"
                />
                <div
                  className={`absolute inset-0 bg-gradient-to-t ${
                    darkMode ? "from-gray-900" : "from-gray-100"
                  } opacity-20`}
                ></div>
              </div>
            </div>
          </div>
        </section>

        {/* Roadmaps Section */}
        <section className="w-full mx-auto px-4 py-12">
          <h2
            className={`text-4xl font-bold text-center mb-20 bg-clip-text text-transparent bg-gradient-to-r ${
                darkMode ? "from-purple-400 to-pink-500" : "from-purple-600 to-pink-600"
            }`}
          >
            Your Learning Journey Starts Here
          </h2>
          <Roadmap
            title="DSA Mastery Roadmap"
            steps={dsaRoadmap}
            route="/Home2"
            gradient={darkMode ? "bg-gradient-to-br from-pink-500 to-red-500" : "bg-gradient-to-br from-pink-400 to-red-400"}
            titleGradient={darkMode ? "bg-gradient-to-r from-pink-600 to-red-600" : "bg-gradient-to-r from-pink-500 to-red-500"}
          />
          <Roadmap
            title="AI/ML Mastery Roadmap"
            steps={aimlRoadmap}
            route="/Ai_Ml"
            gradient={darkMode ? "bg-gradient-to-br from-sky-500 to-teal-500" : "bg-gradient-to-br from-sky-400 to-teal-400"}
            titleGradient={darkMode ? "bg-gradient-to-r from-cyan-600 to-teal-600" : "bg-gradient-to-r from-cyan-500 to-teal-500"}
          />
        </section>

        {/* Footer Section (unchanged) */}
        <footer
          className={`w-full mt-12 py-6 border-t ${
            darkMode
              ? "border-gray-800 bg-zinc-800"
              : "border-gray-200 bg-gray-50"
          } rounded-t-lg`}
        >
          <div className="mx-auto px-4 text-center">
            <h3
              className={`text-xl font-bold mb-3 ${
                darkMode ? "text-gray-200" : "text-gray-800"
              }`}
            >
              Ready to Master Coding?
            </h3>
            <p
              className={`mb-4 text-sm ${
                darkMode ? "text-gray-400" : "text-gray-600"
              }`}
            >
              Join thousands of developers who have transformed their careers
              with our comprehensive learning paths.
            </p>
            <div className="flex justify-center space-x-4">
              <button
                onClick={() => (window.location.href = "/login")}
                className={`px-5 py-2 rounded-lg text-sm font-semibold transition-all duration-300 ${
                  darkMode
                    ? "bg-purple-600 text-white hover:bg-purple-700"
                    : "bg-purple-500 text-white hover:bg-purple-600"
                }`}
              >
                Sign Up for Free
              </button>
              <button
                onClick={() => (window.location.href = "/about")}
                className={`px-5 py-2 rounded-lg text-sm font-semibold transition-all duration-300 ${
                  darkMode
                    ? "border border-purple-600 text-purple-400 hover:bg-purple-900 hover:bg-opacity-30"
                    : "border border-purple-500 text-purple-600 hover:bg-purple-100"
                }`}
              >
                Learn More
              </button>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}