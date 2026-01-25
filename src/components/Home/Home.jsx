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
    { text: "Data Structure & Algorithm", color: "text-orange-500" },
    { text: "Most Important LeetCode Questions", color: "text-orange-500" },
    { text: "Most Important MAANG Interview Questions", color: "text-orange-500" },
    { text: "Learn DSA and Get Placed in MAANG", color: "text-orange-500" },
  ];

  const mlFeatures = [
    { text: "Machine Learning Fundamentals", color: "text-blue-500" },
    { text: "Most Important ML Algorithms", color: "text-blue-500" },
    { text: "Key AI Interview Questions", color: "text-blue-500" },
    { text: "Learn ML and Get Hired by Top AI Companies", color: "text-blue-500" },
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
    <div className="flex items-center space-x-3 group">
      <div className={`w-1.5 h-1.5 rounded-full ${feature.color} opacity-60 group-hover:scale-150 transition-transform duration-300`} />
      <p className={`text-sm md:text-base font-medium tracking-tight transition duration-300 group-hover:translate-x-2 ${
        darkMode ? "text-gray-400 group-hover:text-white" : "text-gray-600 group-hover:text-gray-900"
      }`}>
        {feature.text}
      </p>
    </div>
  );
  
  const RoadmapCard = ({ step, gradient }) => (
    <div className={`group relative p-6 rounded-2xl shadow-sm transition-all duration-500 hover:-translate-y-2 hover:shadow-2xl ${
        darkMode ? "bg-zinc-800/40 border border-zinc-700/50 hover:bg-zinc-800/80" : "bg-white border border-gray-100"
      }`}>
      <div className="flex items-start space-x-4">
        <div className={`flex-shrink-0 w-12 h-12 rounded-xl ${gradient} flex items-center justify-center transform group-hover:rotate-6 transition-transform duration-300 shadow-lg`}>
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor" className="w-6 h-6 text-white">
            {step.icon}
          </svg>
        </div>
        <div>
          <h3 className={`font-bold text-lg mb-1 tracking-tight ${darkMode ? "text-zinc-100" : "text-zinc-900"}`}>
            {step.title}
          </h3>
          <p className={`text-sm leading-relaxed ${darkMode ? "text-zinc-400" : "text-zinc-500"}`}>
            {step.description}
          </p>
        </div>
      </div>
    </div>
  );

  const Roadmap = ({ title, steps, route, gradient, titleGradient }) => (
    <div className="mb-32">
      <div className="flex flex-col items-center mb-16">
        <NavLink to={route} onClick={scrollToTop} className="group">
          <h2 className={`text-2xl md:text-4xl font-extrabold px-10 py-4 rounded-full text-white shadow-xl transition-all duration-300 group-hover:scale-105 ${titleGradient}`}>
            {title}
          </h2>
        </NavLink>
        <div className={`w-px h-12 mt-4 ${gradient} opacity-30`} />
      </div>

      <div className="relative">
        <div className={`hidden lg:block absolute top-0 left-1/2 w-0.5 h-full -translate-x-1/2 bg-gradient-to-b from-transparent via-gray-400 to-transparent opacity-20`} />
        <div className="space-y-12 lg:space-y-0 lg:grid lg:grid-cols-[1fr_auto_1fr] lg:gap-y-16 items-center">
          {steps.map((step, index) => {
            const isEven = index % 2 === 0;
            return (
              <React.Fragment key={index}>
                <div className="lg:hidden flex items-start space-x-6">
                   <div className={`flex-shrink-0 flex items-center justify-center w-10 h-10 rounded-full font-bold text-white shadow-md ${gradient}`}>
                    {step.step}
                  </div>
                  <RoadmapCard step={step} gradient={gradient} />
                </div>
                {isEven ? (
                  <>
                    <div className="hidden lg:block"><RoadmapCard step={step} gradient={gradient} /></div>
                    <div className="hidden lg:flex justify-center z-10">
                      <div className={`w-12 h-12 rounded-full flex items-center justify-center font-bold text-white shadow-xl ring-4 ${darkMode ? 'ring-zinc-900' : 'ring-white'} ${gradient}`}>{step.step}</div>
                    </div>
                    <div />
                  </>
                ) : (
                  <>
                    <div />
                    <div className="hidden lg:flex justify-center z-10">
                      <div className={`w-12 h-12 rounded-full flex items-center justify-center font-bold text-white shadow-xl ring-4 ${darkMode ? 'ring-zinc-900' : 'ring-white'} ${gradient}`}>{step.step}</div>
                    </div>
                    <div className="hidden lg:block"><RoadmapCard step={step} gradient={gradient} /></div>
                  </>
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>
    </div>
  );

  return (
    <div className={`min-h-screen ${darkMode ? "bg-zinc-900 text-zinc-100" : "bg-[#fcfcfd] text-zinc-900"} transition-colors duration-500 font-sans`}>
      <div className="max-w-7xl mx-auto px-6 lg:px-8 pt-12">
        
        {/* Hero Section */}
        <section className={`relative mb-24 rounded-[2.5rem] overflow-hidden border ${
          darkMode ? "bg-zinc-800/30 border-zinc-700/50" : "bg-white border-zinc-200/60 shadow-sm"
        }`}>
          <div className="absolute top-0 right-0 -mt-20 -mr-20 w-96 h-96 bg-orange-500/10 rounded-full blur-3xl" />
          <div className="relative z-10 p-8 md:p-16 grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
            <div className="space-y-8">
              <div>
                <span className={`inline-block px-4 py-1.5 mb-4 text-xs font-bold tracking-widest uppercase rounded-full ${darkMode ? "bg-orange-500/20 text-orange-400" : "bg-orange-100 text-orange-600"}`}>
                  Accelerate your growth
                </span>
                <h1 className="text-5xl md:text-6xl font-black tracking-tight leading-none mb-4">
                  All About <span className="bg-clip-text text-transparent bg-gradient-to-r from-orange-500 to-purple-600">Coding</span>
                </h1>
                <p className={`text-lg max-w-md ${darkMode ? "text-zinc-400" : "text-zinc-500"}`}>
                  Industry-leading curriculum for Data Structures, Algorithms, and Artificial Intelligence.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-4">
                <div className="space-y-3">{learningFeatures.map((f, i) => <FeatureText key={i} feature={f} />)}</div>
                <div className="space-y-3">{mlFeatures.map((f, i) => <FeatureText key={i} feature={f} />)}</div>
              </div>

              <div className="flex flex-wrap gap-4 pt-4">
                 <button onClick={() => window.location.href="/login"} className="px-8 py-4 rounded-xl bg-orange-500 text-white font-bold hover:bg-orange-600 transition shadow-lg shadow-orange-500/20 transform hover:-translate-y-0.5 active:scale-95">
                    Sign Up / Login
                 </button>
              </div>
            </div>

            <div className="relative group">
              <div className="absolute -inset-1 bg-gradient-to-r from-orange-500 to-purple-600 rounded-[2rem] blur opacity-25 group-hover:opacity-40 transition duration-1000"></div>
              <div className={`relative overflow-hidden rounded-[2rem] border-4 ${darkMode ? 'border-zinc-800' : 'border-white'} shadow-2xl`}>
                <img className="w-full h-auto object-cover transform transition duration-700 group-hover:scale-105" src="./aac2.jpg" alt="Platform Preview" />
              </div>
            </div>
          </div>
        </section>

        {/* Roadmap Section */}
        <section className="pb-24">
          <div className="text-center mb-24">
            <h2 className={`text-3xl md:text-5xl font-black mb-4 ${darkMode ? 'text-white' : 'text-zinc-900'}`}>Learning Journeys</h2>
            <p className={`text-lg ${darkMode ? 'text-zinc-400' : 'text-zinc-500'}`}>Structured paths designed to take you from beginner to job-ready.</p>
          </div>

          <Roadmap
            title="DSA Mastery"
            steps={dsaRoadmap}
            route="/Home2"
            gradient="bg-gradient-to-br from-pink-500 to-rose-600"
            titleGradient="bg-gradient-to-r from-pink-500 to-rose-600"
          />
          
          <Roadmap
            title="AI/ML Expert"
            steps={aimlRoadmap}
            route="/AIML"
            gradient="bg-gradient-to-br from-indigo-500 to-cyan-500"
            titleGradient="bg-gradient-to-r from-indigo-500 to-cyan-500"
          />
        </section>

        {/* Simple Minimal Footer */}
      </div>
    </div>
  );
}