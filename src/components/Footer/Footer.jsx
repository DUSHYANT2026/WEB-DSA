import { NavLink } from "react-router-dom";
import { Github, Linkedin, ExternalLink, ArrowUp } from "lucide-react";

export default function Footer() {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-gradient-to-b from-gray-900 to-black text-gray-300 py-10 relative z-20">
      <div className="max-w-7xl mx-auto px-6 lg:px-12">

        {/* ================= TOP SECTION ================= */}
        <div className="flex flex-col sm:flex-row justify-between gap-10 mb-10">

          {/* Coding Journey (LEFT) */}
          <div>
            <h2 className="text-sm font-semibold uppercase tracking-wider text-orange-400 mb-4">
              Coding Journey
            </h2>
            <p className="text-sm text-gray-400 leading-relaxed mb-3 max-w-md">
              Track progress, practice consistently, and grow as a developer.
            </p>
            <a
              href="https://codolio.com/"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-sm text-gray-300 hover:text-orange-400 transition"
            >
              Visit Codolio
              <ExternalLink className="w-4 h-4" />
            </a>
          </div>

          {/* Connect (RIGHT) */}
          <div className="sm:text-right">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-orange-400 mb-4">
              Connect
            </h2>
            <ul className="space-y-3 text-sm">
              <li>
                <a
                  href="https://github.com/DUSHYANT2026"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-3 hover:text-orange-400 transition"
                >
                  <Github className="w-4 h-4" />
                  GitHub
                </a>
              </li>
              <li>
                <a
                  href="https://www.linkedin.com/in/dushyant-kumar-b8594a251/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-3 hover:text-orange-400 transition"
                >
                  <Linkedin className="w-4 h-4" />
                  LinkedIn
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* ================= RESOURCES CARD ================= */}
        <div className="bg-gray-900/60 border border-gray-800 rounded-xl p-6 mb-8">

          {/* Learning Resources */}
          <h3 className="text-center text-sm font-semibold uppercase tracking-wider text-orange-400 mb-4">
            Learning Resources
          </h3>
          <div className="flex flex-wrap justify-center gap-3 mb-6">
            {[
              ["GFG DSA", "https://www.geeksforgeeks.org/dsa-tutorial-learn-data-structures-and-algorithms/"],
              ["Take U Forward", "https://takeuforward.org/strivers-a2z-dsa-course/"],
              ["USACO Guide", "https://usaco.guide/"],
              ["W3Schools", "https://www.w3schools.com/dsa/"],
              ["JavaTpoint", "https://www.javatpoint.com/data-structure-tutorial"],
              ["CP Algorithms", "https://cp-algorithms.com/"],
            ].map(([name, link]) => (
              <a
                key={name}
                href={link}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs px-4 py-1.5 rounded-full bg-gray-800 hover:bg-orange-500 hover:text-white transition shadow-sm"
              >
                {name}
              </a>
            ))}
          </div>

          {/* Divider */}
          <div className="border-t border-gray-800 my-4"></div>

          {/* Coding Platforms */}
          <h3 className="text-center text-sm font-semibold uppercase tracking-wider text-orange-400 mb-4">
            Coding Platforms
          </h3>
          <div className="flex flex-wrap justify-center gap-3">
            {[
              ["LeetCode", "https://leetcode.com/problemset/"],
              ["Codeforces", "https://codeforces.com/"],
              ["CodeChef", "https://www.codechef.com/"],
              ["GeeksForGeeks", "https://www.geeksforgeeks.org/"],
              ["HackerRank", "https://www.hackerrank.com/dashboard"],
              ["CodeStudio", "https://www.naukri.com/code360/problems"],
            ].map(([name, link]) => (
              <a
                key={name}
                href={link}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs px-4 py-1.5 rounded-full bg-gray-800 hover:bg-orange-500 hover:text-white transition shadow-sm"
              >
                {name}
              </a>
            ))}
          </div>
        </div>

        {/* ================= BOTTOM BAR ================= */}
        <div className="flex items-center justify-between border-t border-gray-800 pt-5 text-xs text-gray-500">
          <p>
            © {currentYear}{" "}
            <span className="text-gray-400">All About Coding</span>. All rights reserved.
          </p>

          <button
            onClick={scrollToTop}
            aria-label="Scroll to top"
            className="p-2 rounded-full bg-orange-500 hover:bg-orange-600 text-white transition"
          >
            <ArrowUp className="w-4 h-4" />
          </button>
        </div>

      </div>
    </footer>
  );
}
