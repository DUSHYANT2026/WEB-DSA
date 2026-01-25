import React, { useState } from "react";

function Leetcode() {
  const [username, setUsername] = useState("");
  const [userData, setUserData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchLeetCodeData = async () => {
    if (!username.trim()) {
      setError("Please enter a LeetCode username.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `https://leetcode-api-faisalshohag.vercel.app/${username}`
      );

      if (!response.ok) {
        throw new Error("User not found or API limit reached.");
      }

      const data = await response.json();

      setUserData({
        username: data.username || username,
        totalSolved: data.totalSolved || 0,
        easySolved: data.easySolved || 0,
        mediumSolved: data.mediumSolved || 0,
        hardSolved: data.hardSolved || 0,
        ranking: data.ranking || "N/A",
        contests: data.contestParticipation || [],
      });
    } catch (err) {
      setError(err.message);
      setUserData(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 text-white px-4 py-10">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-blue-500">
            LeetCode Profile Viewer
          </h1>
          <p className="mt-3 text-gray-300">
            Visualize your LeetCode progress beautifully
          </p>
        </div>

        {/* Search Box */}
        <div className="max-w-md mx-auto bg-gray-800/60 backdrop-blur-md border border-gray-700 rounded-2xl p-5 shadow-xl">
          <div className="relative">
            <input
              type="text"
              placeholder="Enter LeetCode username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && fetchLeetCodeData()}
              className="w-full p-4 pr-24 rounded-xl bg-gray-900 border border-gray-700 focus:ring-2 focus:ring-purple-500 outline-none"
            />

            {username && (
              <button
                onClick={() => setUsername("")}
                className="absolute right-16 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white"
              >
                ✕
              </button>
            )}

            <button
              onClick={fetchLeetCodeData}
              disabled={loading}
              className="absolute right-2 top-1/2 -translate-y-1/2 bg-gradient-to-r from-purple-600 to-blue-500 px-4 py-2 rounded-lg font-semibold hover:shadow-lg hover:shadow-purple-500/40 transition-all disabled:opacity-70"
            >
              {loading ? "..." : "Fetch"}
            </button>
          </div>
        </div>

        {/* States */}
        {loading && (
          <p className="text-center mt-10 text-purple-300 animate-pulse">
            Crunching LeetCode data...
          </p>
        )}

        {error && (
          <div className="mt-6 max-w-md mx-auto bg-red-900/40 border border-red-700 text-red-300 p-4 rounded-xl text-center">
            {error}
          </div>
        )}

        {!userData && !loading && !error && (
          <div className="mt-16 text-center text-gray-400">
            <p className="text-lg">👋 Enter a username to get started</p>
            <p className="text-sm mt-2">
              Try: <span className="text-purple-400">tourist</span>,{" "}
              <span className="text-purple-400">errichto</span>,{" "}
              <span className="text-purple-400">neetcode</span>
            </p>
          </div>
        )}

        {/* Profile */}
        {userData && (
          <div className="mt-14 bg-gray-800/60 backdrop-blur-md border border-gray-700 rounded-2xl p-8 shadow-2xl">
            {/* Profile Header */}
            <div className="flex flex-col md:flex-row items-center justify-between gap-6 mb-10">
              <div className="flex items-center gap-4">
                <div className="w-20 h-20 rounded-full bg-gradient-to-br from-purple-600 to-blue-500 flex items-center justify-center text-3xl font-bold">
                  {userData.username.charAt(0).toUpperCase()}
                </div>
                <div>
                  <a
                    href={`https://leetcode.com/${userData.username}`}
                    target="_blank"
                    rel="noreferrer"
                    className="text-2xl font-bold hover:text-purple-400 transition"
                  >
                    {userData.username}
                  </a>
                  <p className="text-gray-400">LeetCode Enthusiast</p>
                </div>
              </div>

              <button
                onClick={() =>
                  window.open(
                    `https://leetcode.com/${userData.username}`,
                    "_blank"
                  )
                }
                className="bg-gradient-to-r from-purple-600 to-blue-500 px-6 py-2 rounded-lg font-semibold hover:shadow-xl hover:shadow-purple-500/40 transition"
              >
                View Profile →
              </button>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Problem Stats */}
              <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-6">
                <h2 className="text-xl font-semibold mb-1">Problem Stats</h2>
                <p className="text-sm text-gray-400 mb-6">
                  Difficulty-wise solved problems
                </p>

                <div className="flex justify-around">
                  <ProgressRing
                    label="Easy"
                    solved={userData.easySolved}
                    total={500}
                    color="text-green-400"
                  />
                  <ProgressRing
                    label="Medium"
                    solved={userData.mediumSolved}
                    total={1000}
                    color="text-yellow-400"
                  />
                  <ProgressRing
                    label="Hard"
                    solved={userData.hardSolved}
                    total={500}
                    color="text-red-400"
                  />
                </div>
              </div>

              {/* Contest Stats */}
              <div className="bg-gray-900/50 border border-gray-700 rounded-xl p-6">
                <h2 className="text-xl font-semibold mb-6">Contest Stats</h2>

                <StatRow label="Global Ranking" value={userData.ranking} icon="🏆" />
                <StatRow
                  label="Contests Participated"
                  value={userData.contests.length}
                  icon="📊"
                />

                <div className="mt-6">
                  <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-purple-500 to-blue-500"
                      style={{
                        width: `${Math.min(
                          100,
                          (userData.totalSolved / 500) * 100
                        )}%`,
                      }}
                    />
                  </div>
                  <p className="text-sm text-gray-400 mt-2">
                    {500 - userData.totalSolved} problems to reach{" "}
                    <span className="text-purple-400 font-semibold">
                      500 milestone 🚀
                    </span>
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------------- Components ---------------- */

function ProgressRing({ label, solved, total, color }) {
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.min(100, (solved / total) * 100);
  const offset = circumference * (1 - progress / 100);

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-24 h-24">
        <svg className="w-full h-full -rotate-90">
          <circle
            cx="48"
            cy="48"
            r={radius}
            stroke="currentColor"
            strokeWidth="8"
            fill="transparent"
            className="text-gray-700"
          />
          <circle
            cx="48"
            cy="48"
            r={radius}
            stroke="currentColor"
            strokeWidth="8"
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className={color}
          />
        </svg>
        <span className="absolute inset-0 flex items-center justify-center font-bold">
          {solved}
        </span>
      </div>
      <p className="mt-2 text-sm text-gray-300">{label}</p>
    </div>
  );
}

function StatRow({ label, value, icon }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-700 last:border-none">
      <div className="flex items-center gap-2 text-gray-300">
        <span>{icon}</span>
        {label}
      </div>
      <span className="font-bold">{value}</span>
    </div>
  );
}

export default Leetcode;
