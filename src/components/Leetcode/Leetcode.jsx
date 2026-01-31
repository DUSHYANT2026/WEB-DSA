import React, { useState, useEffect } from "react";

function Leetcode() {
  const [username, setUsername] = useState("");
  const [userData, setUserData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const CACHE_DURATION = 10 * 60 * 1000; // 10 minutes

  // NEW: Effect to restore data on page refresh
  useEffect(() => {
    const lastUser = localStorage.getItem("last_searched_user");
    if (lastUser) {
      const cacheKey = `leetcode_${lastUser.toLowerCase()}`;
      const cachedItem = localStorage.getItem(cacheKey);

      if (cachedItem) {
        const { data, timestamp } = JSON.parse(cachedItem);
        const isExpired = Date.now() - timestamp > CACHE_DURATION;

        if (!isExpired) {
          setUserData(data);
          setLastUpdated(timestamp);
          setUsername(lastUser);
        }
      }
    }
  }, []);

  const fetchLeetCodeData = async (targetUser = username) => {
    if (!targetUser.trim()) return;

    const cacheKey = `leetcode_${targetUser.trim().toLowerCase()}`;
    const cachedItem = localStorage.getItem(cacheKey);

    if (cachedItem) {
      const { data, timestamp } = JSON.parse(cachedItem);
      if (Date.now() - timestamp < CACHE_DURATION) {
        setUserData(data);
        setLastUpdated(timestamp);
        setError(null);
        return;
      }
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`https://leetcode-api-faisalshohag.vercel.app/${targetUser}`);
      if (!response.ok) throw new Error("User not found");
      const data = await response.json();

      const timestamp = Date.now();
      localStorage.setItem(cacheKey, JSON.stringify({ data, timestamp }));
      localStorage.setItem("last_searched_user", targetUser.trim());

      setUserData(data);
      setLastUpdated(timestamp);
    } catch (err) {
      setError("Failed to fetch data");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-tr from-[#d31027] via-[#ea384d] to-[#ff9900] text-white p-6 font-sans">
      <div className="max-w-6xl mx-auto">
        <div className="flex flex-col items-center mb-10">
          <div className="flex items-center gap-2 mb-4">
            <div className="bg-white p-1 rounded-md">
                <img src="https://leetcode.com/static/images/LeetCode_logo_rvs.png" alt="logo" className="w-6 h-6 invert" />
            </div>
            <h1 className="text-2xl font-bold tracking-tight">LeetCode <span className="font-light opacity-80">api</span></h1>
          </div>
          
          <div className="flex items-center bg-black/40 rounded-lg overflow-hidden border border-white/20 shadow-xl mb-4">
            <input
              type="text"
              className="bg-transparent px-4 py-2 outline-none w-48 text-sm placeholder-white/50"
              placeholder="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && fetchLeetCodeData()}
            />
            <button 
              onClick={() => fetchLeetCodeData()}
              className="bg-white/10 hover:bg-white/20 px-6 py-2 text-sm transition-colors border-l border-white/20 font-bold"
            >
              Okay
            </button>
          </div>

          {lastUpdated && userData && (
            <p className="text-[10px] uppercase tracking-widest opacity-50 mb-6">
              Data Cached: {new Date(lastUpdated).toLocaleTimeString()} (Valid for 10m)
            </p>
          )}

          {userData && (
            <div className="flex flex-col items-center gap-4 animate-fadeIn">
                <div className="relative group">
                    <div className="absolute -inset-1 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-full blur opacity-40"></div>
                    <img 
                        src={userData.avatar || `https://ui-avatars.com/api/?name=${username}&background=random`} 
                        alt="Profile" 
                        className="relative w-24 h-24 rounded-full border-4 border-white/10 object-cover shadow-2xl"
                    />
                </div>
                <div className="text-center">
                    <h2 className="text-3xl font-black tracking-tight uppercase">{userData.username}</h2>
                    <a 
                        href={`https://leetcode.com/${userData.username}`} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="mt-2 inline-flex items-center gap-2 bg-white/10 hover:bg-white/20 px-4 py-1.5 rounded-full text-xs font-bold transition-all border border-white/10 shadow-lg"
                    >
                        View Official Profile 
                    </a>
                </div>
            </div>
          )}
        </div>

        {userData && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 animate-slideUp">
            <StatCard label="Total Solved" value={`${userData.totalSolved}/${userData.totalQuestions}`} progress={userData.totalSolved / userData.totalQuestions} />
            <StatCard label="Ranking" value={userData.ranking} noRing />
            <StatCard label="Easy Solved" value={`${userData.easySolved}/${userData.totalEasy}`} color="text-teal-400" progress={userData.easySolved / userData.totalEasy} />
            <StatCard label="Medium Solved" value={`${userData.mediumSolved}/${userData.totalMedium}`} color="text-yellow-400" progress={userData.mediumSolved / userData.totalMedium} />
            <StatCard label="Hard Solved" value={`${userData.hardSolved}/${userData.totalHard}`} color="text-red-400" progress={userData.hardSolved / userData.totalHard} />
            <StatCard label="Contribution Points" value={userData.contributionPoints} noRing />

            <div className="md:col-span-3 bg-black/30 backdrop-blur-md rounded-3xl overflow-hidden border border-white/10 shadow-2xl mt-4">
              <div className="p-6 pb-2">
                <h3 className="text-xl font-bold">Recent Submissions</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left">
                  <thead>
                    <tr className="bg-black/40 text-xs uppercase tracking-wider text-white/70">
                      <th className="px-6 py-4 font-bold">Title</th>
                      <th className="px-6 py-4 font-bold">Status</th>
                      <th className="px-6 py-4 font-bold">Language</th>
                      <th className="px-6 py-4 font-bold">Timestamp</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 text-sm">
                    {userData.recentSubmissions?.map((sub, index) => (
                      <tr key={index} className="hover:bg-white/5 transition-colors">
                        <td className="px-6 py-4 whitespace-nowrap font-medium">
                          <a href={`https://leetcode.com/problems/${sub.titleSlug}`} target="_blank" rel="noreferrer" className="text-blue-300 hover:text-white transition-colors underline decoration-blue-300/30">
                            {sub.title}
                          </a>
                        </td>
                        <td className={`px-6 py-4 font-bold ${
                          sub.statusDisplay === "Accepted" ? "text-green-400" : 
                          sub.statusDisplay === "Wrong Answer" ? "text-red-400" : "text-orange-400"
                        }`}>
                          {sub.statusDisplay}
                        </td>
                        <td className="px-6 py-4 text-white/80">{sub.lang}</td>
                        <td className="px-6 py-4 text-white/60">
                          {new Date(sub.timestamp * 1000).toLocaleString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
        
        {loading && (
          <div className="flex justify-center mt-20">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-white"></div>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value, color = "text-blue-400", progress, noRing }) {
  const radius = 35;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (progress || 0) * circumference;

  return (
    <div className="bg-white/10 backdrop-blur-lg border border-white/10 rounded-3xl p-8 flex justify-between items-center shadow-xl">
      <div>
        <h3 className="text-xs font-bold uppercase tracking-widest opacity-60 mb-1">{label}</h3>
        <p className="text-3xl font-black">{value}</p>
      </div>
      {!noRing && (
        <div className="relative w-20 h-20">
          <svg className="w-full h-full -rotate-90">
            <circle cx="40" cy="40" r={radius} stroke="rgba(255,255,255,0.05)" strokeWidth="6" fill="transparent" />
            <circle cx="40" cy="40" r={radius} stroke="currentColor" strokeWidth="6" fill="transparent" strokeDasharray={circumference} strokeDashoffset={offset} strokeLinecap="round" className={color} />
          </svg>
        </div>
      )}
    </div>
  );
}

export default Leetcode;