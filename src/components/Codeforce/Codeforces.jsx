import React, { useState, useEffect } from "react";
import { User, Trophy, BarChart3, MapPin, Globe, Calendar, ExternalLink, Award } from "lucide-react";

function Codeforces() {
    const [handle, setHandle] = useState("");
    const [userData, setUserData] = useState(null);
    const [submissions, setSubmissions] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Cache duration: 30 minutes
    const CACHE_DURATION = 30 * 60 * 1000; 

    // Restore from local storage on mount
    useEffect(() => {
        const lastHandle = localStorage.getItem("cf_last_handle");
        if (lastHandle) {
            const cachedData = localStorage.getItem(`cf_data_${lastHandle.toLowerCase()}`);
            if (cachedData) {
                const { user, subs, timestamp } = JSON.parse(cachedData);
                const isExpired = Date.now() - timestamp > CACHE_DURATION;

                if (!isExpired) {
                    setUserData(user);
                    setSubmissions(subs);
                    setHandle(lastHandle); // Keep the input in sync
                }
            }
        }
    }, []);

    const fetchCodeforcesData = async () => {
        const query = handle.trim();
        if (!query) {
            setError("Please enter a Codeforces handle.");
            return;
        }

        // Check local storage before API call
        const cacheKey = `cf_data_${query.toLowerCase()}`;
        const cachedItem = localStorage.getItem(cacheKey);
        
        if (cachedItem) {
            const { user, subs, timestamp } = JSON.parse(cachedItem);
            if (Date.now() - timestamp < CACHE_DURATION) {
                setUserData(user);
                setSubmissions(subs);
                setError(null);
                return;
            }
        }

        setLoading(true);
        setError(null);

        try {
            const [userRes, subsRes] = await Promise.all([
                fetch(`https://codeforces.com/api/user.info?handles=${query}`),
                fetch(`https://codeforces.com/api/user.status?handle=${query}&from=1&count=50`)
            ]);

            const userJson = await userRes.json();
            const subsJson = await subsRes.json();

            if (userJson.status !== 'OK') throw new Error(userJson.comment || "User not found.");
            
            const user = userJson.result[0];
            const subs = subsJson.result;

            // Strict Storage Update
            localStorage.setItem(cacheKey, JSON.stringify({
                user,
                subs,
                timestamp: Date.now()
            }));
            localStorage.setItem("cf_last_handle", query);

            setUserData(user);
            setSubmissions(subs);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const getRankColor = (rating) => {
        if (!rating) return 'text-gray-400';
        if (rating < 1200) return 'text-gray-400';
        if (rating < 1400) return 'text-green-500';
        if (rating < 1600) return 'text-cyan-400';
        if (rating < 1900) return 'text-blue-600';
        if (rating < 2100) return 'text-purple-500';
        if (rating < 2400) return 'text-orange-400';
        return 'text-red-600';
    };

    const getVerdictStyle = (verdict) => {
        if (verdict === 'OK') return 'bg-green-500/20 text-green-400 border-green-500/50';
        if (verdict === 'WRONG_ANSWER') return 'bg-red-500/20 text-red-400 border-red-500/50';
        return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
    };

    return (
        <div className="min-h-screen bg-[#0b0f1a] text-slate-200 p-4 md:p-10 selection:bg-cyan-500/30">
            <div className="fixed top-0 left-0 w-full h-full overflow-hidden pointer-events-none -z-10">
                <div className="absolute top-[-10%] right-[-10%] w-[500px] h-[500px] bg-cyan-500/10 blur-[120px] rounded-full" />
                <div className="absolute bottom-[-10%] left-[-10%] w-[500px] h-[500px] bg-blue-600/10 blur-[120px] rounded-full" />
            </div>

            <div className="max-w-6xl mx-auto">
                <header className="flex flex-col items-center mb-16 space-y-6">
                    <h1 className="text-5xl md:text-6xl font-black tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-600">
                        Codeforces <span className="text-white/90">Pulse</span>
                    </h1>
                    <div className="relative w-full max-w-xl group">
                        <div className="absolute -inset-1 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-2xl blur opacity-25 group-focus-within:opacity-50 transition duration-1000"></div>
                        <div className="relative flex items-center bg-gray-900 border border-white/10 p-2 rounded-2xl">
                            <SearchIcon className="ml-4 text-gray-500" />
                            <input
                                type="text"
                                placeholder="Search handle (e.g. tourist)"
                                value={handle}
                                onChange={(e) => setHandle(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && fetchCodeforcesData()}
                                className="w-full p-3 bg-transparent outline-none text-white placeholder-gray-500"
                            />
                            <button
                                onClick={fetchCodeforcesData}
                                disabled={loading}
                                className="bg-gradient-to-r from-cyan-500 to-blue-600 px-6 py-3 rounded-xl font-bold hover:scale-105 transition-all disabled:opacity-50"
                            >
                                {loading ? "..." : "Analyze"}
                            </button>
                        </div>
                    </div>
                </header>

                {error && <div className="text-center p-4 bg-red-500/10 border border-red-500/50 rounded-xl text-red-400 mb-8 max-w-md mx-auto">{error}</div>}

                {userData && (
                    <main className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in fade-in slide-in-from-bottom-10 duration-700">
                        <div className="lg:col-span-4 space-y-6">
                            <div className="bg-white/5 border border-white/10 backdrop-blur-xl p-8 rounded-[2.5rem] shadow-2xl relative overflow-hidden group">
                                <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:rotate-12 transition-transform">
                                    <Trophy size={100} />
                                </div>
                                
                                <div className="flex flex-col items-center text-center">
                                    <img src={userData.titlePhoto} className="w-32 h-32 rounded-3xl object-cover border-2 border-white/20 mb-6 shadow-2xl" alt="Avatar" />
                                    <h2 className={`text-3xl font-black mb-1 ${getRankColor(userData.rating)}`}>
                                        {userData.handle}
                                    </h2>
                                    <span className="text-xs font-bold uppercase tracking-widest text-gray-500 mb-6">
                                        {userData.rank || "Unranked"}
                                    </span>

                                    <div className="grid grid-cols-2 gap-4 w-full">
                                        <div className="bg-white/5 p-4 rounded-2xl border border-white/5">
                                            <p className={`text-2xl font-black ${getRankColor(userData.rating)}`}>{userData.rating || 0}</p>
                                            <p className="text-[10px] font-bold uppercase text-gray-500 tracking-tighter">Current</p>
                                        </div>
                                        <div className="bg-white/5 p-4 rounded-2xl border border-white/5">
                                            <p className={`text-2xl font-black ${getRankColor(userData.maxRating)}`}>{userData.maxRating || 0}</p>
                                            <p className="text-[10px] font-bold uppercase text-gray-500 tracking-tighter">Peak</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="mt-8 pt-8 border-t border-white/5 space-y-4">
                                    <InfoItem icon={<MapPin size={18}/>} text={userData.country || "Unknown Location"} />
                                    <InfoItem icon={<Award size={18}/>} text={userData.organization || "Independent"} />
                                    <InfoItem icon={<Calendar size={18}/>} text={`Joined ${new Date(userData.registrationTimeSeconds * 1000).getFullYear()}`} />
                                </div>
                            </div>
                        </div>

                        <div className="lg:col-span-8 space-y-6">
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                <StatCard title="Submissions" val={submissions.length} detail="Recent Status" color="text-cyan-400" />
                                <StatCard title="Contribution" val={userData.contribution} detail="Community Score" color="text-purple-400" />
                                <StatCard title="Rating" val={userData.rating || "N/A"} detail="Competitive Tier" color="text-green-400" />
                            </div>

                            <div className="bg-white/5 border border-white/10 backdrop-blur-xl rounded-[2.5rem] p-8">
                                <div className="flex items-center justify-between mb-8">
                                    <h3 className="text-2xl font-bold flex items-center">
                                        <BarChart3 className="mr-3 text-cyan-400" />
                                        Submission Activity
                                    </h3>
                                    <a href={`https://codeforces.com/submissions/${userData.handle}`} target="_blank" className="text-sm font-bold text-cyan-400 flex items-center hover:underline">
                                        View All <ExternalLink size={14} className="ml-1" />
                                    </a>
                                </div>

                                <div className="space-y-3 overflow-y-auto max-h-[500px] pr-4 custom-scrollbar">
                                    {submissions.map((sub, idx) => (
                                        <div key={idx} className="group bg-white/5 hover:bg-white/10 border border-white/5 p-5 rounded-2xl flex items-center justify-between transition-all">
                                            <div>
                                                <h4 className="font-bold text-white group-hover:text-cyan-400 transition-colors">
                                                    {sub.problem.name}
                                                </h4>
                                                <div className="flex items-center mt-1 space-x-3 text-xs text-gray-500">
                                                    <span>{sub.programmingLanguage}</span>
                                                    <span>•</span>
                                                    <span>{new Date(sub.creationTimeSeconds * 1000).toLocaleDateString()}</span>
                                                </div>
                                            </div>
                                            <div className={`px-4 py-1.5 rounded-full border text-xs font-black tracking-tight ${getVerdictStyle(sub.verdict)}`}>
                                                {sub.verdict.replace(/_/g, ' ')}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </main>
                )}
            </div>
        </div>
    );
}

const SearchIcon = (props) => (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
);

const InfoItem = ({ icon, text }) => (
    <div className="flex items-center text-sm text-gray-400">
        <span className="mr-3 text-cyan-500">{icon}</span>
        {text}
    </div>
);

const StatCard = ({ title, val, detail, color }) => (
    <div className="bg-white/5 border border-white/10 p-6 rounded-3xl">
        <p className="text-[10px] font-black uppercase text-gray-500 tracking-widest mb-1">{title}</p>
        <p className={`text-3xl font-black ${color}`}>{val}</p>
        <p className="text-xs text-gray-500 mt-2">{detail}</p>
    </div>
);

export default Codeforces;