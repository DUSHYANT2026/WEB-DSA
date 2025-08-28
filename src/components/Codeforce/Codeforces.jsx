import React, { useState } from "react";

function Codeforces() {
    const [handle, setHandle] = useState("");
    const [userData, setUserData] = useState(null);
    const [submissions, setSubmissions] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const fetchCodeforcesData = async () => {
        if (!handle.trim()) {
            setError("Please enter a Codeforces handle.");
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const [userResponse, submissionsResponse] = await Promise.all([
                fetch(`https://codeforces.com/api/user.info?handles=${handle}`),
                fetch(`https://codeforces.com/api/user.status?handle=${handle}&from=1&count=50`)
            ]);

            const userJson = await userResponse.json();
            const submissionsJson = await submissionsResponse.json();

            if (userJson.status !== 'OK') {
                throw new Error(userJson.comment || "User not found.");
            }
            if (submissionsJson.status !== 'OK') {
                throw new Error(submissionsJson.comment || "Could not fetch submissions.");
            }
            
            setUserData(userJson.result[0]);
            setSubmissions(submissionsJson.result);

        } catch (err) {
            setError(err.message);
            setUserData(null);
            setSubmissions([]);
        } finally {
            setLoading(false);
        }
    };

    const getVerdictColor = (verdict) => {
        if (verdict === 'OK') return 'text-green-400';
        if (verdict === 'WRONG_ANSWER') return 'text-red-400';
        if (verdict === 'TIME_LIMIT_EXCEEDED') return 'text-yellow-400';
        return 'text-gray-400';
    };
    
    const getRatingColor = (rating) => {
        if (rating < 1200) return 'text-gray-400';
        if (rating < 1400) return 'text-green-400';
        if (rating < 1600) return 'text-cyan-400';
        if (rating < 1900) return 'text-blue-400';
        if (rating < 2100) return 'text-purple-400';
        if (rating < 2400) return 'text-yellow-400';
        return 'text-red-400';
    };


    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-800 text-white p-4 md:p-8">
            <div className="max-w-7xl mx-auto">
                {/* Header Section */}
                <header className="text-center mb-12">
                    <h1 className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-500 mb-2">
                        Codeforces Profile Explorer
                    </h1>
                    <p className="text-gray-300 max-w-lg mx-auto">
                        Explore Codeforces profiles, submission stats, and more.
                    </p>
                </header>

                {/* Search Section */}
                <div className="bg-gray-800/50 backdrop-blur-md p-6 rounded-2xl shadow-2xl max-w-2xl mx-auto mb-12 border border-gray-700/50">
                    <div className="flex flex-col md:flex-row gap-4">
                        <input
                            type="text"
                            placeholder="Enter Codeforces Handle"
                            value={handle}
                            onChange={(e) => setHandle(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && fetchCodeforcesData()}
                            className="flex-grow p-4 rounded-xl bg-gray-700/50 border border-gray-600 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/50 outline-none transition-all text-white placeholder-gray-400"
                        />
                        <button
                            onClick={fetchCodeforcesData}
                            disabled={loading}
                            className={`p-4 rounded-xl font-semibold transition-all duration-300 shadow-lg ${loading ? 'bg-cyan-600/50 cursor-not-allowed' : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 hover:shadow-xl hover:scale-[1.02]'} flex items-center justify-center`}
                        >
                            {loading ? (
                                <>
                                    <svg className="animate-spin h-5 w-5 mr-2 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Fetching...
                                </>
                            ) : 'Fetch Profile'}
                        </button>
                    </div>
                </div>

                {/* Loading & Error States */}
                {loading && (
                    <div className="flex justify-center my-12">
                        <div className="animate-pulse flex flex-col items-center">
                            <div className="h-32 w-32 bg-gray-700 rounded-full mb-4"></div>
                            <div className="h-6 w-48 bg-gray-700 rounded mb-2"></div>
                            <div className="h-4 w-64 bg-gray-700 rounded"></div>
                        </div>
                    </div>
                )}

                {error && (
                    <div className="max-w-2xl mx-auto bg-red-900/50 border border-red-700 rounded-xl p-6 mb-12 backdrop-blur-sm text-center animate-fade-in">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mx-auto text-red-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                        <p className="text-xl font-medium">{error}</p>
                        <p className="text-gray-300 mt-2">Please check the handle and try again.</p>
                    </div>
                )}

                {/* User Profile Section */}
                {userData && (
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-16">
                        {/* Profile Card */}
                        <div className="bg-gray-800/60 backdrop-blur-md rounded-2xl shadow-2xl p-6 border border-gray-700/50 lg:col-span-1 transform transition-all duration-500 hover:scale-[1.01]">
                            <div className="flex flex-col items-center">
                                <a href={`https://codeforces.com/profile/${userData.handle}`} target="_blank" rel="noopener noreferrer">
                                    <img
                                        src={userData.titlePhoto}
                                        alt="Codeforces Profile"
                                        className="w-40 h-40 rounded-full border-4 border-gradient-to-r from-cyan-500 to-blue-600 shadow-xl mb-6 transition-all duration-300 hover:scale-105"
                                    />
                                </a>
                                <h2 className={`text-2xl font-bold text-center mb-1 ${getRatingColor(userData.rating)}`}>{userData.handle}</h2>
                                <p className="text-gray-400 mb-4">{userData.rank}</p>
                                
                                <div className="grid grid-cols-2 gap-4 w-full mb-6">
                                    <div className="bg-gray-700/40 rounded-lg p-4 text-center">
                                        <p className={`text-2xl font-bold ${getRatingColor(userData.rating)}`}>{userData.rating || 'N/A'}</p>
                                        <p className="text-gray-400 text-sm">Rating</p>
                                    </div>
                                    <div className="bg-gray-700/40 rounded-lg p-4 text-center">
                                        <p className={`text-2xl font-bold ${getRatingColor(userData.maxRating)}`}>{userData.maxRating || 'N/A'}</p>
                                        <p className="text-gray-400 text-sm">Max Rating</p>
                                    </div>
                                </div>
                                
                                {(userData.firstName || userData.lastName) && (
                                     <p className="text-gray-300 text-center mb-2">{userData.firstName} {userData.lastName}</p>
                                )}
                                {userData.country && (
                                    <div className="flex items-center text-gray-400 mb-2">
                                        <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
                                        {userData.country}
                                    </div>
                                )}
                                 {userData.organization && (
                                    <p className="text-gray-400 text-sm mt-2">
                                       Organization: {userData.organization}
                                    </p>
                                )}
                                <p className="text-gray-400 text-sm mt-4">
                                    Registered {new Date(userData.registrationTimeSeconds * 1000).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}
                                </p>
                            </div>
                        </div>

                        {/* Submissions */}
                        <div className="lg:col-span-2 space-y-8">
                            {submissions.length > 0 && (
                                <div className="bg-gray-800/60 backdrop-blur-md rounded-2xl shadow-2xl p-6 border border-gray-700/50">
                                    <h3 className="text-xl font-bold mb-6 pb-2 border-b border-gray-700/50 flex items-center">
                                        <svg className="w-6 h-6 mr-2 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
                                        Recent Submissions
                                    </h3>
                                    <div className="space-y-4 max-h-96 overflow-y-auto pr-2">
                                        {submissions.map((sub) => (
                                            <div key={sub.id} className="bg-gray-700/30 hover:bg-gray-700/50 rounded-xl p-4 border border-gray-700/50 transition-all duration-300">
                                                <div className="flex justify-between items-center">
                                                    <a href={`https://codeforces.com/contest/${sub.contestId}/problem/${sub.problem.index}`} target="_blank" rel="noopener noreferrer" className="font-semibold hover:underline">
                                                        {sub.problem.name}
                                                    </a>
                                                    <span className={`font-bold ${getVerdictColor(sub.verdict)}`}>
                                                        {sub.verdict.replace(/_/g, ' ')}
                                                    </span>
                                                </div>
                                                <div className="text-sm text-gray-400 mt-2 flex justify-between">
                                                     <span>{sub.programmingLanguage}</span>
                                                     <span>{new Date(sub.creationTimeSeconds * 1000).toLocaleString()}</span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default Codeforces;
