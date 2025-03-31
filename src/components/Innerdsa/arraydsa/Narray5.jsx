import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";
import { ChevronDown, ChevronUp } from "react-feather";
import { useTheme } from "../../../ThemeContext.jsx";

const CodeExample = React.memo(
  ({ example, isVisible, language, code, darkMode }) => (
    <div
      className={`rounded-lg overflow-hidden border-2 ${getBorderColor(
        language
      )} transition-all duration-300 ${isVisible ? "block" : "hidden"}`}
    >
      <SyntaxHighlighter
        language={language}
        style={tomorrow}
        showLineNumbers
        wrapLines
        customStyle={{
          padding: "1.5rem",
          fontSize: "0.95rem",
          background: darkMode ? "#1e293b" : "#f9f9f9",
          borderRadius: "0.5rem",
        }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  )
);

const getBorderColor = (language) => {
  switch (language) {
    case "cpp":
      return "border-indigo-100 dark:border-indigo-900";
    case "java":
      return "border-green-100 dark:border-green-900";
    case "python":
      return "border-yellow-100 dark:border-yellow-900";
    default:
      return "border-gray-100 dark:border-gray-800";
  }
};

const getButtonColor = (language) => {
  switch (language) {
    case "cpp":
      return "from-pink-500 to-red-500 hover:from-pink-600 hover:to-red-600 dark:from-pink-600 dark:to-red-600 dark:hover:from-pink-700 dark:hover:to-red-700";
    case "java":
      return "from-green-500 to-teal-500 hover:from-green-600 hover:to-teal-600 dark:from-green-600 dark:to-teal-600 dark:hover:from-green-700 dark:hover:to-teal-700";
    case "python":
      return "from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600 dark:from-yellow-600 dark:to-orange-600 dark:hover:from-yellow-700 dark:hover:to-orange-700";
    default:
      return "from-gray-500 to-blue-500 hover:from-gray-600 hover:to-blue-600 dark:from-gray-600 dark:to-blue-600 dark:hover:from-gray-700 dark:hover:to-blue-700";
  }
};

const ToggleCodeButton = ({ language, isVisible, onClick }) => (
  <button
    onClick={onClick}
    className={`inline-block bg-gradient-to-r ${getButtonColor(
      language
    )} text-white px-4 py-2 sm:px-6 sm:py-3 rounded-lg transition-all transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-offset-2 ${
      language === "cpp"
        ? "focus:ring-pink-500 dark:focus:ring-pink-600"
        : language === "java"
        ? "focus:ring-green-500 dark:focus:ring-green-600"
        : "focus:ring-yellow-500 dark:focus:ring-yellow-600"
    }`}
    aria-expanded={isVisible}
    aria-controls={`${language}-code`}
  >
    {isVisible
      ? `Hide ${
          language === "cpp" ? "C++" : language === "java" ? "Java" : "Python"
        } Code`
      : `Show ${
          language === "cpp" ? "C++" : language === "java" ? "Java" : "Python"
        } Code`}
  </button>
);

function Narray5() {
  const { darkMode } = useTheme();
  const [visibleCodes, setVisibleCodes] = useState({
    cpp: null,
    java: null,
    python: null,
  });
  const [expandedSections, setExpandedSections] = useState({});

  const toggleCodeVisibility = (language, index) => {
    setVisibleCodes({
      cpp: language === "cpp" && visibleCodes.cpp !== index ? index : null,
      java: language === "java" && visibleCodes.java !== index ? index : null,
      python:
        language === "python" && visibleCodes.python !== index ? index : null,
    });
  };

  const toggleDetails = (index) => {
    setExpandedSections((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  const formatDescription = (desc) => {
    return desc.split("\n").map((paragraph, i) => (
      <p key={i} className="mb-4 whitespace-pre-line dark:text-gray-300">
        {paragraph}
      </p>
    ));
  };

  const codeExamples = [
    {
      title: "Maximum Points You Can Obtain from Cards",
      description: "Select k cards from either end of the array to maximize points.",
      approach: `1. Calculate total sum of first k cards
  2. Slide window from left to right
  3. Track maximum sum by swapping one card from each end`,
      algorithm: `• Sliding window technique
  • O(n) time complexity`,
      code: `function maxScore(cardPoints, k) {
    let leftSum = 0;
    for (let i = 0; i < k; i++) {
      leftSum += cardPoints[i];
    }
    
    let maxSum = leftSum;
    let rightSum = 0;
    
    for (let i = 0; i < k; i++) {
      rightSum += cardPoints[cardPoints.length - 1 - i];
      leftSum -= cardPoints[k - 1 - i];
      maxSum = Math.max(maxSum, leftSum + rightSum);
    }
    
    return maxSum;
  }`,
      language: "javascript",
      complexity: "O(k) time, O(1) space",
      link: "https://leetcode.com/problems/maximum-points-you-can-obtain-from-cards/"
    },
    {
      title: "Grid Game",
      description: "Two robots collect points from grid with optimal paths to minimize second robot's score.",
      approach: `1. Calculate prefix sums for both rows
  2. Find optimal turning point for first robot
  3. Minimize the maximum of remaining paths for second robot`,
      algorithm: `• Prefix sum array
  • O(n) time solution`,
      code: `function gridGame(grid) {
    const n = grid[0].length;
    const prefixTop = [...grid[0]];
    const prefixBottom = [...grid[1]];
    
    for (let i = 1; i < n; i++) {
      prefixTop[i] += prefixTop[i - 1];
      prefixBottom[i] += prefixBottom[i - 1];
    }
    
    let res = Infinity;
    
    for (let i = 0; i < n; i++) {
      const top = prefixTop[n - 1] - prefixTop[i];
      const bottom = i > 0 ? prefixBottom[i - 1] : 0;
      res = Math.min(res, Math.max(top, bottom));
    }
    
    return res;
  }`,
      language: "javascript",
      complexity: "O(n) time, O(n) space",
      link: "https://leetcode.com/problems/grid-game/"
    },
    {
      title: "Maximum Binary String After Change",
      description: "Maximize binary string by changing '00' to '10' or '10' to '01' operations.",
      approach: `1. Find first occurrence of '0'
  2. Count subsequent '0's
  3. Construct optimal string`,
      algorithm: `• Greedy construction
  • O(n) time solution`,
      code: `function maximumBinaryString(binary) {
    let firstZero = -1;
    let zeroCount = 0;
    
    for (let i = 0; i < binary.length; i++) {
      if (binary[i] === '0') {
        zeroCount++;
        if (firstZero === -1) firstZero = i;
      }
    }
    
    if (zeroCount <= 1) return binary;
    
    const res = binary.split('');
    for (let i = 0; i < res.length; i++) {
      res[i] = '1';
    }
    
    res[firstZero + zeroCount - 1] = '0';
    return res.join('');
  }`,
      language: "javascript",
      complexity: "O(n) time, O(n) space",
      link: "https://leetcode.com/problems/maximum-binary-string-after-change/"
    },
    {
      title: "Minimum Remove to Make Valid Parentheses",
      description: "Remove minimum parentheses to make string valid.",
      approach: `1. First pass: track balance and mark invalid ')'
  2. Second pass: mark excess '(' from end
  3. Build result skipping marked indices`,
      algorithm: `• Two-pass solution
  • O(n) time complexity`,
      code: `function minRemoveToMakeValid(s) {
    const stack = [];
    const toRemove = new Set();
    
    // First pass to mark invalid ')'
    for (let i = 0; i < s.length; i++) {
      if (s[i] === '(') {
        stack.push(i);
      } else if (s[i] === ')') {
        if (stack.length === 0) {
          toRemove.add(i);
        } else {
          stack.pop();
        }
      }
    }
    
    // Mark remaining '(' as invalid
    while (stack.length) {
      toRemove.add(stack.pop());
    }
    
    // Build result
    let res = [];
    for (let i = 0; i < s.length; i++) {
      if (!toRemove.has(i)) {
        res.push(s[i]);
      }
    }
    
    return res.join('');
  }`,
      language: "javascript",
      complexity: "O(n) time, O(n) space",
      link: "https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/"
    },
    {
      title: "Shortest Palindrome",
      description: "Add characters in front to make string a palindrome.",
      approach: `1. Find longest palindromic prefix
  2. Reverse remaining suffix and prepend`,
      algorithm: `• KMP algorithm adaptation
  • O(n) time solution`,
      code: `function shortestPalindrome(s) {
    const rev = s.split('').reverse().join('');
    const combined = s + '#' + rev;
    const lps = new Array(combined.length).fill(0);
    
    for (let i = 1; i < combined.length; i++) {
      let len = lps[i - 1];
      while (len > 0 && combined[i] !== combined[len]) {
        len = lps[len - 1];
      }
      if (combined[i] === combined[len]) {
        len++;
      }
      lps[i] = len;
    }
    
    const longest = lps[combined.length - 1];
    return rev.substring(0, rev.length - longest) + s;
  }`,
      language: "javascript",
      complexity: "O(n) time, O(n) space",
      link: "https://leetcode.com/problems/shortest-palindrome/"
    },
    {
      title: "Search a 2D Matrix II",
      description: "Search target in matrix sorted row-wise and column-wise.",
      approach: `1. Start from top-right corner
  2. Move left if current > target
  3. Move down if current < target`,
      algorithm: `• Staircase search
  • O(m + n) time`,
      code: `function searchMatrix(matrix, target) {
    if (!matrix.length || !matrix[0].length) return false;
    
    let row = 0;
    let col = matrix[0].length - 1;
    
    while (row < matrix.length && col >= 0) {
      if (matrix[row][col] === target) return true;
      if (matrix[row][col] > target) {
        col--;
      } else {
        row++;
      }
    }
    
    return false;
  }`,
      language: "javascript",
      complexity: "O(m + n) time, O(1) space",
      link: "https://leetcode.com/problems/search-a-2d-matrix-ii/"
    },
    {
      title: "Split Array Largest Sum",
      description: "Split array into m subarrays to minimize largest sum.",
      approach: `1. Binary search possible sums
  2. Check if array can be split with current sum
  3. Adjust search range`,
      algorithm: `• Binary search on answer
  • O(n log s) time`,
      code: `function splitArray(nums, m) {
    let left = Math.max(...nums);
    let right = nums.reduce((a, b) => a + b, 0);
    
    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (canSplit(nums, m, mid)) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
    
    return left;
  }
  
  function canSplit(nums, m, maxSum) {
    let sum = 0;
    let count = 1;
    
    for (const num of nums) {
      sum += num;
      if (sum > maxSum) {
        sum = num;
        count++;
        if (count > m) return false;
      }
    }
    
    return true;
  }`,
      language: "javascript",
      complexity: "O(n log s) time, O(1) space",
      link: "https://leetcode.com/problems/split-array-largest-sum/"
    },
    {
      title: "Path With Minimum Effort",
      description: "Find path with minimum maximum effort between adjacent cells.",
      approach: `1. Binary search possible effort values
  2. Check if path exists with BFS/DFS
  3. Adjust search range`,
      algorithm: `• Binary search + BFS
  • O(mn log H) time`,
      code: `function minimumEffortPath(heights) {
    const rows = heights.length;
    const cols = heights[0].length;
    let left = 0;
    let right = 0;
    
    // Find max possible effort
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        if (i > 0) right = Math.max(right, Math.abs(heights[i][j] - heights[i-1][j]));
        if (j > 0) right = Math.max(right, Math.abs(heights[i][j] - heights[i][j-1]));
      }
    }
    
    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (canReachEnd(heights, mid)) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
    
    return left;
  }
  
  function canReachEnd(heights, maxEffort) {
    const rows = heights.length;
    const cols = heights[0].length;
    const visited = new Array(rows).fill().map(() => new Array(cols).fill(false));
    const queue = [[0, 0]];
    const dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]];
    
    visited[0][0] = true;
    
    while (queue.length) {
      const [i, j] = queue.shift();
      if (i === rows - 1 && j === cols - 1) return true;
      
      for (const [di, dj] of dirs) {
        const ni = i + di;
        const nj = j + dj;
        if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && !visited[ni][nj]) {
          if (Math.abs(heights[ni][nj] - heights[i][j]) <= maxEffort) {
            visited[ni][nj] = true;
            queue.push([ni, nj]);
          }
        }
      }
    }
    
    return false;
  }`,
      language: "javascript",
      complexity: "O(mn log H) time, O(mn) space",
      link: "https://leetcode.com/problems/path-with-minimum-effort/"
    },
    {
      title: "Longest Increasing Path in a Matrix",
      description: "Find longest strictly increasing path in matrix moving adjacent.",
      approach: `1. DFS with memoization
  2. Cache results for each cell
  3. Explore all directions`,
      algorithm: `• Memoization
  • O(mn) time`,
      code: `function longestIncreasingPath(matrix) {
    if (!matrix.length || !matrix[0].length) return 0;
    
    const rows = matrix.length;
    const cols = matrix[0].length;
    const memo = new Array(rows).fill().map(() => new Array(cols).fill(0));
    const dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]];
    let maxLen = 0;
    
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        maxLen = Math.max(maxLen, dfs(i, j));
      }
    }
    
    return maxLen;
    
    function dfs(i, j) {
      if (memo[i][j] !== 0) return memo[i][j];
      
      let max = 1;
      for (const [di, dj] of dirs) {
        const ni = i + di;
        const nj = j + dj;
        if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && matrix[ni][nj] > matrix[i][j]) {
          max = Math.max(max, 1 + dfs(ni, nj));
        }
      }
      
      memo[i][j] = max;
      return max;
    }
  }`,
      language: "javascript",
      complexity: "O(mn) time, O(mn) space",
      link: "https://leetcode.com/problems/longest-increasing-path-in-a-matrix/"
    },
    {
      title: "Bricks Falling When Hit",
      description: "Determine how many bricks fall after each hit.",
      approach: `1. Reverse process (add hits back)
  2. Union-Find data structure
  3. Count connected components`,
      algorithm: `• Reverse Union-Find
  • O(h * α(mn)) time`,
      code: `function hitBricks(grid, hits) {
    const rows = grid.length;
    const cols = grid[0].length;
    const dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]];
    
    // Mark hits
    const copy = grid.map(row => [...row]);
    for (const [i, j] of hits) {
      copy[i][j] = 0;
    }
    
    // Initialize Union-Find with virtual roof node
    const size = rows * cols + 1;
    const parent = Array(size).fill().map((_, i) => i);
    const rank = Array(size).fill(1);
    
    function find(u) {
      if (parent[u] !== u) parent[u] = find(parent[u]);
      return parent[u];
    }
    
    function union(u, v) {
      const rootU = find(u);
      const rootV = find(v);
      if (rootU === rootV) return;
      
      if (rank[rootU] > rank[rootV]) {
        parent[rootV] = rootU;
        rank[rootU] += rank[rootV];
      } else {
        parent[rootU] = rootV;
        rank[rootV] += rank[rootU];
      }
    }
    
    // Connect top row to roof (size-1)
    const roof = rows * cols;
    for (let j = 0; j < cols; j++) {
      if (copy[0][j] === 1) {
        union(j, roof);
      }
    }
    
    // Connect remaining cells
    for (let i = 1; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        if (copy[i][j] === 1) {
          if (copy[i-1][j] === 1) union(i*cols + j, (i-1)*cols + j);
          if (j > 0 && copy[i][j-1] === 1) union(i*cols + j, i*cols + j-1);
        }
      }
    }
    
    // Process hits in reverse
    const result = Array(hits.length).fill(0);
    for (let k = hits.length - 1; k >= 0; k--) {
      const [i, j] = hits[k];
      if (grid[i][j] === 0) continue;
      
      const pos = i * cols + j;
      const before = rank[find(roof)];
      
      // Reconnect if top row
      if (i === 0) {
        union(j, roof);
      }
      
      // Check neighbors
      for (const [di, dj] of dirs) {
        const ni = i + di;
        const nj = j + dj;
        if (ni >= 0 && ni < rows && nj >= 0 && nj < cols && copy[ni][nj] === 1) {
          union(pos, ni * cols + nj);
        }
      }
      
      copy[i][j] = 1;
      const after = rank[find(roof)];
      result[k] = Math.max(0, after - before - 1);
    }
    
    return result;
  }`,
      language: "javascript",
      complexity: "O(h * α(mn)) time, O(mn) space",
      link: "https://leetcode.com/problems/bricks-falling-when-hit/"
    }
  ];

  return (
    <div
      className={`container mx-auto px-4 sm:px-6 py-12 rounded-2xl shadow-xl max-w-7xl ${
        darkMode
          ? "bg-gradient-to-br from-gray-900 to-gray-800"
          : "bg-gradient-to-br from-indigo-50 to-purple-50"
      }`}
    >
      <h1
        className={`text-4xl sm:text-5xl md:text-6xl font-extrabold text-center text-transparent bg-clip-text ${
          darkMode
            ? "bg-gradient-to-r from-indigo-400 to-purple-400"
            : "bg-gradient-to-r from-indigo-600 to-purple-600"
        } mb-8 sm:mb-12`}
      >
        Most Asked Leetcode Questions
      </h1>

      <div className="space-y-8">
        {codeExamples.map((example, index) => (
          <article
            key={index}
            className={`p-6 sm:p-8 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 border ${
              darkMode
                ? "bg-gray-800 border-gray-700"
                : "bg-white border-indigo-100"
            }`}
            aria-labelledby={`algorithm-${index}-title`}
          >
            <header className="mb-6">
              <button
                onClick={() => toggleDetails(index)}
                className="w-full flex justify-between items-center focus:outline-none"
              >
                <h2
                  id={`algorithm-${index}-title`}
                  className={`text-2xl sm:text-3xl font-bold text-left ${
                    darkMode ? "text-indigo-300" : "text-indigo-800"
                  }`}
                >
                  {example.title}
                </h2>
                <span
                  className={darkMode ? "text-indigo-400" : "text-indigo-600"}
                >
                  {expandedSections[index] ? (
                    <ChevronUp size={24} />
                  ) : (
                    <ChevronDown size={24} />
                  )}
                </span>
              </button>

              {expandedSections[index] && (
                <div className="space-y-4 mt-4">
                  <div
                    className={`p-4 sm:p-6 rounded-lg shadow-sm border ${
                      darkMode
                        ? "bg-gray-700 border-gray-600"
                        : "bg-gray-50 border-gray-100"
                    }`}
                  >
                    <h3
                      className={`font-bold mb-3 text-lg ${
                        darkMode ? "text-gray-300" : "text-gray-800"
                      }`}
                    >
                      Description
                    </h3>
                    <div
                      className={`${
                        darkMode ? "text-gray-300" : "text-gray-700"
                      } font-medium leading-relaxed`}
                    >
                      {formatDescription(example.description)}
                    </div>
                  </div>

                  <div
                    className={`p-4 sm:p-6 rounded-lg shadow-sm border ${
                      darkMode
                        ? "bg-blue-900 border-blue-800"
                        : "bg-blue-50 border-blue-100"
                    }`}
                  >
                    <h3
                      className={`font-bold mb-3 text-lg ${
                        darkMode ? "text-blue-300" : "text-blue-800"
                      }`}
                    >
                      Approach
                    </h3>
                    <div
                      className={`${
                        darkMode ? "text-blue-300" : "text-blue-800"
                      } font-semibold leading-relaxed whitespace-pre-line`}
                    >
                      {formatDescription(example.approach)}
                    </div>
                  </div>

                  <div
                    className={`p-4 sm:p-6 rounded-lg shadow-sm border ${
                      darkMode
                        ? "bg-green-900 border-green-800"
                        : "bg-green-50 border-green-100"
                    }`}
                  >
                    <h3
                      className={`font-bold mb-3 text-lg ${
                        darkMode ? "text-green-300" : "text-green-800"
                      }`}
                    >
                      Algorithm Characteristics
                    </h3>
                    <div
                      className={`${
                        darkMode ? "text-green-300" : "text-green-800"
                      } font-semibold leading-relaxed whitespace-pre-line`}
                    >
                      {formatDescription(example.algorithm)}
                    </div>
                  </div>
                </div>
              )}

              <p
                className={`font-semibold mt-4 ${
                  darkMode ? "text-gray-300" : "text-gray-800"
                }`}
              >
                <span
                  className={`font-bold ${
                    darkMode ? "text-indigo-400" : "text-indigo-700"
                  }`}
                >
                  Complexity:
                </span>{" "}
                {example.complexity}
              </p>
            </header>

            <div className="flex flex-wrap gap-3 mb-6">
              <a
                href={example.link}
                target="_blank"
                rel="noopener noreferrer"
                className={`inline-block bg-gradient-to-r ${
                  darkMode
                    ? "from-indigo-500 to-purple-500 hover:from-indigo-600 hover:to-purple-600"
                    : "from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700"
                } text-white px-4 py-2 sm:px-6 sm:py-3 rounded-lg transition-all transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2`}
              >
                View Problem
              </a>

              <ToggleCodeButton
                language="cpp"
                isVisible={visibleCodes.cpp === index}
                onClick={() => toggleCodeVisibility("cpp", index)}
              />

              <ToggleCodeButton
                language="java"
                isVisible={visibleCodes.java === index}
                onClick={() => toggleCodeVisibility("java", index)}
              />

              <ToggleCodeButton
                language="python"
                isVisible={visibleCodes.python === index}
                onClick={() => toggleCodeVisibility("python", index)}
              />
            </div>

            <CodeExample
              example={example}
              isVisible={visibleCodes.cpp === index}
              language="cpp"
              code={example.cppcode}
              darkMode={darkMode}
            />

            <CodeExample
              example={example}
              isVisible={visibleCodes.java === index}
              language="java"
              code={example.javacode}
              darkMode={darkMode}
            />

            <CodeExample
              example={example}
              isVisible={visibleCodes.python === index}
              language="python"
              code={example.pythoncode}
              darkMode={darkMode}
            />
          </article>
        ))}
      </div>
    </div>
  );
}

export default Narray5;