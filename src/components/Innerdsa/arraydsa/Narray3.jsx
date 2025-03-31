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

function Narray3() {
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
      title: "Binary Search",
      description: "Standard implementation to find a target in a sorted array.",
      approach: `1. Initialize left and right pointers
  2. Calculate mid index
  3. Compare mid element with target
  4. Adjust search range based on comparison`,
      algorithm: `• Time: O(log n)
  • Space: O(1)
  • Requires sorted input`,
      code: `function binarySearch(nums, target) {
    let left = 0, right = nums.length - 1;
    while (left <= right) {
      const mid = Math.floor((left + right) / 2);
      if (nums[mid] === target) return mid;
      if (nums[mid] < target) left = mid + 1;
      else right = mid - 1;
    }
    return -1;
  }`,
      language: "javascript",
      complexity: "O(log n) time, O(1) space",
      link: "https://leetcode.com/problems/binary-search/"
    },
    {
      title: "Search Insert Position",
      description: "Find the index where target should be inserted to maintain order.",
      approach: `1. Standard binary search
  2. Return left pointer if not found
  3. Left indicates insertion point`,
      algorithm: `• Same as binary search
  • Returns proper position for insertion`,
      code: `function searchInsert(nums, target) {
    let left = 0, right = nums.length - 1;
    while (left <= right) {
      const mid = Math.floor((left + right) / 2);
      if (nums[mid] === target) return mid;
      if (nums[mid] < target) left = mid + 1;
      else right = mid - 1;
    }
    return left;
  }`,
      language: "javascript",
      complexity: "O(log n) time, O(1) space",
      link: "https://leetcode.com/problems/search-insert-position/"
    },
    {
      title: "Count Occurrences in Sorted Array",
      description: "Count frequency of target in sorted array with duplicates.",
      approach: `1. Find first occurrence
  2. Find last occurrence
  3. Return count = last - first + 1`,
      algorithm: `• Two binary searches
  • Still logarithmic time`,
      code: `function countOccurrences(nums, target) {
    function findFirst() {
      let left = 0, right = nums.length - 1, res = -1;
      while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (nums[mid] >= target) {
          if (nums[mid] === target) res = mid;
          right = mid - 1;
        } else left = mid + 1;
      }
      return res;
    }
  
    function findLast() {
      let left = 0, right = nums.length - 1, res = -1;
      while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (nums[mid] <= target) {
          if (nums[mid] === target) res = mid;
          left = mid + 1;
        } else right = mid - 1;
      }
      return res;
    }
  
    const first = findFirst();
    return first === -1 ? 0 : findLast() - first + 1;
  }`,
      language: "javascript",
      complexity: "O(log n) time, O(1) space",
      link: "https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/"
    },
    {
      title: "Search in Rotated Sorted Array",
      description: "Search target in array that was sorted then rotated.",
      approach: `1. Check which half is properly sorted
  2. Determine if target is in sorted half
  3. Otherwise search other half`,
      algorithm: `• Modified binary search
  • Handles rotation point`,
      code: `function searchRotated(nums, target) {
    let left = 0, right = nums.length - 1;
    while (left <= right) {
      const mid = Math.floor((left + right) / 2);
      if (nums[mid] === target) return mid;
  
      // Left half is sorted
      if (nums[left] <= nums[mid]) {
        if (nums[left] <= target && target < nums[mid]) {
          right = mid - 1;
        } else {
          left = mid + 1;
        }
      } 
      // Right half is sorted
      else {
        if (nums[mid] < target && target <= nums[right]) {
          left = mid + 1;
        } else {
          right = mid - 1;
        }
      }
    }
    return -1;
  }`,
      language: "javascript",
      complexity: "O(log n) time, O(1) space",
      link: "https://leetcode.com/problems/search-in-rotated-sorted-array/"
    },
    {
      title: "Search in Rotated Sorted Array II",
      description: "Search in rotated array that may contain duplicates.",
      approach: `1. Handle duplicates by shrinking bounds
  2. Check which half is sorted
  3. Adjust search range accordingly`,
      algorithm: `• Worst case O(n)
  • Handles duplicates`,
      code: `function searchRotatedWithDups(nums, target) {
    let left = 0, right = nums.length - 1;
    while (left <= right) {
      const mid = Math.floor((left + right) / 2);
      if (nums[mid] === target) return true;
  
      // Handle duplicates
      if (nums[left] === nums[mid] && nums[right] === nums[mid]) {
        left++;
        right--;
      }
      // Left half sorted
      else if (nums[left] <= nums[mid]) {
        if (nums[left] <= target && target < nums[mid]) {
          right = mid - 1;
        } else {
          left = mid + 1;
        }
      }
      // Right half sorted
      else {
        if (nums[mid] < target && target <= nums[right]) {
          left = mid + 1;
        } else {
          right = mid - 1;
        }
      }
    }
    return false;
  }`,
      language: "javascript",
      complexity: "O(log n) average, O(n) worst with many duplicates",
      link: "https://leetcode.com/problems/search-in-rotated-sorted-array-ii/"
    },
    {
      title: "Find Peak Element",
      description: "Find any peak element where neighbors are smaller.",
      approach: `1. Compare mid with mid+1
  2. If increasing, peak is on right
  3. If decreasing, peak is on left`,
      algorithm: `• Binary search adaptation
  • Finds any local peak`,
      code: `function findPeakElement(nums) {
    let left = 0, right = nums.length - 1;
    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (nums[mid] < nums[mid + 1]) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    return left;
  }`,
      language: "javascript",
      complexity: "O(log n) time, O(1) space",
      link: "https://leetcode.com/problems/find-peak-element/"
    },
    {
      title: "Find Square Root",
      description: "Compute integer square root of a number.",
      approach: `1. Binary search between 0 and x
  2. Compare mid*mid with x
  3. Adjust search range`,
      algorithm: `• Binary search on answer space
  • Returns floor value`,
      code: `function mySqrt(x) {
    if (x < 2) return x;
    let left = 1, right = Math.floor(x / 2);
    while (left <= right) {
      const mid = Math.floor((left + right) / 2);
      const squared = mid * mid;
      if (squared === x) return mid;
      if (squared < x) left = mid + 1;
      else right = mid - 1;
    }
    return right;
  }`,
      language: "javascript",
      complexity: "O(log n) time, O(1) space",
      link: "https://leetcode.com/problems/sqrtx/"
    },
    {
      title: "Koko Eating Bananas",
      description: "Find minimum eating speed to finish all bananas in h hours.",
      approach: `1. Binary search possible speeds
  2. Calculate hours needed for each speed
  3. Find minimal valid speed`,
      algorithm: `• Binary search on answer
  • Requires helper function`,
      code: `function minEatingSpeed(piles, h) {
    let left = 1, right = Math.max(...piles);
    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (canEatAll(piles, mid, h)) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
    return left;
  }
  
  function canEatAll(piles, speed, h) {
    let hours = 0;
    for (const pile of piles) {
      hours += Math.ceil(pile / speed);
      if (hours > h) return false;
    }
    return true;
  }`,
      language: "javascript",
      complexity: "O(n log m) time where m is max pile, O(1) space",
      link: "https://leetcode.com/problems/koko-eating-bananas/"
    },
    {
      title: "Bouquets on Consecutive Days",
      description: "Find minimum days to make m bouquets requiring k adjacent flowers.",
      approach: `1. Binary search possible days
  2. Check if we can make enough bouquets
  3. Adjust search range`,
      algorithm: `• Binary search on day count
  • Sliding window check`,
      code: `function minDays(bloomDay, m, k) {
    if (m * k > bloomDay.length) return -1;
    
    let left = Math.min(...bloomDay);
    let right = Math.max(...bloomDay);
    
    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (canMakeBouquets(bloomDay, m, k, mid)) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
    return left;
  }
  
  function canMakeBouquets(bloomDay, m, k, day) {
    let bouquets = 0, flowers = 0;
    for (const bloom of bloomDay) {
      flowers = bloom <= day ? flowers + 1 : 0;
      if (flowers === k) {
        bouquets++;
        flowers = 0;
        if (bouquets === m) return true;
      }
    }
    return false;
  }`,
      language: "javascript",
      complexity: "O(n log d) where d is max bloom day, O(1) space",
      link: "https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/"
    },
    {
      title: "Kth Missing Positive Number",
      description: "Find the kth missing positive integer in increasing sequence.",
      approach: `1. Binary search to find where missing count >= k
  2. Calculate missing numbers before index
  3. Derive missing number`,
      algorithm: `• Binary search adaptation
  • Mathematical relationship`,
      code: `function findKthPositive(arr, k) {
    let left = 0, right = arr.length;
    while (left < right) {
      const mid = Math.floor((left + right) / 2);
      if (arr[mid] - mid - 1 < k) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    return left + k;
  }`,
      language: "javascript",
      complexity: "O(log n) time, O(1) space",
      link: "https://leetcode.com/problems/kth-missing-positive-number/"
    },
    {
      title: "Aggressive Cows",
      description: "Maximize minimum distance between cows in stalls.",
      approach: `1. Binary search possible distances
  2. Check if cows can be placed with min distance
  3. Adjust search range`,
      algorithm: `• Binary search on answer
  • Greedy placement check`,
      code: `function maxDistance(stalls, cows) {
    stalls.sort((a, b) => a - b);
    let left = 1, right = stalls[stalls.length - 1] - stalls[0];
    
    while (left <= right) {
      const mid = Math.floor((left + right) / 2);
      if (canPlaceCows(stalls, cows, mid)) {
        left = mid + 1;
      } else {
        right = mid - 1;
      }
    }
    return right;
  }
  
  function canPlaceCows(stalls, cows, minDist) {
    let count = 1, last = stalls[0];
    for (let i = 1; i < stalls.length; i++) {
      if (stalls[i] - last >= minDist) {
        last = stalls[i];
        count++;
        if (count === cows) return true;
      }
    }
    return false;
  }`,
      language: "javascript",
      complexity: "O(n log n) for sort + O(n log d) for search",
      link: "https://www.spoj.com/problems/AGGRCOW/"
    },
    {
      title: "Search in Row-wise and Column-wise Sorted Matrix",
      description: "Search target in matrix sorted row-wise and column-wise.",
      approach: `1. Start from top-right corner
  2. Move left if current > target
  3. Move down if current < target`,
      algorithm: `• Staircase search
  • O(m+n) time`,
      code: `function searchMatrix(matrix, target) {
    if (!matrix.length || !matrix[0].length) return false;
    let row = 0, col = matrix[0].length - 1;
    while (row < matrix.length && col >= 0) {
      if (matrix[row][col] === target) return true;
      if (matrix[row][col] > target) col--;
      else row++;
    }
    return false;
  }`,
      language: "javascript",
      complexity: "O(m + n) time, O(1) space",
      link: "https://leetcode.com/problems/search-a-2d-matrix-ii/"
    },
    {
      title: "Search in Strictly Sorted 2D Matrix",
      description: "Search target in matrix where first element of row > last element of previous row.",
      approach: `1. Treat matrix as 1D array
  2. Convert indices for 2D access
  3. Standard binary search`,
      algorithm: `• Binary search with index conversion
  • Efficient access`,
      code: `function search2DMatrix(matrix, target) {
    if (!matrix.length || !matrix[0].length) return false;
    const m = matrix.length, n = matrix[0].length;
    let left = 0, right = m * n - 1;
    
    while (left <= right) {
      const mid = Math.floor((left + right) / 2);
      const row = Math.floor(mid / n);
      const col = mid % n;
      if (matrix[row][col] === target) return true;
      if (matrix[row][col] < target) left = mid + 1;
      else right = mid - 1;
    }
    return false;
  }`,
      language: "javascript",
      complexity: "O(log(mn)) time, O(1) space",
      link: "https://leetcode.com/problems/search-a-2d-matrix/"
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
        Binary Search Problems with Solutions
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

export default Narray3;