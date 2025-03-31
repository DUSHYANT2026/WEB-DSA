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

function Narray6() {
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
        title: "Two Sum",
        description: "Given an array of integers, return indices of the two numbers such that they add up to a specific target.",
        approach: `
1. Use a hash map to store value-index pairs
2. For each element, calculate complement (target - current)
3. Check if complement exists in hash map
4. If found, return current index and complement's index
5. If not found, store current value and index in hash map`,
        algorithm: `
• Time complexity: O(N) - single pass through array
• Space complexity: O(N) - for hash map storage
• Works for both sorted and unsorted arrays
• Efficient lookup using hash map`,
        code: `def two_sum(nums, target):
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []`,
        language: "python",
        complexity: "Time Complexity: O(N), Space Complexity: O(N)",
        link: "https://leetcode.com/problems/two-sum/"
    },
    {
        title: "Maximum Subarray (Kadane's Algorithm)",
        description: "Find the contiguous subarray which has the largest sum and return its sum.",
        approach: `
1. Initialize current and maximum sums with first element
2. Iterate through array starting from second element
3. For each element, decide whether to add to current subarray or start new subarray
4. Update maximum sum whenever current sum exceeds it`,
        algorithm: `
• Time complexity: O(N) - single pass through array
• Space complexity: O(1) - constant extra space
• Handles all negative numbers case
• Classic dynamic programming problem`,
        code: `def max_subarray(nums):
    if not nums:
        return 0
    
    max_sum = current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum`,
        language: "python",
        complexity: "Time Complexity: O(N), Space Complexity: O(1)",
        link: "https://leetcode.com/problems/maximum-subarray/"
    },
    {
        title: "Subarray Sum Equals K",
        description: "Find the total number of contiguous subarrays whose sum equals to a target value K.",
        approach: `
1. Use prefix sum with hash map to store sum frequencies
2. Initialize with sum 0 having one occurrence
3. For each element, calculate running sum
4. Check if (sum - K) exists in hash map
5. Add count of (sum - K) to result
6. Update hash map with current sum`,
        algorithm: `
• Time complexity: O(N) - single pass through array
• Space complexity: O(N) - for hash map storage
• Handles negative numbers in array
• Efficient count using prefix sums`,
        code: `def subarray_sum(nums, k):
    sum_map = {0: 1}
    current_sum = 0
    count = 0
    
    for num in nums:
        current_sum += num
        count += sum_map.get(current_sum - k, 0)
        sum_map[current_sum] = sum_map.get(current_sum, 0) + 1
    
    return count`,
        language: "python",
        complexity: "Time Complexity: O(N), Space Complexity: O(N)",
        link: "https://leetcode.com/problems/subarray-sum-equals-k/"
    },
    {
        title: "First Missing Positive",
        description: "Find the smallest missing positive integer in an unsorted array in O(n) time and constant space.",
        approach: `
1. Segregate positive numbers and ignore negatives
2. Use array indices as hash keys to mark presence
3. First pass: place each number in its correct position
4. Second pass: find first index where number doesn't match position`,
        algorithm: `
• Time complexity: O(N) - two passes through array
• Space complexity: O(1) - in-place modification
• Handles duplicates and large numbers
• Optimal for constant space requirement`,
        code: `def first_missing_positive(nums):
    n = len(nums)
    
    # Place each number in its correct position
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
    
    # Find first missing positive
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    
    return n + 1`,
        language: "python",
        complexity: "Time Complexity: O(N), Space Complexity: O(1)",
        link: "https://leetcode.com/problems/first-missing-positive/"
    },
    {
        title: "Product of Array Except Self",
        description: "Given an array nums, return an array where each element is the product of all elements except nums[i].",
        approach: `
1. Initialize result array with 1s
2. First pass (left to right): compute product of elements to the left
3. Second pass (right to left): multiply with product of elements to the right
4. Avoid division operation to handle zeros`,
        algorithm: `
• Time complexity: O(N) - two passes through array
• Space complexity: O(1) - output array doesn't count
• Handles zeros in input array
• Division-free solution`,
        code: `def product_except_self(nums):
    n = len(nums)
    result = [1] * n
    
    # Left pass
    left_product = 1
    for i in range(n):
        result[i] = left_product
        left_product *= nums[i]
    
    # Right pass
    right_product = 1
    for i in range(n-1, -1, -1):
        result[i] *= right_product
        right_product *= nums[i]
    
    return result`,
        language: "python",
        complexity: "Time Complexity: O(N), Space Complexity: O(1)",
        link: "https://leetcode.com/problems/product-of-array-except-self/"
    },
    {
        title: "Group Anagrams",
        description: "Given an array of strings, group anagrams together (words with same letters in different order).",
        approach: `
1. Use a dictionary to group anagrams
2. For each word, create a key by sorting its letters
3. Alternatively, use character count tuple as key
4. Add word to dictionary under its key
5. Return grouped values from dictionary`,
        algorithm: `
• Time complexity: O(N*KlogK) - N words with K being max length
• Space complexity: O(N*K) - for storing all strings
• Handles empty strings
• Efficient grouping with hash map`,
        code: `def group_anagrams(strs):
    from collections import defaultdict
    
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())`,
        language: "python",
        complexity: "Time Complexity: O(N*KlogK), Space Complexity: O(N*K)",
        link: "https://leetcode.com/problems/group-anagrams/"
    },
    {
        title: "Longest Palindromic Substring",
        description: "Find the longest substring which reads the same forwards and backwards in a given string.",
        approach: `
1. Expand around center for both odd and even length palindromes
2. For each character, expand to left and right while palindrome condition holds
3. Track longest palindrome found
4. Handle edge cases (empty string, single character)`,
        algorithm: `
• Time complexity: O(N²) - worst case for all characters as centers
• Space complexity: O(1) - constant extra space
• Optimal for longest palindrome detection
• Handles both odd and even length palindromes`,
        code: `def longest_palindrome(s):
    if not s:
        return ""
    
    start, end = 0, 0
    for i in range(len(s)):
        len1 = expand_around_center(s, i, i)  # Odd length
        len2 = expand_around_center(s, i, i+1)  # Even length
        max_len = max(len1, len2)
        if max_len > end - start:
            start = i - (max_len - 1) // 2
            end = i + max_len // 2
    
    return s[start:end+1]

def expand_around_center(s, left, right):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return right - left - 1`,
        language: "python",
        complexity: "Time Complexity: O(N²), Space Complexity: O(1)",
        link: "https://leetcode.com/problems/longest-palindromic-substring/"
    },
    {
        title: "Minimum Window Substring",
        description: "Find the minimum window in string S that contains all characters of string T.",
        approach: `
1. Use sliding window technique with two pointers
2. Maintain character frequency map for T
3. Expand right pointer until all characters are included
4. Contract left pointer to find minimum valid window
5. Track minimum window throughout process`,
        algorithm: `
• Time complexity: O(M+N) - where M and N are lengths of S and T
• Space complexity: O(1) - fixed size character maps
• Handles duplicate characters in T
• Optimal sliding window solution`,
        code: `def min_window(s, t):
    from collections import defaultdict
    
    if not t or not s:
        return ""
    
    target_counts = defaultdict(int)
    for char in t:
        target_counts[char] += 1
    
    required = len(target_counts)
    formed = 0
    window_counts = defaultdict(int)
    result = (float('inf'), None, None)
    left = 0
    
    for right, char in enumerate(s):
        if char in target_counts:
            window_counts[char] += 1
            if window_counts[char] == target_counts[char]:
                formed += 1
        
        while formed == required and left <= right:
            if right - left + 1 < result[0]:
                result = (right - left + 1, left, right)
            
            left_char = s[left]
            if left_char in target_counts:
                window_counts[left_char] -= 1
                if window_counts[left_char] < target_counts[left_char]:
                    formed -= 1
            left += 1
    
    return "" if result[0] == float('inf') else s[result[1]:result[2]+1]`,
        language: "python",
        complexity: "Time Complexity: O(M+N), Space Complexity: O(1)",
        link: "https://leetcode.com/problems/minimum-window-substring/"
    },
    {
        title: "Median of Two Sorted Arrays",
        description: "Find the median of two sorted arrays in O(log(min(m,n))) time complexity.",
        approach: `
1. Ensure first array is smaller for binary search efficiency
2. Perform binary search on the smaller array
3. Partition both arrays such that left halves contain median elements
4. Adjust partitions based on comparison of border elements
5. Calculate median based on even/odd total length`,
        algorithm: `
• Time complexity: O(log(min(m,n))) - binary search on smaller array
• Space complexity: O(1) - constant extra space
• Handles arrays of different lengths
• Optimal logarithmic solution`,
        code: `def find_median_sorted_arrays(nums1, nums2):
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    total = m + n
    
    while left <= right:
        partition1 = (left + right) // 2
        partition2 = (total + 1) // 2 - partition1
        
        max_left1 = float('-inf') if partition1 == 0 else nums1[partition1-1]
        min_right1 = float('inf') if partition1 == m else nums1[partition1]
        
        max_left2 = float('-inf') if partition2 == 0 else nums2[partition2-1]
        min_right2 = float('inf') if partition2 == n else nums2[partition2]
        
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            if total % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            else:
                return max(max_left1, max_left2)
        elif max_left1 > min_right2:
            right = partition1 - 1
        else:
            left = partition1 + 1
    
    raise ValueError("Input arrays are not sorted")`,
        language: "python",
        complexity: "Time Complexity: O(log(min(m,n))), Space Complexity: O(1)",
        link: "https://leetcode.com/problems/median-of-two-sorted-arrays/"
    },
    {
        title: "Word Search",
        description: "Given a 2D board and a word, determine if the word exists in the grid by adjacent cells (up, down, left, right).",
        approach: `
1. Use backtracking/DFS with pruning
2. For each cell matching first character, start search
3. Mark visited cells to prevent reuse
4. Recursively check all four directions
5. Unmark cells when backtracking`,
        algorithm: `
• Time complexity: O(M*N*4^L) - where L is word length
• Space complexity: O(L) - recursion stack depth
• Handles large boards with pruning
• Classic backtracking problem`,
        code: `def exist(board, word):
    if not board or not word:
        return False
    
    rows, cols = len(board), len(board[0])
    
    def dfs(i, j, index):
        if index == len(word):
            return True
        if i < 0 or i >= rows or j < 0 or j >= cols or board[i][j] != word[index]:
            return False
        
        temp = board[i][j]
        board[i][j] = '#'  # Mark as visited
        
        found = (dfs(i+1, j, index+1) or
                 dfs(i-1, j, index+1) or
                 dfs(i, j+1, index+1) or
                 dfs(i, j-1, index+1))
        
        board[i][j] = temp  # Backtrack
        return found
    
    for i in range(rows):
        for j in range(cols):
            if dfs(i, j, 0):
                return True
    
    return False`,
        language: "python",
        complexity: "Time Complexity: O(M*N*4^L), Space Complexity: O(L)",
        link: "https://leetcode.com/problems/word-search/"
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
        Most Asked MAANG Questions
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

export default Narray6;