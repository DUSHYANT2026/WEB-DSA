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

function Narray4() {
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
        title: "Spiral Matrix from Array",
        description: "Create a spiral matrix of N*M size from a given 1D array, filling elements in a clockwise spiral pattern.",
        approach: `
1. Initialize an empty N x M matrix
2. Define boundaries (top, bottom, left, right)
3. Traverse the array while filling matrix in layers:
   - Left to right along top boundary
   - Top to bottom along right boundary
   - Right to left along bottom boundary
   - Bottom to top along left boundary
4. Adjust boundaries after completing each layer`,
        algorithm: `
• Time complexity: O(N*M) - must fill every element
• Space complexity: O(N*M) - for the resulting matrix
• Works for any rectangular matrix dimensions
• Maintains original array order in spiral pattern`,
        code: `def spiral_matrix_from_array(arr, rows, cols):
    if len(arr) != rows * cols:
        return []
    
    matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    top, bottom = 0, rows - 1
    left, right = 0, cols - 1
    index = 0
    
    while top <= bottom and left <= right:
        # Fill top row
        for i in range(left, right + 1):
            matrix[top][i] = arr[index]
            index += 1
        top += 1
        
        # Fill right column
        for i in range(top, bottom + 1):
            matrix[i][right] = arr[index]
            index += 1
        right -= 1
        
        if top <= bottom:
            # Fill bottom row
            for i in range(right, left - 1, -1):
                matrix[bottom][i] = arr[index]
                index += 1
            bottom -= 1
        
        if left <= right:
            # Fill left column
            for i in range(bottom, top - 1, -1):
                matrix[i][left] = arr[index]
                index += 1
            left += 1
    
    return matrix`,
        language: "python",
        complexity: "Time Complexity: O(N*M), Space Complexity: O(N*M)",
        link: "https://www.geeksforgeeks.org/print-a-given-matrix-in-spiral-form/"
    },
    {
        title: "Rotate Matrix 90 Degrees Clockwise",
        description: "Rotate an N x N matrix by 90 degrees in clockwise direction without using extra space (in-place rotation).",
        approach: `
1. Transpose the matrix (swap rows and columns)
2. Reverse each row of the transposed matrix
3. For non-square matrices, create new matrix with different approach`,
        algorithm: `
• Time complexity: O(N²) - must touch all elements
• Space complexity: O(1) - in-place rotation
• Works only for square matrices
• Efficient for in-memory operations`,
        code: `def rotate_90_clockwise(matrix):
    n = len(matrix)
    # Transpose matrix
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # Reverse each row
    for row in matrix:
        row.reverse()
    return matrix`,
        language: "python",
        complexity: "Time Complexity: O(N²), Space Complexity: O(1)",
        link: "https://leetcode.com/problems/rotate-image/"
    },
    {
        title: "Rotate Matrix 180 Degrees Counter-Clockwise",
        description: "Rotate an N x N matrix by 180 degrees in counter-clockwise direction using efficient in-place operations.",
        approach: `
1. Reverse each row of the matrix
2. Then reverse the matrix column-wise (first row becomes last)
3. Alternatively, rotate 90 degrees twice`,
        algorithm: `
• Time complexity: O(N²) - must process all elements
• Space complexity: O(1) - in-place rotation
• Preserves matrix dimensions
• Equivalent to upside-down flip`,
        code: `def rotate_180_counterclockwise(matrix):
    n = len(matrix)
    # Reverse each row
    for row in matrix:
        row.reverse()
    
    # Reverse the matrix vertically
    matrix.reverse()
    return matrix`,
        language: "python",
        complexity: "Time Complexity: O(N²), Space Complexity: O(1)",
        link: "https://www.geeksforgeeks.org/rotate-matrix-180-degree/"
    },
    {
        title: "Search in Sorted Matrix",
        description: "Search for a target value in a matrix where each row and column is sorted in ascending order (Binary Search approach).",
        approach: `
1. Start from top-right corner (or bottom-left)
2. If current element == target: return position
3. If current element > target: move left
4. If current element < target: move down
5. Continue until element found or boundaries exceeded`,
        algorithm: `
• Time complexity: O(N+M) - worst case
• Space complexity: O(1) - no extra space
• Works for row-wise and column-wise sorted matrices
• More efficient than binary search for 2D matrices`,
        code: `def search_sorted_matrix(matrix, target):
    if not matrix:
        return [-1, -1]
    
    rows, cols = len(matrix), len(matrix[0])
    row, col = 0, cols - 1  # Start from top-right corner
    
    while row < rows and col >= 0:
        if matrix[row][col] == target:
            return [row, col]
        elif matrix[row][col] > target:
            col -= 1  # Move left
        else:
            row += 1  # Move down
    
    return [-1, -1]`,
        language: "python",
        complexity: "Time Complexity: O(N+M), Space Complexity: O(1)",
        link: "https://leetcode.com/problems/search-a-2d-matrix-ii/"
    },
    {
        title: "Set Matrix Zeroes",
        description: "If an element in an M x N matrix is 0, set its entire row and column to 0, do it in-place.",
        approach: `
1. First pass: mark rows and columns containing zeros
2. Use first row and column as markers to save space
3. Second pass: set elements to zero based on markers
4. Handle first row and column separately`,
        algorithm: `
• Time complexity: O(M*N) - must check all elements
• Space complexity: O(1) - in-place modification
• Efficient space usage by reusing matrix borders
• Preserves non-zero elements when possible`,
        code: `def set_zeroes(matrix):
    if not matrix:
        return
    
    rows, cols = len(matrix), len(matrix[0])
    first_row_has_zero = any(matrix[0][j] == 0 for j in range(cols))
    first_col_has_zero = any(matrix[i][0] == 0 for i in range(rows))
    
    # Mark zeros on first row and column
    for i in range(1, rows):
        for j in range(1, cols):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0
    
    # Set zeros based on marks
    for i in range(1, rows):
        for j in range(1, cols):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    
    # Handle first row and column
    if first_row_has_zero:
        for j in range(cols):
            matrix[0][j] = 0
    if first_col_has_zero:
        for i in range(rows):
            matrix[i][0] = 0`,
        language: "python",
        complexity: "Time Complexity: O(M*N), Space Complexity: O(1)",
        link: "https://leetcode.com/problems/set-matrix-zeroes/"
    },
    {
        title: "Generate Matrix with Given Row/Column Sums",
        description: "Construct a matrix where each row sums to given row sums and each column sums to given column sums.",
        approach: `
1. Initialize matrix with zeros
2. For each cell (i,j), set value to min(row_sum[i], col_sum[j])
3. Subtract this value from both row_sum[i] and col_sum[j]
4. Proceed until all sums are satisfied
5. Works for non-negative integers`,
        algorithm: `
• Time complexity: O(M*N) - must fill each cell
• Space complexity: O(M*N) - for the resulting matrix
• Greedy approach works for non-negative sums
• May have multiple valid solutions`,
        code: `def generate_matrix(row_sum, col_sum):
    rows, cols = len(row_sum), len(col_sum)
    matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            min_val = min(row_sum[i], col_sum[j])
            matrix[i][j] = min_val
            row_sum[i] -= min_val
            col_sum[j] -= min_val
    
    return matrix`,
        language: "python",
        complexity: "Time Complexity: O(M*N), Space Complexity: O(M*N)",
        link: "https://www.geeksforgeeks.org/find-a-matrix-with-given-row-and-column-sums/"
    },
    {
        title: "Make Matrix Beautiful",
        description: "A matrix is beautiful if all rows and columns sums are equal. Find minimum operations to make matrix beautiful by incrementing elements.",
        approach: `
1. Calculate maximum of all row sums and column sums
2. The target beautiful sum is this maximum value
3. For each row/column not matching target, increment elements
4. Count total operations needed`,
        algorithm: `
• Time complexity: O(M*N) - must check all elements
• Space complexity: O(M+N) - for storing sums
• Minimizes operations by targeting max sum
• Preserves matrix values when possible`,
        code: `def make_matrix_beautiful(matrix):
    if not matrix:
        return 0
    
    rows, cols = len(matrix), len(matrix[0])
    row_sums = [sum(row) for row in matrix]
    col_sums = [sum(matrix[i][j] for i in range(rows)) for j in range(cols)]
    
    max_sum = max(max(row_sums), max(col_sums))
    operations = 0
    
    for i in range(rows):
        for j in range(cols):
            diff = min(max_sum - row_sums[i], max_sum - col_sums[j])
            matrix[i][j] += diff
            row_sums[i] += diff
            col_sums[j] += diff
            operations += diff
    
    return operations`,
        language: "python",
        complexity: "Time Complexity: O(M*N), Space Complexity: O(M+N)",
        link: "https://www.geeksforgeeks.org/make-matrix-beautiful-incrementing-rows-and-columns/"
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
        Matrix Problems with Solutions
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

export default Narray4;