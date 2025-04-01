import React, { useState, useEffect, useMemo } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";
import PropTypes from "prop-types";
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

CodeExample.propTypes = {
  example: PropTypes.object.isRequired,
  isVisible: PropTypes.bool.isRequired,
  language: PropTypes.string.isRequired,
  code: PropTypes.string.isRequired,
  darkMode: PropTypes.bool.isRequired,
};

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

ToggleCodeButton.propTypes = {
  language: PropTypes.string.isRequired,
  isVisible: PropTypes.bool.isRequired,
  onClick: PropTypes.func.isRequired,
};

function Bit4() {
  const { darkMode, toggleTheme } = useTheme();
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
      title: "Single Number III",
      description: "Find the two numbers that appear only once in an array where every other number appears exactly twice.",
      approach: `
  1. Compute XOR of all numbers to get XOR of the two unique numbers
  2. Find a set bit in the XOR result (rightmost set bit)
  3. Partition numbers into two groups based on whether the bit is set
  4. Compute XOR within each group to find the two unique numbers`,
      algorithm: `
  • Time complexity: O(n)
  • Space complexity: O(1)
  • Efficient bit manipulation solution
  • Works for any number range
  • Handles negative numbers properly`,
      cppcode: `#include <iostream>
  #include <vector>
  using namespace std;
  
  vector<int> singleNumber(vector<int>& nums) {
      long diff = 0;
      for (int num : nums) diff ^= num;
      
      diff &= -diff;  // Get rightmost set bit
      
      vector<int> res(2, 0);
      for (int num : nums) {
          if ((num & diff) == 0) res[0] ^= num;
          else res[1] ^= num;
      }
      return res;
  }
  
  int main() {
      vector<int> nums = {1,2,1,3,2,5};
      vector<int> result = singleNumber(nums);
      cout << "Single numbers: " << result[0] << " " << result[1] << endl;
      return 0;
  }`,
      javacode: `public class SingleNumberIII {
      public static int[] singleNumber(int[] nums) {
          int diff = 0;
          for (int num : nums) diff ^= num;
          
          diff &= -diff;  // Get rightmost set bit
          
          int[] res = new int[2];
          for (int num : nums) {
              if ((num & diff) == 0) res[0] ^= num;
              else res[1] ^= num;
          }
          return res;
      }
      
      public static void main(String[] args) {
          int[] nums = {1,2,1,3,2,5};
          int[] result = singleNumber(nums);
          System.out.println("Single numbers: " + result[0] + " " + result[1]);
      }
  }`,
      pythoncode: `def single_number(nums):
      diff = 0
      for num in nums:
          diff ^= num
      
      diff &= -diff  # Get rightmost set bit
      
      res = [0, 0]
      for num in nums:
          if (num & diff) == 0:
              res[0] ^= num
          else:
              res[1] ^= num
      return res
  
  nums = [1,2,1,3,2,5]
  print("Single numbers:", single_number(nums))`,
      language: "cpp",
      javaLanguage: "java",
      pythonlanguage: "python",
      complexity: "Time Complexity: O(n), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/single-number-iii/"
    },
    {
      title: "Binary Number Palindrome Check (Bit Manipulation)",
      description: "Determine if the binary representation of a number is a palindrome using bit manipulation.",
      approach: `
  1. Find the number of bits in the number
  2. Compare the leftmost and rightmost bits
  3. Move inward until all bits are checked
  4. Return true if all corresponding bits match`,
      algorithm: `
  • Time complexity: O(log n) - proportional to number of bits
  • Space complexity: O(1)
  • Uses bit shifting operations
  • Doesn't require converting to string
  • Works for all positive integers`,
      cppcode: `#include <iostream>
  using namespace std;
  
  bool isBinaryPalindrome(int n) {
      if (n == 0) return true;
      
      int left = 1 << (31 - __builtin_clz(n));
      int right = 1;
      
      while (left > right) {
          if (((n & left) != 0) != ((n & right) != 0))
              return false;
          left >>= 1;
          right <<= 1;
      }
      return true;
  }
  
  int main() {
      int num;
      cout << "Enter a number: ";
      cin >> num;
      cout << (isBinaryPalindrome(num) ? "True" : "False") << endl;
      return 0;
  }`,
      javacode: `public class BinaryPalindrome {
      public static boolean isBinaryPalindrome(int n) {
          if (n == 0) return true;
          
          int left = 1 << (31 - Integer.numberOfLeadingZeros(n));
          int right = 1;
          
          while (left > right) {
              if (((n & left) != 0) != ((n & right) != 0))
                  return false;
              left >>= 1;
              right <<= 1;
          }
          return true;
      }
      
      public static void main(String[] args) {
          System.out.println(isBinaryPalindrome(9));  // 1001 is palindrome
      }
  }`,
      pythoncode: `def is_binary_palindrome(n):
      if n == 0:
          return True
      
      left = 1 << (n.bit_length() - 1)
      right = 1
      
      while left > right:
          if ((n & left) != 0) != ((n & right) != 0):
              return False
          left >>= 1
          right <<= 1
      return True
  
  num = int(input("Enter a number: "))
  print(is_binary_palindrome(num))`,
      language: "cpp",
      javaLanguage: "java",
      pythonlanguage: "python",
      complexity: "Time Complexity: O(log n), Space Complexity: O(1)",
      link: "https://www.geeksforgeeks.org/check-binary-representation-number-palindrome/"
    },
    {
      title: "Greatest Common Divisor (GCD)",
      description: "Find the greatest common divisor of two numbers using the Euclidean algorithm.",
      approach: `
  1. Use Euclidean algorithm
  2. While b is not zero:
     - Calculate remainder of a divided by b
     - Set a to b and b to remainder
  3. When b becomes zero, a contains the GCD`,
      algorithm: `
  • Time complexity: O(log(min(a,b)))
  • Space complexity: O(1)
  • Efficient mathematical algorithm
  • Works for both positive and negative numbers
  • Handles zero as input`,
      cppcode: `#include <iostream>
  using namespace std;
  
  int gcd(int a, int b) {
      while (b != 0) {
          int temp = b;
          b = a % b;
          a = temp;
      }
      return a;
  }
  
  int main() {
      int a, b;
      cout << "Enter two numbers: ";
      cin >> a >> b;
      cout << "GCD: " << gcd(a, b) << endl;
      return 0;
  }`,
      javacode: `public class GCD {
      public static int gcd(int a, int b) {
          while (b != 0) {
              int temp = b;
              b = a % b;
              a = temp;
          }
          return a;
      }
      
      public static void main(String[] args) {
          System.out.println("GCD: " + gcd(48, 18));
      }
  }`,
      pythoncode: `def gcd(a, b):
      while b:
          a, b = b, a % b
      return a
  
  a = int(input("Enter first number: "))
  b = int(input("Enter second number: "))
  print("GCD:", gcd(a, b))`,
      language: "cpp",
      javaLanguage: "java",
      pythonlanguage: "python",
      complexity: "Time Complexity: O(log(min(a,b))), Space Complexity: O(1)",
      link: "https://www.geeksforgeeks.org/euclidean-algorithms-basic-and-extended/"
    },
    {
      title: "Least Common Multiple (LCM)",
      description: "Find the least common multiple of two numbers using their GCD.",
      approach: `
  1. Calculate GCD of the two numbers
  2. Use formula: LCM(a,b) = (a*b)/GCD(a,b)
  3. Handle division carefully to avoid overflow
  4. Take absolute values if needed`,
      algorithm: `
  • Time complexity: O(log(min(a,b))) - same as GCD
  • Space complexity: O(1)
  • Efficient mathematical solution
  • Works for both positive and negative numbers
  • Handles zero as input (LCM is 0 if either number is 0)`,
      cppcode: `#include <iostream>
  #include <cmath>
  using namespace std;
  
  int gcd(int a, int b) {
      while (b != 0) {
          int temp = b;
          b = a % b;
          a = temp;
      }
      return a;
  }
  
  int lcm(int a, int b) {
      if (a == 0 || b == 0) return 0;
      return abs(a / gcd(a, b) * b);
  }
  
  int main() {
      int a, b;
      cout << "Enter two numbers: ";
      cin >> a >> b;
      cout << "LCM: " << lcm(a, b) << endl;
      return 0;
  }`,
      javacode: `public class LCM {
      public static int gcd(int a, int b) {
          while (b != 0) {
              int temp = b;
              b = a % b;
              a = temp;
          }
          return a;
      }
      
      public static int lcm(int a, int b) {
          if (a == 0 || b == 0) return 0;
          return Math.abs(a / gcd(a, b) * b);
      }
      
      public static void main(String[] args) {
          System.out.println("LCM: " + lcm(12, 18));
      }
  }`,
      pythoncode: `def gcd(a, b):
      while b:
          a, b = b, a % b
      return a
  
  def lcm(a, b):
      if a == 0 or b == 0:
          return 0
      return abs(a // gcd(a, b) * b)
  
  a = int(input("Enter first number: "))
  b = int(input("Enter second number: "))
  print("LCM:", lcm(a, b))`,
      language: "cpp",
      javaLanguage: "java",
      pythonlanguage: "python",
      complexity: "Time Complexity: O(log(min(a,b))), Space Complexity: O(1)",
      link: "https://www.geeksforgeeks.org/program-to-find-lcm-of-two-numbers/"
    },
    {
      title: "Nth Fibonacci Number",
      description: "Find the nth Fibonacci number efficiently using matrix exponentiation.",
      approach: `
  1. Handle base cases (n = 0 or 1)
  2. Use matrix exponentiation for O(log n) solution
  3. Represent Fibonacci sequence as matrix [[1,1],[1,0]]
  4. Raise matrix to power (n-1)
  5. Extract result from matrix`,
      algorithm: `
  • Time complexity: O(log n)
  • Space complexity: O(1)
  • Much faster than naive recursive solution
  • Works for large n values
  • Handles negative indices if needed`,
      cppcode: `#include <iostream>
  using namespace std;
  
  void multiply(int F[2][2], int M[2][2]) {
      int x = F[0][0] * M[0][0] + F[0][1] * M[1][0];
      int y = F[0][0] * M[0][1] + F[0][1] * M[1][1];
      int z = F[1][0] * M[0][0] + F[1][1] * M[1][0];
      int w = F[1][0] * M[0][1] + F[1][1] * M[1][1];
      
      F[0][0] = x; F[0][1] = y;
      F[1][0] = z; F[1][1] = w;
  }
  
  void power(int F[2][2], int n) {
      if (n == 0 || n == 1) return;
      int M[2][2] = {{1,1},{1,0}};
      
      power(F, n/2);
      multiply(F, F);
      
      if (n % 2 != 0)
          multiply(F, M);
  }
  
  int fibonacci(int n) {
      if (n == 0) return 0;
      int F[2][2] = {{1,1},{1,0}};
      power(F, n-1);
      return F[0][0];
  }
  
  int main() {
      int n;
      cout << "Enter n: ";
      cin >> n;
      cout << "Fibonacci number: " << fibonacci(n) << endl;
      return 0;
  }`,
      javacode: `public class Fibonacci {
      public static void multiply(int[][] F, int[][] M) {
          int x = F[0][0] * M[0][0] + F[0][1] * M[1][0];
          int y = F[0][0] * M[0][1] + F[0][1] * M[1][1];
          int z = F[1][0] * M[0][0] + F[1][1] * M[1][0];
          int w = F[1][0] * M[0][1] + F[1][1] * M[1][1];
          
          F[0][0] = x; F[0][1] = y;
          F[1][0] = z; F[1][1] = w;
      }
      
      public static void power(int[][] F, int n) {
          if (n == 0 || n == 1) return;
          int[][] M = {{1,1},{1,0}};
          
          power(F, n/2);
          multiply(F, F);
          
          if (n % 2 != 0)
              multiply(F, M);
      }
      
      public static int fibonacci(int n) {
          if (n == 0) return 0;
          int[][] F = {{1,1},{1,0}};
          power(F, n-1);
          return F[0][0];
      }
      
      public static void main(String[] args) {
          System.out.println("Fibonacci number: " + fibonacci(10));
      }
  }`,
      pythoncode: `def multiply(F, M):
      x = F[0][0] * M[0][0] + F[0][1] * M[1][0]
      y = F[0][0] * M[0][1] + F[0][1] * M[1][1]
      z = F[1][0] * M[0][0] + F[1][1] * M[1][0]
      w = F[1][0] * M[0][1] + F[1][1] * M[1][1]
      
      F[0][0] = x; F[0][1] = y
      F[1][0] = z; F[1][1] = w
  
  def power(F, n):
      if n == 0 or n == 1:
          return
      M = [[1,1],[1,0]]
      
      power(F, n // 2)
      multiply(F, F)
      
      if n % 2 != 0:
          multiply(F, M)
  
  def fibonacci(n):
      if n == 0:
          return 0
      F = [[1,1],[1,0]]
      power(F, n-1)
      return F[0][0]
  
  n = int(input("Enter n: "))
  print("Fibonacci number:", fibonacci(n))`,
      language: "cpp",
      javaLanguage: "java",
      pythonlanguage: "python",
      complexity: "Time Complexity: O(log n), Space Complexity: O(1)",
      link: "https://www.geeksforgeeks.org/matrix-exponentiation/"
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
        MAANG Hard Questions
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

export default Bit4;
