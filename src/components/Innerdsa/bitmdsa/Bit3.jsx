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

function Bit3() {
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
      title: "Sum of Two Numbers",
      description: "Calculate the sum of two integers without using the + or - operators.",
      approach: `
  1. Use bitwise operations to calculate sum without carry
  2. Calculate carry separately
  3. Iterate until there is no carry left
  4. XOR gives sum without carry
  5. AND plus left shift gives carry`,
      algorithm: `
  • Time complexity: O(1) - fixed number of iterations (32/64)
  • Space complexity: O(1)
  • Works for both positive and negative numbers
  • Uses bitwise XOR and AND operations
  • Handles all integer ranges`,
      cppcode: `#include <iostream>
  using namespace std;
  
  int getSum(int a, int b) {
      while (b != 0) {
          unsigned carry = a & b;
          a = a ^ b;
          b = carry << 1;
      }
      return a;
  }
  
  int main() {
      int a, b;
      cout << "Enter two numbers: ";
      cin >> a >> b;
      cout << "Sum: " << getSum(a, b) << endl;
      return 0;
  }`,
      javacode: `public class SumWithoutOperators {
      public static int getSum(int a, int b) {
          while (b != 0) {
              int carry = a & b;
              a = a ^ b;
              b = carry << 1;
          }
          return a;
      }
      
      public static void main(String[] args) {
          int a = 5, b = 7;
          System.out.println("Sum: " + getSum(a, b));
      }
  }`,
      pythoncode: `def get_sum(a, b):
      mask = 0xFFFFFFFF
      while b != 0:
          carry = (a & b) & mask
          a = (a ^ b) & mask
          b = (carry << 1) & mask
      return a if a <= 0x7FFFFFFF else ~(a ^ mask)
  
  a = int(input("Enter first number: "))
  b = int(input("Enter second number: "))
  print("Sum:", get_sum(a, b))`,
      language: "cpp",
      javaLanguage: "java",
      pythonlanguage: "python",
      complexity: "Time Complexity: O(1), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/sum-of-two-integers/"
    },
    {
      title: "Single Number II",
      description: "Find the number that appears once in an array where every other number appears three times.",
      approach: `
  1. Initialize two variables to track bits
  2. For each number in array:
     - Update first tracker with XOR and mask
     - Update second tracker with XOR and mask
  3. The remaining bits in first tracker is the unique number`,
      algorithm: `
  • Time complexity: O(n)
  • Space complexity: O(1)
  • Efficient bit manipulation solution
  • Works for any number range
  • Handles negative numbers properly`,
      cppcode: `#include <iostream>
  #include <vector>
  using namespace std;
  
  int singleNumber(vector<int>& nums) {
      int ones = 0, twos = 0;
      for (int num : nums) {
          ones = (ones ^ num) & ~twos;
          twos = (twos ^ num) & ~ones;
      }
      return ones;
  }
  
  int main() {
      vector<int> nums = {2,2,3,2};
      cout << "Single number: " << singleNumber(nums) << endl;
      return 0;
  }`,
      javacode: `public class SingleNumberII {
      public static int singleNumber(int[] nums) {
          int ones = 0, twos = 0;
          for (int num : nums) {
              ones = (ones ^ num) & ~twos;
              twos = (twos ^ num) & ~ones;
          }
          return ones;
      }
      
      public static void main(String[] args) {
          int[] nums = {0,1,0,1,0,1,99};
          System.out.println("Single number: " + singleNumber(nums));
      }
  }`,
      pythoncode: `def single_number(nums):
      ones, twos = 0, 0
      for num in nums:
          ones = (ones ^ num) & ~twos
          twos = (twos ^ num) & ~ones
      return ones
  
  nums = [2,2,3,2]
  print("Single number:", single_number(nums))`,
      language: "cpp",
      javaLanguage: "java",
      pythonlanguage: "python",
      complexity: "Time Complexity: O(n), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/single-number-ii/"
    },
    {
      title: "Reverse Bits",
      description: "Reverse the bits of a given 32-bit unsigned integer.",
      approach: `
  1. Initialize result as 0
  2. For each bit in input:
     - Shift result left by 1
     - Add current bit of input
     - Shift input right by 1
  3. Return final reversed result`,
      algorithm: `
  • Time complexity: O(1) - fixed 32 iterations
  • Space complexity: O(1)
  • Uses bit shifting operations
  • Preserves leading zeros
  • Works for all 32-bit unsigned integers`,
      cppcode: `#include <iostream>
  using namespace std;
  
  uint32_t reverseBits(uint32_t n) {
      uint32_t result = 0;
      for (int i = 0; i < 32; i++) {
          result = (result << 1) | (n & 1);
          n >>= 1;
      }
      return result;
  }
  
  int main() {
      uint32_t num;
      cout << "Enter a number: ";
      cin >> num;
      cout << "Reversed bits: " << reverseBits(num) << endl;
      return 0;
  }`,
      javacode: `public class ReverseBits {
      public static int reverseBits(int n) {
          int result = 0;
          for (int i = 0; i < 32; i++) {
              result = (result << 1) | (n & 1);
              n >>>= 1;
          }
          return result;
      }
      
      public static void main(String[] args) {
          int num = 43261596;
          System.out.println("Reversed bits: " + reverseBits(num));
      }
  }`,
      pythoncode: `def reverse_bits(n):
      result = 0
      for _ in range(32):
          result = (result << 1) | (n & 1)
          n >>= 1
      return result
  
  num = int(input("Enter a number: "))
  print("Reversed bits:", reverse_bits(num))`,
      language: "cpp",
      javaLanguage: "java",
      pythonlanguage: "python",
      complexity: "Time Complexity: O(1), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/reverse-bits/"
    },
    {
      title: "Power of Two",
      description: "Determine if a given integer is a power of two.",
      approach: `
  1. Check if number is positive
  2. Use bitwise AND to verify power of two
  3. A power of two has exactly one bit set
  4. n & (n-1) will be 0 for powers of two`,
      algorithm: `
  • Time complexity: O(1)
  • Space complexity: O(1)
  • Single line solution possible
  • Works for all integer values
  • Handles edge case (n = 0)`,
      cppcode: `#include <iostream>
  using namespace std;
  
  bool isPowerOfTwo(int n) {
      return n > 0 && (n & (n - 1)) == 0;
  }
  
  int main() {
      int num;
      cout << "Enter a number: ";
      cin >> num;
      cout << (isPowerOfTwo(num) ? "True" : "False") << endl;
      return 0;
  }`,
      javacode: `public class PowerOfTwo {
      public static boolean isPowerOfTwo(int n) {
          return n > 0 && (n & (n - 1)) == 0;
      }
      
      public static void main(String[] args) {
          int num = 16;
          System.out.println(isPowerOfTwo(num));
      }
  }`,
      pythoncode: `def is_power_of_two(n):
      return n > 0 and (n & (n - 1)) == 0
  
  num = int(input("Enter a number: "))
  print(is_power_of_two(num))`,
      language: "cpp",
      javaLanguage: "java",
      pythonlanguage: "python",
      complexity: "Time Complexity: O(1), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/power-of-two/"
    },
    {
      title: "Divide Two Integers",
      description: "Divide two integers without using multiplication, division or mod operator.",
      approach: `
  1. Handle overflow case (dividend = INT_MIN, divisor = -1)
  2. Determine sign of result
  3. Work with absolute values
  4. Subtract largest possible multiple of divisor
  5. Use bit shifting for efficient multiplication
  6. Accumulate result`,
      algorithm: `
  • Time complexity: O(log n)
  • Space complexity: O(1)
  • Efficient bit manipulation
  • Handles 32-bit integer range
  • Works for both positive and negative numbers`,
      cppcode: `#include <iostream>
  #include <climits>
  using namespace std;
  
  int divide(int dividend, int divisor) {
      if (dividend == INT_MIN && divisor == -1) return INT_MAX;
      
      long dvd = labs(dividend), dvs = labs(divisor);
      int sign = (dividend < 0) ^ (divisor < 0) ? -1 : 1;
      long res = 0;
      
      while (dvd >= dvs) {
          long temp = dvs, multiple = 1;
          while (dvd >= (temp << 1)) {
              temp <<= 1;
              multiple <<= 1;
          }
          dvd -= temp;
          res += multiple;
      }
      
      return sign * res;
  }
  
  int main() {
      int dividend, divisor;
      cout << "Enter dividend and divisor: ";
      cin >> dividend >> divisor;
      cout << "Result: " << divide(dividend, divisor) << endl;
      return 0;
  }`,
      javacode: `public class DivideIntegers {
      public static int divide(int dividend, int divisor) {
          if (dividend == Integer.MIN_VALUE && divisor == -1)
              return Integer.MAX_VALUE;
              
          long dvd = Math.abs((long)dividend), dvs = Math.abs((long)divisor);
          int sign = (dividend < 0) ^ (divisor < 0) ? -1 : 1;
          long res = 0;
          
          while (dvd >= dvs) {
              long temp = dvs, multiple = 1;
              while (dvd >= (temp << 1)) {
                  temp <<= 1;
                  multiple <<= 1;
              }
              dvd -= temp;
              res += multiple;
          }
          
          return (int)(sign * res);
      }
      
      public static void main(String[] args) {
          System.out.println(divide(10, 3));
      }
  }`,
      pythoncode: `def divide(dividend, divisor):
      if dividend == -2**31 and divisor == -1:
          return 2**31 - 1
      
      dvd, dvs = abs(dividend), abs(divisor)
      sign = -1 if (dividend < 0) ^ (divisor < 0) else 1
      res = 0
      
      while dvd >= dvs:
          temp, multiple = dvs, 1
          while dvd >= (temp << 1):
              temp <<= 1
              multiple <<= 1
          dvd -= temp
          res += multiple
      
      return sign * res
  
  dividend = int(input("Enter dividend: "))
  divisor = int(input("Enter divisor: "))
  print("Result:", divide(dividend, divisor))`,
      language: "cpp",
      javaLanguage: "java",
      pythonlanguage: "python",
      complexity: "Time Complexity: O(log n), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/divide-two-integers/"
    },
    {
      title: "Bitwise AND of Numbers Range",
      description: "Find the bitwise AND of all numbers in the range [left, right].",
      approach: `
  1. Initialize a mask
  2. While left and right are not equal:
     - Right shift both left and right
     - Increment mask
  3. Shift left back by mask positions
  4. Return the result`,
      algorithm: `
  • Time complexity: O(1) - max 32 iterations
  • Space complexity: O(1)
  • Finds common prefix of binary representations
  • Efficient bit manipulation
  • Works for all positive integers`,
      cppcode: `#include <iostream>
  using namespace std;
  
  int rangeBitwiseAnd(int left, int right) {
      int shift = 0;
      while (left != right) {
          left >>= 1;
          right >>= 1;
          shift++;
      }
      return left << shift;
  }
  
  int main() {
      int left, right;
      cout << "Enter range [left right]: ";
      cin >> left >> right;
      cout << "Bitwise AND: " << rangeBitwiseAnd(left, right) << endl;
      return 0;
  }`,
      javacode: `public class BitwiseAND {
      public static int rangeBitwiseAnd(int left, int right) {
          int shift = 0;
          while (left != right) {
              left >>= 1;
              right >>= 1;
              shift++;
          }
          return left << shift;
      }
      
      public static void main(String[] args) {
          System.out.println(rangeBitwiseAnd(5, 7));
      }
  }`,
      pythoncode: `def range_bitwise_and(left, right):
      shift = 0
      while left != right:
          left >>= 1
          right >>= 1
          shift += 1
      return left << shift
  
  left = int(input("Enter left: "))
  right = int(input("Enter right: "))
  print("Bitwise AND:", range_bitwise_and(left, right))`,
      language: "cpp",
      javaLanguage: "java",
      pythonlanguage: "python",
      complexity: "Time Complexity: O(1), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/bitwise-and-of-numbers-range/"
    },
    {
      title: "Gray Code",
      description: "Generate the n-bit Gray code sequence where successive numbers differ by exactly one bit.",
      approach: `
  1. Start with base case (n=1: [0,1])
  2. For each subsequent n:
     - Take previous sequence and prepend 0
     - Take reversed previous sequence and prepend 1
     - Concatenate these two sequences
  3. Return final sequence`,
      algorithm: `
  • Time complexity: O(2^n)
  • Space complexity: O(2^n)
  • Uses mirroring property of Gray code
  • Each iteration doubles the sequence size
  • Works for any positive integer n`,
      cppcode: `#include <iostream>
  #include <vector>
  using namespace std;
  
  vector<int> grayCode(int n) {
      vector<int> result = {0};
      for (int i = 0; i < n; i++) {
          int size = result.size();
          for (int j = size - 1; j >= 0; j--) {
              result.push_back(result[j] | (1 << i));
          }
      }
      return result;
  }
  
  int main() {
      int n;
      cout << "Enter n: ";
      cin >> n;
      vector<int> codes = grayCode(n);
      for (int num : codes) {
          cout << num << " ";
      }
      return 0;
  }`,
      javacode: `import java.util.ArrayList;
  import java.util.List;
  
  public class GrayCode {
      public static List<Integer> grayCode(int n) {
          List<Integer> result = new ArrayList<>();
          result.add(0);
          for (int i = 0; i < n; i++) {
              int size = result.size();
              for (int j = size - 1; j >= 0; j--) {
                  result.add(result.get(j) | (1 << i));
              }
          }
          return result;
      }
      
      public static void main(String[] args) {
          System.out.println(grayCode(3));
      }
  }`,
      pythoncode: `def gray_code(n):
      result = [0]
      for i in range(n):
          result += [x | (1 << i) for x in reversed(result)]
      return result
  
  n = int(input("Enter n: "))
  print("Gray code:", gray_code(n))`,
      language: "cpp",
      javaLanguage: "java",
      pythonlanguage: "python",
      complexity: "Time Complexity: O(2^n), Space Complexity: O(2^n)",
      link: "https://leetcode.com/problems/gray-code/"
    },
    {
      title: "Nth Digit",
      description: "Find the nth digit of the infinite sequence 123456789101112131415...",
      approach: `
  1. Determine the range (1-digit, 2-digit numbers etc.)
  2. Find the exact number containing the nth digit
  3. Find the specific digit within that number
  4. Calculate based on digit length and offset`,
      algorithm: `
  • Time complexity: O(log n)
  • Space complexity: O(1)
  • Efficient digit counting
  • Handles very large n values
  • Works by determining digit ranges`,
      cppcode: `#include <iostream>
  #include <cmath>
  using namespace std;
  
  int findNthDigit(int n) {
      int len = 1;
      long count = 9;
      int start = 1;
      
      while (n > len * count) {
          n -= len * count;
          len++;
          count *= 10;
          start *= 10;
      }
      
      int num = start + (n - 1) / len;
      string s = to_string(num);
      return s[(n - 1) % len] - '0';
  }
  
  int main() {
      int n;
      cout << "Enter n: ";
      cin >> n;
      cout << "Nth digit: " << findNthDigit(n) << endl;
      return 0;
  }`,
      javacode: `public class NthDigit {
      public static int findNthDigit(int n) {
          int len = 1;
          long count = 9;
          int start = 1;
          
          while (n > len * count) {
              n -= len * count;
              len++;
              count *= 10;
              start *= 10;
          }
          
          int num = start + (n - 1) / len;
          return Character.getNumericValue(String.valueOf(num).charAt((n - 1) % len));
      }
      
      public static void main(String[] args) {
          System.out.println(findNthDigit(11));
      }
  }`,
      pythoncode: `def find_nth_digit(n):
      len = 1
      count = 9
      start = 1
      
      while n > len * count:
          n -= len * count
          len += 1
          count *= 10
          start *= 10
      
      num = start + (n - 1) // len
      return int(str(num)[(n - 1) % len])
  
  n = int(input("Enter n: "))
  print("Nth digit:", find_nth_digit(n))`,
      language: "cpp",
      javaLanguage: "java",
      pythonlanguage: "python",
      complexity: "Time Complexity: O(log n), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/nth-digit/"
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
        Leetcode  Questions
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

export default Bit3;
