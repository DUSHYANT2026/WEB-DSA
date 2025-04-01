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

function Bit1() {
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

  const codeExamples = useMemo(
    () => [
      {
        title: "Check if a Number is Odd",
        description: "Determine whether a given number is odd without using the modulus operator.",
        approach: `
    1. Use bitwise AND operator with 1
    2. If result is 1, number is odd
    3. If result is 0, number is even
    4. Works because odd numbers have LSB (least significant bit) set to 1`,
        algorithm: `
    • Time complexity: O(1)
    • Space complexity: O(1)
    • More efficient than modulus operation
    • Works for both positive and negative integers
    • Doesn't use division or modulus operators`,
        cppcode: `#include <iostream>
    using namespace std;
    
    bool isOdd(int num) {
        return num & 1;
    }
    
    int main() {
        int num;
        cout << "Enter a number: ";
        cin >> num;
        
        if (isOdd(num)) {
            cout << num << " is odd" << endl;
        } else {
            cout << num << " is even" << endl;
        }
        return 0;
    }`,
        javacode: `import java.util.Scanner;
    
    public class OddEven {
        public static boolean isOdd(int num) {
            return (num & 1) != 0;
        }
        
        public static void main(String[] args) {
            Scanner sc = new Scanner(System.in);
            System.out.print("Enter a number: ");
            int num = sc.nextInt();
            
            if (isOdd(num)) {
                System.out.println(num + " is odd");
            } else {
                System.out.println(num + " is even");
            }
        }
    }`,
        pythoncode: `def is_odd(num):
        return num & 1
    
    num = int(input("Enter a number: "))
    if is_odd(num):
        print(f"{num} is odd")
    else:
        print(f"{num} is even")`,
        language: "cpp",
        javaLanguage: "java",
        pythonlanguage: "python",
        complexity: "Time Complexity: O(1), Space Complexity: O(1)",
        link: "https://www.geeksforgeeks.org/check-if-a-number-is-odd-or-even-using-bitwise-operators/"
      },
      {
        title: "Count the Number of Set Bits",
        description: "Count the number of 1s in the binary representation of a number (Hamming weight).",
        approach: `
    1. Initialize count to 0
    2. While number is greater than 0:
       a. Perform bitwise AND with 1
       b. Add result to count
       c. Right shift number by 1
    3. Return count`,
        algorithm: `
    • Time complexity: O(log n) - runs in number of bits
    • Space complexity: O(1)
    • Works for both positive and negative numbers
    • More efficient methods exist (Brian Kernighan's algorithm)
    • Fundamental operation in many bit manipulation problems`,
        cppcode: `#include <iostream>
    using namespace std;
    
    int countSetBits(int num) {
        int count = 0;
        while (num) {
            count += num & 1;
            num >>= 1;
        }
        return count;
    }
    
    int main() {
        int num;
        cout << "Enter a number: ";
        cin >> num;
        cout << "Number of set bits: " << countSetBits(num) << endl;
        return 0;
    }`,
        javacode: `import java.util.Scanner;
    
    public class CountSetBits {
        public static int countSetBits(int num) {
            int count = 0;
            while (num != 0) {
                count += num & 1;
                num >>>= 1;  // Use unsigned right shift
            }
            return count;
        }
        
        public static void main(String[] args) {
            Scanner sc = new Scanner(System.in);
            System.out.print("Enter a number: ");
            int num = sc.nextInt();
            System.out.println("Number of set bits: " + countSetBits(num));
        }
    }`,
        pythoncode: `def count_set_bits(num):
        count = 0
        while num:
            count += num & 1
            num >>= 1
        return count
    
    num = int(input("Enter a number: "))
    print(f"Number of set bits: {count_set_bits(num)}")`,
        language: "cpp",
        javaLanguage: "java",
        pythonlanguage: "python",
        complexity: "Time Complexity: O(log n), Space Complexity: O(1)",
        link: "https://www.geeksforgeeks.org/count-set-bits-in-an-integer/"
      },
      {
        title: "Swap Two Numbers",
        description: "Swap two numbers without using a temporary variable.",
        approach: `
    1. Using arithmetic operations (addition and subtraction):
       a = a + b
       b = a - b
       a = a - b
    2. Using bitwise XOR operation:
       a = a ^ b
       b = a ^ b
       a = a ^ b`,
        algorithm: `
    • Time complexity: O(1)
    • Space complexity: O(1)
    • Avoids using temporary variable
    • Arithmetic method may overflow with large numbers
    • XOR method works for any primitive data type`,
        cppcode: `#include <iostream>
    using namespace std;
    
    void swapNumbers(int &a, int &b) {
        // Using arithmetic operations
        a = a + b;
        b = a - b;
        a = a - b;
        
        // Alternative using XOR (uncomment to use)
        // a = a ^ b;
        // b = a ^ b;
        // a = a ^ b;
    }
    
    int main() {
        int x = 5, y = 10;
        cout << "Before swap: x = " << x << ", y = " << y << endl;
        swapNumbers(x, y);
        cout << "After swap: x = " << x << ", y = " << y << endl;
        return 0;
    }`,
        javacode: `public class SwapNumbers {
        public static void swapNumbers(int[] nums) {
            // Using arithmetic operations
            nums[0] = nums[0] + nums[1];
            nums[1] = nums[0] - nums[1];
            nums[0] = nums[0] - nums[1];
            
            // Alternative using XOR (uncomment to use)
            // nums[0] = nums[0] ^ nums[1];
            // nums[1] = nums[0] ^ nums[1];
            // nums[0] = nums[0] ^ nums[1];
        }
        
        public static void main(String[] args) {
            int[] nums = {5, 10};
            System.out.println("Before swap: x = " + nums[0] + ", y = " + nums[1]);
            swapNumbers(nums);
            System.out.println("After swap: x = " + nums[0] + ", y = " + nums[1]);
        }
    }`,
        pythoncode: `def swap_numbers(a, b):
        # Using arithmetic operations
        a = a + b
        b = a - b
        a = a - b
        return a, b
        
        # Alternative using XOR (uncomment to use)
        # a = a ^ b
        # b = a ^ b
        # a = a ^ b
        # return a, b
    
    x, y = 5, 10
    print(f"Before swap: x = {x}, y = {y}")
    x, y = swap_numbers(x, y)
    print(f"After swap: x = {x}, y = {y}")`,
        language: "cpp",
        javaLanguage: "java",
        pythonlanguage: "python",
        complexity: "Time Complexity: O(1), Space Complexity: O(1)",
        link: "https://www.geeksforgeeks.org/swap-two-numbers-without-using-temporary-variable/"
      },
      {
        title: "Divide Two Integers Without Using Multiplication, Division or Mod Operator",
        description: "Divide two integers without using multiplication, division or mod operator, handling overflow cases.",
        approach: `
    1. Handle special cases (division by zero, INT_MIN / -1)
    2. Determine sign of result
    3. Work with absolute values
    4. Use bit shifting to find largest multiple
    5. Subtract multiples from dividend
    6. Apply sign to result`,
        algorithm: `
    • Time complexity: O(log n)
    • Space complexity: O(1)
    • Handles 32-bit integer range
    • Efficient using bit manipulation
    • Works for both positive and negative numbers`,
        cppcode: `#include <iostream>
    #include <climits>
    using namespace std;
    
    int divide(int dividend, int divisor) {
        if (dividend == INT_MIN && divisor == -1) {
            return INT_MAX;  // Handle overflow
        }
        
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
        cout << "Enter dividend: ";
        cin >> dividend;
        cout << "Enter divisor: ";
        cin >> divisor;
        
        cout << "Result: " << divide(dividend, divisor) << endl;
        return 0;
    }`,
        javacode: `public class IntegerDivision {
        public static int divide(int dividend, int divisor) {
            if (dividend == Integer.MIN_VALUE && divisor == -1) {
                return Integer.MAX_VALUE;  // Handle overflow
            }
            
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
            int dividend = 10, divisor = 3;
            System.out.println("Result: " + divide(dividend, divisor));
        }
    }`,
        pythoncode: `def divide(dividend, divisor):
        if dividend == -2**31 and divisor == -1:
            return 2**31 - 1  # Handle overflow
        
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
    print(f"Result: {divide(dividend, divisor)}")`,
        language: "cpp",
        javaLanguage: "java",
        pythonlanguage: "python",
        complexity: "Time Complexity: O(log n), Space Complexity: O(1)",
        link: "https://www.geeksforgeeks.org/divide-two-integers-without-using-multiplication-division-mod-operator/"
      },
      {
        title: "Find the Number That Appears Odd Number of Times",
        description: "Given an array where all numbers occur even number of times except one, find the odd occurring number.",
        approach: `
    1. Initialize result to 0
    2. XOR all elements in the array with result
    3. Even occurrences will cancel out (XOR with same number = 0)
    4. Final result will be the number with odd count`,
        algorithm: `
    • Time complexity: O(n)
    • Space complexity: O(1)
    • Efficient one-pass solution
    • Works for both positive and negative numbers
    • Uses XOR properties: 
      - a ^ a = 0
      - a ^ 0 = a
      - XOR is commutative and associative`,
        cppcode: `#include <iostream>
    #include <vector>
    using namespace std;
    
    int findOddOccurrence(const vector<int>& nums) {
        int res = 0;
        for (int num : nums) {
            res ^= num;
        }
        return res;
    }
    
    int main() {
        vector<int> nums = {1, 2, 3, 2, 3, 1, 3};
        cout << "The number appearing odd times is: " 
             << findOddOccurrence(nums) << endl;
        return 0;
    }`,
        javacode: `public class OddOccurrence {
        public static int findOddOccurrence(int[] nums) {
            int res = 0;
            for (int num : nums) {
                res ^= num;
            }
            return res;
        }
        
        public static void main(String[] args) {
            int[] nums = {1, 2, 3, 2, 3, 1, 3};
            System.out.println("The number appearing odd times is: " 
                              + findOddOccurrence(nums));
        }
    }`,
        pythoncode: `def find_odd_occurrence(nums):
        res = 0
        for num in nums:
            res ^= num
        return res
    
    nums = [1, 2, 3, 2, 3, 1, 3]
    print(f"The number appearing odd times is: {find_odd_occurrence(nums)}")`,
        language: "cpp",
        javaLanguage: "java",
        pythonlanguage: "python",
        complexity: "Time Complexity: O(n), Space Complexity: O(1)",
        link: "https://www.geeksforgeeks.org/find-the-number-occurring-odd-number-of-times/"
      }
    ],
    []
  );

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
        Bit Manipulation Questions
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

export default Bit1;
