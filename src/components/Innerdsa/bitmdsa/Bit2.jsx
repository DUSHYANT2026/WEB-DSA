import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";
import { ChevronDown } from "react-feather";
import { useTheme } from "../../../ThemeContext.jsx";

const formatDescription = (desc, darkMode) => {
  if (Array.isArray(desc)) {
    return (
      <ul
        className={`list-disc pl-6 ${
          darkMode ? "text-gray-300" : "text-gray-700"
        }`}
      >
        {desc.map((item, i) => (
          <li key={i} className="mb-2">
            {item}
          </li>
        ))}
      </ul>
    );
  }
  return desc.split("\n").map((paragraph, i) => (
    <p key={i} className="mb-4 whitespace-pre-line">
      {paragraph}
    </p>
  ));
};

const CodeExample = React.memo(
  ({ example, isVisible, language, code, darkMode }) => (
    <div
      className={`rounded-lg overflow-hidden border-2 ${getBorderColor(
        language,
        darkMode
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

const getBorderColor = (language, darkMode) => {
  const base = darkMode ? "border-gray-700" : "border-gray-100";
  switch (language) {
    case "cpp":
      return darkMode ? "border-blue-900" : "border-blue-100";
    case "java":
      return darkMode ? "border-red-900" : "border-red-100";
    case "python":
      return darkMode ? "border-yellow-900" : "border-yellow-100";
    default:
      return base;
  }
};

const LanguageLogo = ({ language, size = 24, darkMode }) => {
  const baseClasses = "rounded-md p-1 flex items-center justify-center";

  const getGradient = (language) => {
    switch (language) {
      case "cpp":
        return darkMode
          ? "bg-gradient-to-br from-blue-900 to-blue-600"
          : "bg-gradient-to-br from-blue-500 to-blue-700";
      case "java":
        return darkMode
          ? "bg-gradient-to-br from-red-800 to-red-600"
          : "bg-gradient-to-br from-red-500 to-red-700";
      case "python":
        return darkMode
          ? "bg-gradient-to-br from-yellow-700 to-yellow-600"
          : "bg-gradient-to-br from-yellow-400 to-yellow-600";
      default:
        return darkMode
          ? "bg-gradient-to-br from-gray-700 to-gray-600"
          : "bg-gradient-to-br from-gray-400 to-gray-600";
    }
  };

  const getLogo = (language) => {
    switch (language) {
      case "cpp":
        return (
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              width: "100%",
              height: "100%",
            }}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width={size * 1.5}
              height={size * 1.5}
              viewBox="0 0 48 48"
            >
              <path
                fill="#00549d"
                fillRule="evenodd"
                d="M22.903,3.286c0.679-0.381,1.515-0.381,2.193,0c3.355,1.883,13.451,7.551,16.807,9.434C42.582,13.1,43,13.804,43,14.566c0,3.766,0,15.101,0,18.867c0,0.762-0.418,1.466-1.097,1.847c-3.355,1.883-13.451,7.551-16.807,9.434c-0.679,0.381-1.515,0.381-2.193,0c-3.355-1.883-13.451-7.551-16.807-9.434C5.418,34.899,5,34.196,5,33.434c0-3.766,0-15.101,0-18.867c0-0.762,0.418-1.466,1.097-1.847C9.451,10.837,19.549,5.169,22.903,3.286z"
                clipRule="evenodd"
              />
              <path
                fill="#0086d4"
                fillRule="evenodd"
                d="M5.304,34.404C5.038,34.048,5,33.71,5,33.255c0-3.744,0-15.014,0-18.759c0-0.758,0.417-1.458,1.094-1.836c3.343-1.872,13.405-7.507,16.748-9.38c0.677-0.379,1.594-0.371,2.271,0.008c3.343,1.872,13.371,7.459,16.714,9.331c0.27,0.152,0.476,0.335,0.66,0.576L5.304,34.404z"
                clipRule="evenodd"
              />
              <path
                fill="#fff"
                fillRule="evenodd"
                d="M24,10c7.727,0,14,6.273,14,14s-6.273,14-14,14s-14-6.273-14-14S16.273,10,24,10z M24,17c3.863,0,7,3.136,7,7c0,3.863-3.137,7-7,7s-7-3.137-7-7C17,20.136,20.136,17,24,17z"
                clipRule="evenodd"
              />
              <path
                fill="#0075c0"
                fillRule="evenodd"
                d="M42.485,13.205c0.516,0.483,0.506,1.211,0.506,1.784c0,3.795-0.032,14.589,0.009,18.384c0.004,0.396-0.127,0.813-0.323,1.127L23.593,24L42.485,13.205z"
                clipRule="evenodd"
              />
              <path
                fill="#fff"
                fillRule="evenodd"
                d="M31 21H33V27H31zM38 21H40V27H38z"
                clipRule="evenodd"
              />
              <path
                fill="#fff"
                fillRule="evenodd"
                d="M29 23H35V25H29zM36 23H42V25H36z"
                clipRule="evenodd"
              />
            </svg>
          </div>
        );
        case "java":
          return (
            <div style={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              width: "100%",
              height: "100%"
            }}>
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width={size * 1.5}
                height={size * 1.5}
                viewBox="0 0 50 50"
              >
                <path
                  d="M 28.1875 0 C 30.9375 6.363281 18.328125 10.292969 17.15625 15.59375 C 16.082031 20.464844 24.648438 26.125 24.65625 26.125 C 23.355469 24.109375 22.398438 22.449219 21.09375 19.3125 C 18.886719 14.007813 34.535156 9.207031 28.1875 0 Z M 36.5625 8.8125 C 36.5625 8.8125 25.5 9.523438 24.9375 16.59375 C 24.6875 19.742188 27.847656 21.398438 27.9375 23.6875 C 28.011719 25.558594 26.0625 27.125 26.0625 27.125 C 26.0625 27.125 29.609375 26.449219 30.71875 23.59375 C 31.949219 20.425781 28.320313 18.285156 28.6875 15.75 C 29.039063 13.324219 36.5625 8.8125 36.5625 8.8125 Z M 19.1875 25.15625 C 19.1875 25.15625 9.0625 25.011719 9.0625 27.875 C 9.0625 30.867188 22.316406 31.089844 31.78125 29.25 C 31.78125 29.25 34.296875 27.519531 34.96875 26.875 C 28.765625 28.140625 14.625 28.28125 14.625 27.1875 C 14.625 26.179688 19.1875 25.15625 19.1875 25.15625 Z M 38.65625 25.15625 C 37.664063 25.234375 36.59375 25.617188 35.625 26.3125 C 37.90625 25.820313 39.84375 27.234375 39.84375 28.84375 C 39.84375 32.46875 34.59375 35.875 34.59375 35.875 C 34.59375 35.875 42.71875 34.953125 42.71875 29 C 42.71875 26.296875 40.839844 24.984375 38.65625 25.15625 Z M 16.75 30.71875 C 15.195313 30.71875 12.875 31.9375 12.875 33.09375 C 12.875 35.417969 24.5625 37.207031 33.21875 33.8125 L 30.21875 31.96875 C 24.351563 33.847656 13.546875 33.234375 16.75 30.71875 Z M 18.1875 35.9375 C 16.058594 35.9375 14.65625 37.222656 14.65625 38.1875 C 14.65625 41.171875 27.371094 41.472656 32.40625 38.4375 L 29.21875 36.40625 C 25.457031 37.996094 16.015625 38.238281 18.1875 35.9375 Z M 11.09375 38.625 C 7.625 38.554688 5.375 40.113281 5.375 41.40625 C 5.375 48.28125 40.875 47.964844 40.875 40.9375 C 40.875 39.769531 39.527344 39.203125 39.03125 38.9375 C 41.933594 45.65625 9.96875 45.121094 9.96875 41.15625 C 9.96875 40.253906 12.320313 39.390625 14.5 39.8125 L 12.65625 38.75 C 12.113281 38.667969 11.589844 38.636719 11.09375 38.625 Z M 44.625 43.25 C 39.226563 48.367188 25.546875 50.222656 11.78125 47.0625 C 25.542969 52.695313 44.558594 49.535156 44.625 43.25 Z"
                />
              </svg>
            </div>
          );case "python":
        return (
          <svg viewBox="0 0 128 128" width={size} height={size}>
            <path
              fill="#3776AB"
              d="M63.391 1.988c-4.222.02-8.252.379-11.8 1.007-10.45 1.846-12.346 5.71-12.346 12.837v9.411h24.693v3.137H29.977c-7.176 0-13.46 4.313-15.426 12.521-2.268 9.405-2.368 15.275 0 25.096 1.755 7.311 5.947 12.519 13.124 12.519h8.491V67.234c0-8.151 7.051-15.34 15.426-15.34h24.665c6.866 0 12.346-5.654 12.346-12.548V15.833c0-6.693-5.646-11.72-12.346-12.837-4.244-.706-8.645-1.027-12.866-1.008zM50.037 9.557c2.55 0 4.634 2.117 4.634 4.721 0 2.593-2.083 4.69-4.634 4.69-2.56 0-4.633-2.097-4.633-4.69-.001-2.604 2.073-4.721 4.633-4.721z"
              transform="translate(0 10.26)"
            ></path>
            <path
              fill="#FFDC41"
              d="M91.682 28.38v10.966c0 8.5-7.208 15.655-15.426 15.655H51.591c-6.756 0-12.346 5.783-12.346 12.549v23.515c0 6.691 5.818 10.628 12.346 12.547 7.816 2.283 16.221 2.713 24.665 0 6.216-1.801 12.346-5.423 12.346-12.547v-9.412H63.938v-3.138h37.012c7.176 0 9.852-5.005 12.348-12.519 2.678-8.084 2.491-15.174 0-25.096-1.774-7.145-5.161-12.521-12.348-12.521h-9.268zM77.809 87.927c2.561 0 4.634 2.097 4.634 4.692 0 2.602-2.074 4.719-4.634 4.719-2.55 0-4.633-2.117-4.633-4.719 0-2.595 2.083-4.692 4.633-4.692z"
              transform="translate(0 10.26)"
            ></path>
          </svg>
        );
      default:
        return null;
    }
  };

  return (
    <div className={`${baseClasses} ${getGradient(language)}`}>
      {getLogo(language)}
    </div>
  );
};

const getButtonColor = (language, darkMode) => {
  switch (language) {
    case "cpp":
      return darkMode
        ? "from-blue-300 to-blue-500 hover:from-blue-400 hover:to-blue-700"
        : "from-blue-400 to-blue-600 hover:from-blue-500 hover:to-blue-700";
    case "java":
      return darkMode
        ? "from-red-700 to-red-900 hover:from-red-800 hover:to-red-950"
        : "from-red-500 to-red-700 hover:from-red-600 hover:to-red-800";
    case "python":
      return darkMode
        ? "from-yellow-600 to-orange-600 hover:from-yellow-700 hover:to-orange-700"
        : "from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600";
    default:
      return darkMode
        ? "from-gray-600 to-blue-600 hover:from-gray-700 hover:to-blue-700"
        : "from-gray-500 to-blue-500 hover:from-gray-600 hover:to-blue-600";
  }
};

const CollapsibleSection = ({
  title,
  content,
  isExpanded,
  onToggle,
  darkMode,
  colorScheme,
}) => (
  <div className="group">
    <button
      onClick={onToggle}
      className={`w-full flex justify-between items-center focus:outline-none p-3 rounded-lg transition-all ${
        isExpanded
          ? `${colorScheme.bg} ${colorScheme.border} border`
          : "hover:bg-opacity-10 hover:bg-gray-500"
      }`}
      aria-expanded={isExpanded}
    >
      <div className="flex items-center">
        <span className={`mr-3 text-lg ${colorScheme.icon}`}>
          {isExpanded ? "▼" : "►"}
        </span>
        <h3 className={`font-bold text-lg ${colorScheme.text}`}>{title}</h3>
      </div>
      <span className={`transition-transform duration-200 ${colorScheme.icon}`}>
        <ChevronDown size={20} className={isExpanded ? "rotate-180" : ""} />
      </span>
    </button>

    {isExpanded && (
      <div
        className={`p-4 sm:p-6 rounded-lg border mt-1 transition-all duration-200 ${colorScheme.bg} ${colorScheme.border} animate-fadeIn`}
      >
        <div
          className={`${colorScheme.text} font-medium leading-relaxed space-y-3`}
        >
          {typeof content === "string" ? (
            <div className="prose prose-sm max-w-none">
              {content.split("\n").map((paragraph, i) => (
                <p key={i} className="mb-3 last:mb-0">
                  {paragraph}
                </p>
              ))}
            </div>
          ) : Array.isArray(content) ? (
            <ul className="space-y-2 list-disc pl-5 marker:text-opacity-60">
              {content.map((item, i) => (
                <li key={i} className="pl-2">
                  {item.includes(":") ? (
                    <>
                      <span className="font-semibold">
                        {item.split(":")[0]}:
                      </span>
                      {item.split(":").slice(1).join(":")}
                    </>
                  ) : (
                    item
                  )}
                </li>
              ))}
            </ul>
          ) : (
            content
          )}
        </div>
      </div>
    )}
  </div>
);

const ToggleCodeButton = ({ language, isVisible, onClick, darkMode }) => (
  <button
    onClick={onClick}
    className={`inline-flex items-center justify-center bg-gradient-to-br ${
      darkMode
        ? language === "cpp"
          ? "from-blue-900 to-blue-700 hover:from-blue-800 hover:to-blue-600"
          : language === "java"
          ? "from-red-900 to-red-700 hover:from-red-800 hover:to-red-600"
          : "from-yellow-800 to-yellow-600 hover:from-yellow-700 hover:to-yellow-500"
        : language === "cpp"
        ? "from-blue-600 to-blue-800 hover:from-blue-500 hover:to-blue-700"
        : language === "java"
        ? "from-red-600 to-red-800 hover:from-red-500 hover:to-red-700"
        : "from-yellow-500 to-yellow-700 hover:from-yellow-400 hover:to-yellow-600"
    } text-white font-medium px-4 py-1.5 rounded-lg transition-all transform hover:scale-[1.05] focus:outline-none focus:ring-2 ${
      language === "cpp"
        ? "focus:ring-blue-400"
        : language === "java"
        ? "focus:ring-red-400"
        : "focus:ring-yellow-400"
    } ${
      darkMode ? "focus:ring-offset-gray-900" : "focus:ring-offset-white"
    } shadow-md ${
      darkMode ? "shadow-gray-800/50" : "shadow-gray-500/40"
    } border ${darkMode ? "border-gray-700/50" : "border-gray-400/50"}`}
    aria-expanded={isVisible}
    aria-controls={`${language}-code`}
  >
    <LanguageLogo
      language={language}
      size={18}
      darkMode={darkMode}
      className="mr-2"
    />
    {language === "cpp" ? "C++" : language === "java" ? "Java" : "Python"}
  </button>
);

function Bit1() {
  const { darkMode } = useTheme();
  const [visibleCode, setVisibleCode] = useState({
    index: null,
    language: null,
  });
  const [expandedSections, setExpandedSections] = useState({});

  const toggleCodeVisibility = (language, index) => {
    setVisibleCode((prev) => {
      // If clicking the same code that's already open, close it
      if (prev.index === index && prev.language === language) {
        return { index: null, language: null };
      }
      // Otherwise open the new code
      return { index, language };
    });
  };

  const toggleDetails = (index, section) => {
    setExpandedSections((prev) => ({
      ...prev,
      [`${index}-${section}`]: !prev[`${index}-${section}`],
    }));
  };

  const codeExamples =  [
      {
        title: "Print Prime Factors of a Number",
        description: "Find and print all prime factors of a given number n.",
        approach: [
          "1. While n is divisible by 2, print 2 and divide n by 2",
          "2. After step 1, n must be odd. Start from i=3 to sqrt(n)",
          "3. While i divides n, print i and divide n by i",
          "4. If n is a prime number > 2, print n"
        ],
        algorithmCharacteristics: [
          "Time complexity: O(√n)",
          "Space complexity: O(1)",
          "Efficient for composite numbers",
          "Handles even numbers first for optimization",
          "Works for all positive integers > 1"
        ],
        complexityDetails: {
          time: "O(√n)",
          space: "O(1)",
          explanation: "Iterates up to square root of n with optimizations for even numbers"
        },
        cppcode: `#include <iostream>
  #include <cmath>
  using namespace std;
  
  void printPrimeFactors(int n) {
      // Handle 2s that divide n
      while (n % 2 == 0) {
          cout << 2 << " ";
          n /= 2;
      }
      
      // n must be odd at this point
      for (int i = 3; i <= sqrt(n); i += 2) {
          while (n % i == 0) {
              cout << i << " ";
              n /= i;
          }
      }
      
      // Handle case when n is a prime > 2
      if (n > 2)
          cout << n << " ";
  }
  
  int main() {
      int n;
      cout << "Enter a number: ";
      cin >> n;
      cout << "Prime factors: ";
      printPrimeFactors(n);
      return 0;
  }`,
        javacode: `import java.util.Scanner;
  
  public class PrimeFactors {
      public static void printPrimeFactors(int n) {
          // Handle 2s that divide n
          while (n % 2 == 0) {
              System.out.print(2 + " ");
              n /= 2;
          }
          
          // n must be odd at this point
          for (int i = 3; i <= Math.sqrt(n); i += 2) {
              while (n % i == 0) {
                  System.out.print(i + " ");
                  n /= i;
              }
          }
          
          // Handle case when n is a prime > 2
          if (n > 2)
              System.out.print(n);
      }
      
      public static void main(String[] args) {
          Scanner sc = new Scanner(System.in);
          System.out.print("Enter a number: ");
          int n = sc.nextInt();
          System.out.print("Prime factors: ");
          printPrimeFactors(n);
      }
  }`,
        pythoncode: `import math
  
  def print_prime_factors(n):
      # Handle 2s that divide n
      while n % 2 == 0:
          print(2, end=" ")
          n = n // 2
      
      # n must be odd at this point
      for i in range(3, int(math.sqrt(n)) + 1, 2):
          while n % i == 0:
              print(i, end=" ")
              n = n // i
      
      # Handle case when n is a prime > 2
      if n > 2:
          print(n, end=" ")
  
  n = int(input("Enter a number: "))
  print("Prime factors:", end=" ")
  print_prime_factors(n)`,
        language: "cpp",
        javaLanguage: "java",
        pythonlanguage: "python",
        complexity: "Time Complexity: O(√n), Space Complexity: O(1)",
        link: "https://www.geeksforgeeks.org/print-all-prime-factors-of-a-given-number/"
      },
      {
        title: "All Divisors of a Number",
        description: "Find and print all divisors of a given number n in sorted order.",
        approach: [
          "1. Iterate from 1 to √n",
          "2. For each i, check if it divides n",
          "3. If it does, add both i and n/i to the result",
          "4. Handle edge case when i == n/i to avoid duplicates",
          "5. Sort the final list of divisors"
        ],
        algorithmCharacteristics: [
          "Time complexity: O(√n)",
          "Space complexity: O(√n) for storing divisors",
          "Efficient by only checking up to √n",
          "Maintains sorted order without full sorting",
          "Works for all positive integers"
        ],
        complexityDetails: {
          time: "O(√n)",
          space: "O(√n)",
          explanation: "Iterates up to square root of n and stores divisors"
        },
        cppcode: `#include <iostream>
  #include <vector>
  #include <algorithm>
  using namespace std;
  
  void printDivisors(int n) {
      vector<int> divisors;
      
      for (int i = 1; i <= sqrt(n); i++) {
          if (n % i == 0) {
              if (n / i == i)
                  divisors.push_back(i);
              else {
                  divisors.push_back(i);
                  divisors.push_back(n / i);
              }
          }
      }
      
      sort(divisors.begin(), divisors.end());
      
      for (int d : divisors)
          cout << d << " ";
  }
  
  int main() {
      int n;
      cout << "Enter a number: ";
      cin >> n;
      cout << "Divisors: ";
      printDivisors(n);
      return 0;
  }`,
        javacode: `import java.util.*;
  
  public class Divisors {
      public static void printDivisors(int n) {
          List<Integer> divisors = new ArrayList<>();
          
          for (int i = 1; i <= Math.sqrt(n); i++) {
              if (n % i == 0) {
                  if (n / i == i)
                      divisors.add(i);
                  else {
                      divisors.add(i);
                      divisors.add(n / i);
                  }
              }
          }
          
          Collections.sort(divisors);
          
          for (int d : divisors)
              System.out.print(d + " ");
      }
      
      public static void main(String[] args) {
          Scanner sc = new Scanner(System.in);
          System.out.print("Enter a number: ");
          int n = sc.nextInt();
          System.out.print("Divisors: ");
          printDivisors(n);
      }
  }`,
        pythoncode: `import math
  
  def print_divisors(n):
      divisors = set()
      
      for i in range(1, int(math.sqrt(n)) + 1):
          if n % i == 0:
              divisors.add(i)
              divisors.add(n // i)
      
      for d in sorted(divisors):
          print(d, end=" ")
  
  n = int(input("Enter a number: "))
  print("Divisors:", end=" ")
  print_divisors(n)`,
        language: "cpp",
        javaLanguage: "java",
        pythonlanguage: "python",
        complexity: "Time Complexity: O(√n), Space Complexity: O(√n)",
        link: "https://www.geeksforgeeks.org/find-all-divisors-of-a-natural-number-set-2/"
      },
      {
        title: "Sieve of Eratosthenes",
        description: "Find all prime numbers up to a given limit n using the Sieve of Eratosthenes algorithm.",
        approach: [
          "1. Create a boolean array 'prime[0..n]' and initialize all entries as true",
          "2. Mark 0 and 1 as non-prime",
          "3. Start from first prime (2), mark all its multiples as non-prime",
          "4. Move to next unmarked number and repeat",
          "5. Numbers remaining marked as true are primes"
        ],
        algorithmCharacteristics: [
          "Time complexity: O(n log log n)",
          "Space complexity: O(n)",
          "Most efficient way to find primes up to large n",
          "Uses boolean array for marking composites",
          "Can be optimized further with segmented sieve"
        ],
        complexityDetails: {
          time: "O(n log log n)",
          space: "O(n)",
          explanation: "Efficiently marks composite numbers in a boolean array"
        },
        cppcode: `#include <iostream>
  #include <vector>
  using namespace std;
  
  void sieveOfEratosthenes(int n) {
      vector<bool> prime(n+1, true);
      prime[0] = prime[1] = false;
      
      for (int p = 2; p * p <= n; p++) {
          if (prime[p]) {
              for (int i = p * p; i <= n; i += p)
                  prime[i] = false;
          }
      }
      
      cout << "Primes up to " << n << ": ";
      for (int p = 2; p <= n; p++) {
          if (prime[p])
              cout << p << " ";
      }
  }
  
  int main() {
      int n;
      cout << "Enter upper limit: ";
      cin >> n;
      sieveOfEratosthenes(n);
      return 0;
  }`,
        javacode: `import java.util.Scanner;
  
  public class Sieve {
      public static void sieveOfEratosthenes(int n) {
          boolean[] prime = new boolean[n+1];
          for (int i = 0; i <= n; i++)
              prime[i] = true;
          
          prime[0] = prime[1] = false;
          
          for (int p = 2; p * p <= n; p++) {
              if (prime[p]) {
                  for (int i = p * p; i <= n; i += p)
                      prime[i] = false;
              }
          }
          
          System.out.print("Primes up to " + n + ": ");
          for (int p = 2; p <= n; p++) {
              if (prime[p])
                  System.out.print(p + " ");
          }
      }
      
      public static void main(String[] args) {
          Scanner sc = new Scanner(System.in);
          System.out.print("Enter upper limit: ");
          int n = sc.nextInt();
          sieveOfEratosthenes(n);
      }
  }`,
        pythoncode: `def sieve_of_eratosthenes(n):
      prime = [True] * (n + 1)
      prime[0] = prime[1] = False
      
      p = 2
      while p * p <= n:
          if prime[p]:
              for i in range(p * p, n + 1, p):
                  prime[i] = False
          p += 1
      
      print(f"Primes up to {n}:", end=" ")
      for p in range(2, n + 1):
          if prime[p]:
              print(p, end=" ")
  
  n = int(input("Enter upper limit: "))
  sieve_of_eratosthenes(n)`,
        language: "cpp",
        javaLanguage: "java",
        pythonlanguage: "python",
        complexity: "Time Complexity: O(n log log n), Space Complexity: O(n)",
        link: "https://www.geeksforgeeks.org/sieve-of-eratosthenes/"
      },
      {
        title: "Prime Factorization Using Sieve",
        description: "Efficiently compute prime factorization of numbers using precomputed smallest prime factors (SPF).",
        approach: [
          "1. Precompute smallest prime factor (SPF) for every number up to n",
          "2. For each query number, repeatedly divide by its SPF",
          "3. Collect all prime factors in the process",
          "4. Continue until number becomes 1"
        ],
        algorithmCharacteristics: [
          "Time complexity: O(log n) per query after O(n) pre-processing",
          "Space complexity: O(n)",
          "Extremely efficient for multiple queries",
          "Uses modified Sieve of Eratosthenes",
          "Precomputes smallest prime factors for quick lookup"
        ],
        complexityDetails: {
          time: "Preprocessing: O(n), Query: O(log n)",
          space: "O(n)",
          explanation: "Preprocessing step enables fast factorization queries"
        },
        cppcode: `#include <iostream>
  #include <vector>
  using namespace std;
  
  vector<int> precomputeSPF(int max_num) {
      vector<int> spf(max_num + 1);
      for (int i = 0; i <= max_num; i++)
          spf[i] = i;
      
      for (int i = 2; i * i <= max_num; i++) {
          if (spf[i] == i) {
              for (int j = i * i; j <= max_num; j += i) {
                  if (spf[j] == j)
                      spf[j] = i;
              }
          }
      }
      return spf;
  }
  
  void printPrimeFactors(int n, const vector<int>& spf) {
      cout << "Prime factors of " << n << ": ";
      while (n != 1) {
          cout << spf[n] << " ";
          n /= spf[n];
      }
  }
  
  int main() {
      const int MAX = 1000000;
      vector<int> spf = precomputeSPF(MAX);
      
      int n;
      cout << "Enter a number (<= " << MAX << "): ";
      cin >> n;
      printPrimeFactors(n, spf);
      return 0;
  }`,
        javacode: `import java.util.Scanner;
  
  public class PrimeFactorizationSieve {
      public static int[] precomputeSPF(int maxNum) {
          int[] spf = new int[maxNum + 1];
          for (int i = 0; i <= maxNum; i++)
              spf[i] = i;
          
          for (int i = 2; i * i <= maxNum; i++) {
              if (spf[i] == i) {
                  for (int j = i * i; j <= maxNum; j += i) {
                      if (spf[j] == j)
                          spf[j] = i;
                  }
              }
          }
          return spf;
      }
      
      public static void printPrimeFactors(int n, int[] spf) {
          System.out.print("Prime factors of " + n + ": ");
          while (n != 1) {
              System.out.print(spf[n] + " ");
              n /= spf[n];
          }
      }
      
      public static void main(String[] args) {
          final int MAX = 1000000;
          int[] spf = precomputeSPF(MAX);
          
          Scanner sc = new Scanner(System.in);
          System.out.print("Enter a number (<= " + MAX + "): ");
          int n = sc.nextInt();
          printPrimeFactors(n, spf);
      }
  }`,
        pythoncode: `def precompute_spf(max_num):
      spf = [0] * (max_num + 1)
      for i in range(max_num + 1):
          spf[i] = i
      
      for i in range(2, int(max_num**0.5) + 1):
          if spf[i] == i:
              for j in range(i * i, max_num + 1, i):
                  if spf[j] == j:
                      spf[j] = i
      return spf
  
  def print_prime_factors(n, spf):
      print(f"Prime factors of {n}:", end=" ")
      while n != 1:
          print(spf[n], end=" ")
          n = n // spf[n]
  
  MAX = 1000000
  spf = precompute_spf(MAX)
  n = int(input(f"Enter a number (<= {MAX}): "))
  print_prime_factors(n, spf)`,
        language: "cpp",
        javaLanguage: "java",
        pythonlanguage: "python",
        complexity: "Preprocessing: O(n), Query: O(log n), Space: O(n)",
        link: "https://www.geeksforgeeks.org/prime-factorization-using-sieve-olog-n-multiple-queries/"
      },
      {
        title: "Power(n, k)",
        description: "Compute n raised to the power k efficiently using exponentiation by squaring.",
        approach: [
          "1. Handle base cases (k = 0, k = 1)",
          "2. For even exponent: power(n, k) = power(n*n, k/2)",
          "3. For odd exponent: power(n, k) = n * power(n*n, (k-1)/2)",
          "4. Handle negative exponents by using 1/power(n, -k)"
        ],
        algorithmCharacteristics: [
          "Time complexity: O(log k)",
          "Space complexity: O(log k) for recursion, O(1) for iterative",
          "Much faster than naive O(k) multiplication",
          "Works for both integer and floating point bases",
          "Handles negative exponents"
        ],
        complexityDetails: {
          time: "O(log k)",
          space: "O(1)",
          explanation: "Uses divide and conquer approach to reduce multiplications"
        },
        cppcode: `#include <iostream>
  using namespace std;
  
  double power(double x, int n) {
      long long N = n;
      if (N < 0) {
          x = 1 / x;
          N = -N;
      }
      
      double result = 1;
      double current_product = x;
      
      for (long long i = N; i > 0; i /= 2) {
          if (i % 2 == 1)
              result *= current_product;
          current_product *= current_product;
      }
      return result;
  }
  
  int main() {
      double base;
      int exponent;
      cout << "Enter base: ";
      cin >> base;
      cout << "Enter exponent: ";
      cin >> exponent;
      
      cout << base << "^" << exponent << " = " << power(base, exponent);
      return 0;
  }`,
        javacode: `import java.util.Scanner;
  
  public class Power {
      public static double power(double x, int n) {
          long N = n;
          if (N < 0) {
              x = 1 / x;
              N = -N;
          }
          
          double result = 1;
          double currentProduct = x;
          
          for (long i = N; i > 0; i /= 2) {
              if (i % 2 == 1)
                  result *= currentProduct;
              currentProduct *= currentProduct;
          }
          return result;
      }
      
      public static void main(String[] args) {
          Scanner sc = new Scanner(System.in);
          System.out.print("Enter base: ");
          double base = sc.nextDouble();
          System.out.print("Enter exponent: ");
          int exponent = sc.nextInt();
          
          System.out.println(base + "^" + exponent + " = " + power(base, exponent));
      }
  }`,
        pythoncode: `def power(x, n):
      N = n
      if N < 0:
          x = 1 / x
          N = -N
      
      result = 1
      current_product = x
      
      i = N
      while i > 0:
          if i % 2 == 1:
              result *= current_product
          current_product *= current_product
          i = i // 2
      return result
  
  base = float(input("Enter base: "))
  exponent = int(input("Enter exponent: "))
  print(f"{base}^{exponent} = {power(base, exponent)}")`,
        language: "cpp",
        javaLanguage: "java",
        pythonlanguage: "python",
        complexity: "Time Complexity: O(log n), Space Complexity: O(1)",
        link: "https://www.geeksforgeeks.org/write-a-c-program-to-calculate-powxn/"
      }
    ];
  return (
    <div
      className={`container mx-auto px-4 sm:px-6 py-12 rounded-2xl shadow-xl max-w-7xl transition-colors duration-300 ${
        darkMode
          ? "bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900"
          : "bg-gradient-to-br from-indigo-50 via-purple-50 to-indigo-50"
      }`}
    >
      <h1
        className={`text-4xl sm:text-5xl md:text-6xl font-extrabold text-center text-transparent bg-clip-text mb-8 sm:mb-12 ${
          darkMode
            ? "bg-gradient-to-r from-indigo-300 to-purple-400"
            : "bg-gradient-to-r from-indigo-600 to-purple-700"
        }`}
      >
        Mathematics Based Questions with Solutions
      </h1>

      <div className="space-y-8">
        {codeExamples.map((example, index) => (
          <article
            key={index}
            className={`p-6 sm:p-8 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 border ${
              darkMode
                ? "bg-gray-800/90 border-gray-700 hover:border-gray-600"
                : "bg-white/90 border-indigo-100 hover:border-indigo-200"
            }`}
          >
            <header className="mb-6">
              <h2
                className={`text-2xl sm:text-3xl font-bold text-left mb-4 ${
                  darkMode ? "text-indigo-300" : "text-indigo-800"
                }`}
              >
                {example.title}
              </h2>

              <div
                className={`p-4 sm:p-6 rounded-lg border transition-colors ${
                  darkMode
                    ? "bg-gray-700/50 border-gray-600"
                    : "bg-gray-50 border-gray-200"
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
                    darkMode ? "text-gray-300" : "text-gray-900"
                  } font-medium leading-relaxed space-y-2`}
                >
                  {formatDescription(example.description, darkMode)}
                </div>
              </div>

              <div className="space-y-4 mt-6">
                <CollapsibleSection
                  title="Approach"
                  content={example.approach}
                  isExpanded={expandedSections[`${index}-approach`]}
                  onToggle={() => toggleDetails(index, "approach")}
                  darkMode={darkMode}
                  colorScheme={{
                    bg: darkMode ? "bg-blue-900/30" : "bg-blue-50",
                    border: darkMode ? "border-blue-700" : "border-blue-200",
                    text: darkMode ? "text-blue-200" : "text-blue-800",
                    icon: darkMode ? "text-blue-300" : "text-blue-500",
                    hover: darkMode
                      ? "hover:bg-blue-900/20"
                      : "hover:bg-blue-50/70",
                  }}
                />

                <CollapsibleSection
                  title="Algorithm Characteristics"
                  content={example.algorithmCharacteristics}
                  isExpanded={expandedSections[`${index}-characteristics`]}
                  onToggle={() => toggleDetails(index, "characteristics")}
                  darkMode={darkMode}
                  colorScheme={{
                    bg: darkMode ? "bg-purple-900/30" : "bg-purple-50",
                    border: darkMode
                      ? "border-purple-700"
                      : "border-purple-200",
                    text: darkMode ? "text-purple-200" : "text-purple-800",
                    icon: darkMode ? "text-purple-300" : "text-purple-500",
                    hover: darkMode
                      ? "hover:bg-purple-900/20"
                      : "hover:bg-purple-50/70",
                  }}
                />

                <CollapsibleSection
                  title="Complexity Analysis"
                  content={
                    <div className="space-y-3">
                      <div className="flex flex-wrap gap-4">
                        <div
                          className={`px-3 py-2 rounded-lg ${
                            darkMode
                              ? "bg-blue-900/30 border border-blue-800"
                              : "bg-blue-100"
                          }`}
                        >
                          <div
                            className={`text-xs font-semibold ${
                              darkMode ? "text-blue-300" : "text-blue-600"
                            }`}
                          >
                            TIME COMPLEXITY
                          </div>
                          <div
                            className={`font-bold ${
                              darkMode ? "text-blue-100" : "text-blue-800"
                            }`}
                          >
                            {example.complexityDetails.time}
                          </div>
                        </div>
                        <div
                          className={`px-3 py-2 rounded-lg ${
                            darkMode
                              ? "bg-green-900/30 border border-green-800"
                              : "bg-green-100"
                          }`}
                        >
                          <div
                            className={`text-xs font-semibold ${
                              darkMode ? "text-green-300" : "text-green-600"
                            }`}
                          >
                            SPACE COMPLEXITY
                          </div>
                          <div
                            className={`font-bold ${
                              darkMode ? "text-green-100" : "text-green-800"
                            }`}
                          >
                            {example.complexityDetails.space}
                          </div>
                        </div>
                      </div>
                      <div
                        className={`prose prose-sm max-w-none ${
                          darkMode ? "text-gray-300" : "text-gray-700"
                        }`}
                      >
                        <p className="font-semibold">Explanation:</p>
                        <p>{example.complexityDetails.explanation}</p>
                      </div>
                    </div>
                  }
                  isExpanded={expandedSections[`${index}-complexity`]}
                  onToggle={() => toggleDetails(index, "complexity")}
                  darkMode={darkMode}
                  colorScheme={{
                    bg: darkMode ? "bg-green-900/30" : "bg-green-50",
                    border: darkMode ? "border-green-700" : "border-green-200",
                    text: darkMode ? "text-green-200" : "text-green-800",
                    icon: darkMode ? "text-green-300" : "text-green-500",
                    hover: darkMode
                      ? "hover:bg-green-900/20"
                      : "hover:bg-green-50/70",
                  }}
                />
              </div>
            </header>

            <div className="flex flex-wrap gap-3 mb-6">
              <a
                href={example.link}
                target="_blank"
                rel="noopener noreferrer"
                className={`inline-flex items-center justify-center bg-gradient-to-r ${
                  darkMode
                    ? "from-gray-900 to-gray-700 hover:from-gray-800 hover:to-gray-600"
                    : "from-gray-600 to-gray-800 hover:from-gray-600 hover:to-gray-900"
                } text-white font-medium px-4 py-2 sm:px-6 sm:py-3 rounded-lg transition-all transform hover:scale-[1.05] focus:outline-none focus:ring-2 focus:ring-yellow-500 focus:ring-offset-2 ${
                  darkMode
                    ? "focus:ring-offset-gray-900"
                    : "focus:ring-offset-white"
                }`}
              >
                <img
                  src={
                    darkMode
                      ? "https://upload.wikimedia.org/wikipedia/commons/a/ab/LeetCode_logo_white_no_text.svg"
                      : "https://upload.wikimedia.org/wikipedia/commons/1/19/LeetCode_logo_black.png"
                  }
                  alt="LeetCode Logo"
                  className="w-6 h-6 mr-2"
                />
                View Problem
              </a>

              <ToggleCodeButton
                language="cpp"
                isVisible={
                  visibleCode.index === index && visibleCode.language === "cpp"
                }
                onClick={() => toggleCodeVisibility("cpp", index)}
                darkMode={darkMode}
              />

              <ToggleCodeButton
                language="java"
                isVisible={
                  visibleCode.index === index && visibleCode.language === "java"
                }
                onClick={() => toggleCodeVisibility("java", index)}
                darkMode={darkMode}
              />

              <ToggleCodeButton
                language="python"
                isVisible={
                  visibleCode.index === index &&
                  visibleCode.language === "python"
                }
                onClick={() => toggleCodeVisibility("python", index)}
                darkMode={darkMode}
              />
            </div>

            <div className="space-y-4">
              <CodeExample
                example={example}
                isVisible={
                  visibleCode.index === index && visibleCode.language === "cpp"
                }
                language="cpp"
                code={example.cppcode}
                darkMode={darkMode}
              />

              <CodeExample
                example={example}
                isVisible={
                  visibleCode.index === index && visibleCode.language === "java"
                }
                language="java"
                code={example.javacode}
                darkMode={darkMode}
              />

              <CodeExample
                example={example}
                isVisible={
                  visibleCode.index === index &&
                  visibleCode.language === "python"
                }
                language="python"
                code={example.pythoncode}
                darkMode={darkMode}
              />
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}

export default Bit2;
