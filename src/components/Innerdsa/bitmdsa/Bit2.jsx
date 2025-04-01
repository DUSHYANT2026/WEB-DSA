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

function Bit2() {
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
        title: "Print Prime Factors of a Number",
        description: "Find and print all prime factors of a given number n.",
        approach: `
    1. While n is divisible by 2, print 2 and divide n by 2
    2. After step 1, n must be odd. Start from i=3 to sqrt(n)
    3. While i divides n, print i and divide n by i
    4. If n is a prime number > 2, print n`,
        algorithm: `
    • Time complexity: O(sqrt(n))
    • Space complexity: O(1)
    • Efficient for composite numbers
    • Handles even numbers first for optimization
    • Works for all positive integers > 1`,
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
        approach: `
    1. Iterate from 1 to sqrt(n)
    2. For each i, check if it divides n
    3. If it does, add both i and n/i to the result
    4. Handle edge case when i == n/i to avoid duplicates
    5. Sort the final list of divisors`,
        algorithm: `
    • Time complexity: O(sqrt(n))
    • Space complexity: O(sqrt(n)) for storing divisors
    • Efficient by only checking up to sqrt(n)
    • Maintains sorted order without full sorting
    • Works for all positive integers`,
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
        approach: `
    1. Create a boolean array "prime[0..n]" and initialize all entries as true
    2. Mark 0 and 1 as non-prime
    3. Start from first prime (2), mark all its multiples as non-prime
    4. Move to next unmarked number and repeat
    5. Numbers remaining marked as true are primes`,
        algorithm: `
    • Time complexity: O(n log log n)
    • Space complexity: O(n)
    • Most efficient way to find primes up to large n
    • Uses boolean array for marking composites
    • Can be optimized further with segmented sieve`,
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
        approach: `
    1. Precompute smallest prime factor (SPF) for every number up to n
    2. For each query number, repeatedly divide by its SPF
    3. Collect all prime factors in the process
    4. Continue until number becomes 1`,
        algorithm: `
    • Time complexity: O(log n) per query after O(n) pre-processing
    • Space complexity: O(n)
    • Extremely efficient for multiple queries
    • Uses modified Sieve of Eratosthenes
    • Precomputes smallest prime factors for quick lookup`,
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
        approach: `
    1. Handle base cases (k = 0, k = 1)
    2. For even exponent: power(n, k) = power(n*n, k/2)
    3. For odd exponent: power(n, k) = n * power(n*n, (k-1)/2)
    4. Handle negative exponents by using 1/power(n, -k)`,
        algorithm: `
    • Time complexity: O(log k)
    • Space complexity: O(log k) for recursion, O(1) for iterative
    • Much faster than naive O(k) multiplication
    • Works for both integer and floating point bases
    • Handles negative exponents`,
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
        Mathematics Based Questions
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

export default Bit2;
