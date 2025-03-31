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

function Narray2() {
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
      title: "Implement Atoi",
      description: "Convert a string to a 32-bit signed integer (mimics C/C++'s atoi function).",
      approach: `
1. Discard leading whitespace
2. Check for optional '+' or '-' sign
3. Read in digits until non-digit or end of string
4. Handle overflow by clamping to INT_MAX/MIN
5. Return converted integer or 0 if no valid conversion`,
      algorithm: `
• Time complexity: O(n)
• Space complexity: O(1)
• Handles leading whitespace, signs
• Clamps overflow values
• Stops at first non-digit character`,
      cppcode: `#include <climits>
#include <string>
using namespace std;

int myAtoi(string s) {
    int i = 0, sign = 1, result = 0;
    while (s[i] == ' ') i++;
    
    if (s[i] == '-' || s[i] == '+') {
        sign = (s[i++] == '-') ? -1 : 1;
    }
    
    while (isdigit(s[i])) {
        int digit = s[i++] - '0';
        if (result > INT_MAX/10 || (result == INT_MAX/10 && digit > 7)) {
            return (sign == 1) ? INT_MAX : INT_MIN;
        }
        result = result * 10 + digit;
    }
    return result * sign;
}`,
      javacode: `public class Solution {
    public int myAtoi(String s) {
        int i = 0, sign = 1, result = 0;
        while (i < s.length() && s.charAt(i) == ' ') i++;
        
        if (i < s.length() && (s.charAt(i) == '-' || s.charAt(i) == '+')) {
            sign = (s.charAt(i++) == '-') ? -1 : 1;
        }
        
        while (i < s.length() && Character.isDigit(s.charAt(i))) {
            int digit = s.charAt(i++) - '0';
            if (result > Integer.MAX_VALUE/10 || 
                (result == Integer.MAX_VALUE/10 && digit > 7)) {
                return (sign == 1) ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            }
            result = result * 10 + digit;
        }
        return result * sign;
    }
}`,
      pythoncode: `def myAtoi(s: str) -> int:
    s = s.lstrip()
    if not s:
        return 0
    
    sign = 1
    if s[0] in '+-':
        sign = -1 if s[0] == '-' else 1
        s = s[1:]
    
    result = 0
    for c in s:
        if not c.isdigit():
            break
        digit = int(c)
        if result > (2**31 - 1) // 10 or (result == (2**31 - 1) // 10 and digit > 7):
            return 2**31 - 1 if sign == 1 else -2**31
        result = result * 10 + digit
    
    return sign * result`,
      complexity: "Time Complexity: O(n), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/string-to-integer-atoi/"
    },
    {
      title: "Add Binary Strings",
      description: "Given two binary strings, return their sum as a binary string.",
      approach: `
1. Start from end of both strings
2. Perform bit-by-bit addition with carry
3. Handle different length strings
4. Reverse final result`,
      algorithm: `
• Time complexity: O(max(m,n))
• Space complexity: O(max(m,n))
• Simple bit addition with carry
• Handles leading zeros`,
      cppcode: `#include <algorithm>
#include <string>
using namespace std;

string addBinary(string a, string b) {
    string result;
    int i = a.size() - 1, j = b.size() - 1;
    int carry = 0;
    
    while (i >= 0 || j >= 0 || carry) {
        int sum = carry;
        if (i >= 0) sum += a[i--] - '0';
        if (j >= 0) sum += b[j--] - '0';
        result.push_back(sum % 2 + '0');
        carry = sum / 2;
    }
    
    reverse(result.begin(), result.end());
    return result;
}`,
      javacode: `public class Solution {
    public String addBinary(String a, String b) {
        StringBuilder sb = new StringBuilder();
        int i = a.length() - 1, j = b.length() - 1;
        int carry = 0;
        
        while (i >= 0 || j >= 0 || carry > 0) {
            int sum = carry;
            if (i >= 0) sum += a.charAt(i--) - '0';
            if (j >= 0) sum += b.charAt(j--) - '0';
            sb.append(sum % 2);
            carry = sum / 2;
        }
        
        return sb.reverse().toString();
    }
}`,
      pythoncode: `def addBinary(a: str, b: str) -> str:
    result = []
    carry = 0
    i, j = len(a)-1, len(b)-1
    
    while i >= 0 or j >= 0 or carry:
        sum_val = carry
        if i >= 0:
            sum_val += int(a[i])
            i -= 1
        if j >= 0:
            sum_val += int(b[j])
            j -= 1
        result.append(str(sum_val % 2))
        carry = sum_val // 2
    
    return ''.join(reversed(result))`,
      complexity: "Time Complexity: O(max(m,n)), Space Complexity: O(max(m,n))",
      link: "https://leetcode.com/problems/add-binary/"
    },
    {
      title: "Anagram Check",
      description: "Check if two strings are anagrams of each other.",
      approach: `
1. Compare lengths of both strings
2. Use frequency count array for one string
3. Decrement counts for characters in second string
4. All counts should be zero for anagrams`,
      algorithm: `
• Time complexity: O(n)
• Space complexity: O(1) (fixed size count array)
• Case insensitive option available
• Handles Unicode with hash map`,
      cppcode: `#include <string>
#include <array>
using namespace std;

bool isAnagram(string s, string t) {
    if (s.size() != t.size()) return false;
    
    array<int, 26> count{};
    for (char c : s) count[c - 'a']++;
    for (char c : t) {
        if (--count[c - 'a'] < 0) return false;
    }
    return true;
}`,
      javacode: `public class Solution {
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) return false;
        
        int[] count = new int[26];
        for (char c : s.toCharArray()) count[c - 'a']++;
        for (char c : t.toCharArray()) {
            if (--count[c - 'a'] < 0) return false;
        }
        return true;
    }
}`,
      pythoncode: `def isAnagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    
    count = [0] * 26
    for c in s:
        count[ord(c) - ord('a')] += 1
    for c in t:
        count[ord(c) - ord('a')] -= 1
        if count[ord(c) - ord('a')] < 0:
            return False
    return True`,
      complexity: "Time Complexity: O(n), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/valid-anagram/"
    },
    {
      title: "First Non-Repeating Character",
      description: "Find the first non-repeating character in a string and return its index.",
      approach: `
1. Build frequency count of characters
2. Traverse string to find first character with count 1
3. Return index or -1 if none found`,
      algorithm: `
• Time complexity: O(n)
• Space complexity: O(1) (fixed size count array)
• Two-pass approach
• Handles empty string case`,
      cppcode: `#include <string>
#include <array>
using namespace std;

int firstUniqChar(string s) {
    array<int, 26> count{};
    for (char c : s) count[c - 'a']++;
    for (int i = 0; i < s.size(); i++) {
        if (count[s[i] - 'a'] == 1) return i;
    }
    return -1;
}`,
      javacode: `public class Solution {
    public int firstUniqChar(String s) {
        int[] count = new int[26];
        for (char c : s.toCharArray()) count[c - 'a']++;
        for (int i = 0; i < s.length(); i++) {
            if (count[s.charAt(i) - 'a'] == 1) return i;
        }
        return -1;
    }
}`,
      pythoncode: `def firstUniqChar(s: str) -> int:
    count = [0] * 26
    for c in s:
        count[ord(c) - ord('a')] += 1
    for i, c in enumerate(s):
        if count[ord(c) - ord('a')] == 1:
            return i
    return -1`,
      complexity: "Time Complexity: O(n), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/first-unique-character-in-a-string/"
    },
    {
      title: "Search Pattern (KMP Algorithm)",
      description: "Implement strStr() - find first occurrence of needle in haystack using KMP algorithm.",
      approach: `
1. Preprocess pattern to create longest prefix suffix (LPS) array
2. Use LPS array to skip unnecessary comparisons
3. Perform pattern matching with optimized shifts`,
      algorithm: `
• Time complexity: O(m+n)
• Space complexity: O(m) for LPS array
• Efficient for large texts with repeating patterns
• Avoids backtracking in text`,
      cppcode: `#include <vector>
#include <string>
using namespace std;

vector<int> computeLPS(string pattern) {
    vector<int> lps(pattern.size(), 0);
    int len = 0, i = 1;
    while (i < pattern.size()) {
        if (pattern[i] == pattern[len]) {
            lps[i++] = ++len;
        } else {
            if (len != 0) len = lps[len - 1];
            else lps[i++] = 0;
        }
    }
    return lps;
}

int strStr(string haystack, string needle) {
    if (needle.empty()) return 0;
    vector<int> lps = computeLPS(needle);
    int i = 0, j = 0;
    while (i < haystack.size()) {
        if (haystack[i] == needle[j]) {
            i++; j++;
            if (j == needle.size()) return i - j;
        } else {
            if (j != 0) j = lps[j - 1];
            else i++;
        }
    }
    return -1;
}`,
      javacode: `public class Solution {
    private int[] computeLPS(String pattern) {
        int[] lps = new int[pattern.length()];
        int len = 0, i = 1;
        while (i < pattern.length()) {
            if (pattern.charAt(i) == pattern.charAt(len)) {
                lps[i++] = ++len;
            } else {
                if (len != 0) len = lps[len - 1];
                else lps[i++] = 0;
            }
        }
        return lps;
    }
    
    public int strStr(String haystack, String needle) {
        if (needle.isEmpty()) return 0;
        int[] lps = computeLPS(needle);
        int i = 0, j = 0;
        while (i < haystack.length()) {
            if (haystack.charAt(i) == needle.charAt(j)) {
                i++; j++;
                if (j == needle.length()) return i - j;
            } else {
                if (j != 0) j = lps[j - 1];
                else i++;
            }
        }
        return -1;
    }
}`,
      pythoncode: `def strStr(haystack: str, needle: str) -> int:
    if not needle:
        return 0
    
    # Compute LPS array
    lps = [0] * len(needle)
    length = 0
    i = 1
    while i < len(needle):
        if needle[i] == needle[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    
    # KMP search
    i = j = 0
    while i < len(haystack):
        if haystack[i] == needle[j]:
            i += 1
            j += 1
            if j == len(needle):
                return i - j
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return -1`,
      complexity: "Time Complexity: O(m+n), Space Complexity: O(m)",
      link: "https://leetcode.com/problems/implement-strstr/"
    },
    {
      title: "Minimum Characters to Add for Palindrome",
      description: "Find minimum characters to add to make a string palindrome.",
      approach: `
1. Use modified KMP algorithm with LPS array
2. Create a new string: original + '$' + reverse
3. Compute LPS array for this combined string
4. Minimum insertions = length - LPS last value`,
      algorithm: `
• Time complexity: O(n)
• Space complexity: O(n)
• Efficient solution using pattern matching
• Works for any string length`,
      cppcode: `#include <string>
#include <algorithm>
#include <vector>
using namespace std;

int minInsertions(string s) {
    string rev = s;
    reverse(rev.begin(), rev.end());
    string combined = s + "#" + rev;
    vector<int> lps(combined.size(), 0);
    
    for (int i = 1; i < combined.size(); i++) {
        int len = lps[i-1];
        while (len > 0 && combined[i] != combined[len]) {
            len = lps[len-1];
        }
        if (combined[i] == combined[len]) {
            len++;
        }
        lps[i] = len;
    }
    
    return s.size() - lps.back();
}`,
      javacode: `public class Solution {
    public int minInsertions(String s) {
        String rev = new StringBuilder(s).reverse().toString();
        String combined = s + "#" + rev;
        int[] lps = new int[combined.length()];
        
        for (int i = 1; i < combined.length(); i++) {
            int len = lps[i-1];
            while (len > 0 && combined.charAt(i) != combined.charAt(len)) {
                len = lps[len-1];
            }
            if (combined.charAt(i) == combined.charAt(len)) {
                len++;
            }
            lps[i] = len;
        }
        
        return s.length() - lps[lps.length - 1];
    }
}`,
      pythoncode: `def minInsertions(s: str) -> int:
    rev = s[::-1]
    combined = s + '#' + rev
    lps = [0] * len(combined)
    
    for i in range(1, len(combined)):
        length = lps[i-1]
        while length > 0 and combined[i] != combined[length]:
            length = lps[length-1]
        if combined[i] == combined[length]:
            length += 1
        lps[i] = length
    
    return len(s) - lps[-1]`,
      complexity: "Time Complexity: O(n), Space Complexity: O(n)",
      link: "https://www.geeksforgeeks.org/minimum-characters-added-front-make-string-palindrome/"
    },
    {
      title: "String Rotation of Each Other",
      description: "Check if one string is a rotation of another.",
      approach: `
1. Check if lengths are equal
2. Concatenate first string with itself
3. Check if second string is substring of concatenated string`,
      algorithm: `
• Time complexity: O(n)
• Space complexity: O(n) for concatenated string
• Simple and elegant solution
• Uses standard string search`,
      cppcode: `#include <string>
using namespace std;

bool isRotation(string s1, string s2) {
    if (s1.size() != s2.size()) return false;
    string combined = s1 + s1;
    return combined.find(s2) != string::npos;
}`,
      javacode: `public class Solution {
    public boolean isRotation(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        String combined = s1 + s1;
        return combined.contains(s2);
    }
}`,
      pythoncode: `def isRotation(s1: str, s2: str) -> bool:
    if len(s1) != len(s2):
        return False
    return s2 in (s1 + s1)`,
      complexity: "Time Complexity: O(n), Space Complexity: O(n)",
      link: "https://www.geeksforgeeks.org/a-program-to-check-if-strings-are-rotations-of-each-other/"
    },
    {
      title: "Fizz Buzz",
      description: "For numbers 1 to n, return 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both.",
      approach: `
1. Iterate from 1 to n
2. Check divisibility by 3 and 5 first
3. Then check individual divisibility
4. Default to string representation of number`,
      algorithm: `
• Time complexity: O(n)
• Space complexity: O(n) for output
• Simple conditional checks
• Handles edge cases`,
      cppcode: `#include <vector>
#include <string>
using namespace std;

vector<string> fizzBuzz(int n) {
    vector<string> result;
    for (int i = 1; i <= n; i++) {
        if (i % 15 == 0) result.push_back("FizzBuzz");
        else if (i % 3 == 0) result.push_back("Fizz");
        else if (i % 5 == 0) result.push_back("Buzz");
        else result.push_back(to_string(i));
    }
    return result;
}`,
      javacode: `import java.util.*;

public class Solution {
    public List<String> fizzBuzz(int n) {
        List<String> result = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            if (i % 15 == 0) result.add("FizzBuzz");
            else if (i % 3 == 0) result.add("Fizz");
            else if (i % 5 == 0) result.add("Buzz");
            else result.add(String.valueOf(i));
        }
        return result;
    }
}`,
      pythoncode: `def fizzBuzz(n: int) -> List[str]:
    result = []
    for i in range(1, n+1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result`,
      complexity: "Time Complexity: O(n), Space Complexity: O(n)",
      link: "https://leetcode.com/problems/fizz-buzz/"
    },
    {
      title: "CamelCase Pattern Matching",
      description: "Find all dictionary words that match a given CamelCase pattern.",
      approach: `
1. Extract uppercase characters from pattern
2. For each word, extract its uppercase characters
3. Compare with pattern's uppercase sequence
4. Return matching words`,
      algorithm: `
• Time complexity: O(n*k) where n is number of words, k is average length
• Space complexity: O(m) for pattern uppercase storage
• Handles empty pattern case
• Case-sensitive comparison`,
      cppcode: `#include <vector>
#include <string>
using namespace std;

vector<string> camelMatch(vector<string>& queries, string pattern) {
    vector<string> result;
    for (string word : queries) {
        int i = 0;
        bool match = true;
        for (char c : word) {
            if (i < pattern.size() && c == pattern[i]) {
                i++;
            } else if (isupper(c)) {
                match = false;
                break;
            }
        }
        result.push_back(match && i == pattern.size() ? "true" : "false");
    }
    return result;
}`,
      javacode: `import java.util.*;

public class Solution {
    public List<Boolean> camelMatch(String[] queries, String pattern) {
        List<Boolean> result = new ArrayList<>();
        for (String word : queries) {
            int i = 0;
            boolean match = true;
            for (char c : word.toCharArray()) {
                if (i < pattern.length() && c == pattern.charAt(i)) {
                    i++;
                } else if (Character.isUpperCase(c)) {
                    match = false;
                    break;
                }
            }
            result.add(match && i == pattern.length());
        }
        return result;
    }
}`,
      pythoncode: `def camelMatch(queries: List[str], pattern: str) -> List[bool]:
    result = []
    for word in queries:
        i = 0
        match = True
        for c in word:
            if i < len(pattern) and c == pattern[i]:
                i += 1
            elif c.isupper():
                match = False
                break
        result.append(match and i == len(pattern))
    return result`,
      complexity: "Time Complexity: O(n*k), Space Complexity: O(m)",
      link: "https://leetcode.com/problems/camelcase-matching/"
    },
    {
      title: "Minimum Repeat to Make Substring",
      description: "Find minimum repeats of string A needed so that string B is a substring.",
      approach: `
1. Check if all characters of B exist in A
2. Try possible repeats (max 2 needed if B is substring of A+A)
3. Use string find operation`,
      algorithm: `
• Time complexity: O(n)
• Space complexity: O(n)
• Optimal solution with max 3 checks
• Handles character mismatch case`,
      cppcode: `#include <string>
using namespace std;

int minRepeats(string A, string B) {
    string temp = A;
    int repeats = 1;
    while (temp.size() < B.size()) {
        temp += A;
        repeats++;
    }
    if (temp.find(B) != string::npos) return repeats;
    temp += A;
    repeats++;
    return (temp.find(B) != string::npos) ? repeats : -1;
}`,
      javacode: `public class Solution {
    public int minRepeats(String A, String B) {
        String temp = A;
        int repeats = 1;
        while (temp.length() < B.length()) {
            temp += A;
            repeats++;
        }
        if (temp.contains(B)) return repeats;
        temp += A;
        repeats++;
        return temp.contains(B) ? repeats : -1;
    }
}`,
      pythoncode: `def minRepeats(A: str, B: str) -> int:
    temp = A
    repeats = 1
    while len(temp) < len(B):
        temp += A
        repeats += 1
    if B in temp:
        return repeats
    temp += A
    repeats += 1
    return repeats if B in temp else -1`,
      complexity: "Time Complexity: O(n), Space Complexity: O(n)",
      link: "https://www.geeksforgeeks.org/minimum-number-of-times-a-has-to-be-repeated-to-make-b-a-substring/"
    },
    {
      title: "Longest Prefix Suffix (KMP LPS)",
      description: "Find the length of the longest proper prefix which is also a suffix for each prefix of the string.",
      approach: `
1. Initialize LPS array with 0
2. Use two pointers to compare prefix and suffix
3. Build LPS array incrementally`,
      algorithm: `
• Time complexity: O(n)
• Space complexity: O(n)
• KMP preprocessing step
• Used in pattern matching algorithms`,
      cppcode: `#include <vector>
#include <string>
using namespace std;

vector<int> computeLPS(string s) {
    vector<int> lps(s.size(), 0);
    int len = 0, i = 1;
    while (i < s.size()) {
        if (s[i] == s[len]) {
            lps[i++] = ++len;
        } else {
            if (len != 0) len = lps[len-1];
            else lps[i++] = 0;
        }
    }
    return lps;
}`,
      javacode: `public class Solution {
    public int[] computeLPS(String s) {
        int[] lps = new int[s.length()];
        int len = 0, i = 1;
        while (i < s.length()) {
            if (s.charAt(i) == s.charAt(len)) {
                lps[i++] = ++len;
            } else {
                if (len != 0) len = lps[len-1];
                else lps[i++] = 0;
            }
        }
        return lps;
    }
}`,
      pythoncode: `def computeLPS(s: str) -> List[int]:
    lps = [0] * len(s)
    length = 0
    i = 1
    while i < len(s):
        if s[i] == s[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length-1]
            else:
                lps[i] = 0
                i += 1
    return lps`,
      complexity: "Time Complexity: O(n), Space Complexity: O(n)",
      link: "https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/"
    },
    {
      title: "Integer to English Words",
      description: "Convert a non-negative integer to its English words representation.",
      approach: `
1. Break number into chunks of 3 digits (thousands, millions, etc.)
2. Convert each 3-digit chunk to words
3. Combine with appropriate scale words
4. Handle edge cases (zero, teens, tens)`,
      algorithm: `
• Time complexity: O(1) (fixed number of digits)
• Space complexity: O(1)
• Recursive solution
• Uses helper functions for different scales`,
      cppcode: `#include <string>
#include <vector>
using namespace std;

vector<string> ones = {"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"};
vector<string> teens = {"Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
vector<string> tens = {"", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};

string helper(int num) {
    if (num == 0) return "";
    if (num < 10) return ones[num] + " ";
    if (num < 20) return teens[num - 10] + " ";
    if (num < 100) return tens[num / 10] + " " + helper(num % 10);
    return ones[num / 100] + " Hundred " + helper(num % 100);
}

string numberToWords(int num) {
    if (num == 0) return "Zero";
    vector<string> scales = {"", "Thousand", "Million", "Billion"};
    string result;
    int scale = 0;
    while (num > 0) {
        int chunk = num % 1000;
        if (chunk != 0) {
            result = helper(chunk) + scales[scale] + " " + result;
        }
        num /= 1000;
        scale++;
    }
    while (result.back() == ' ') result.pop_back();
    return result;
}`,
      javacode: `class Solution {
    private final String[] ones = {"", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"};
    private final String[] teens = {"Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"};
    private final String[] tens = {"", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"};
    
    private String helper(int num) {
        if (num == 0) return "";
        if (num < 10) return ones[num] + " ";
        if (num < 20) return teens[num - 10] + " ";
        if (num < 100) return tens[num / 10] + " " + helper(num % 10);
        return ones[num / 100] + " Hundred " + helper(num % 100);
    }
    
    public String numberToWords(int num) {
        if (num == 0) return "Zero";
        String[] scales = {"", "Thousand", "Million", "Billion"};
        StringBuilder result = new StringBuilder();
        int scale = 0;
        while (num > 0) {
            int chunk = num % 1000;
            if (chunk != 0) {
                result.insert(0, helper(chunk) + scales[scale] + " ");
            }
            num /= 1000;
            scale++;
        }
        return result.toString().trim();
    }
}`,
      pythoncode: `def numberToWords(num: int) -> str:
    if num == 0:
        return "Zero"
    
    ones = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    teens = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", 
             "Seventeen", "Eighteen", "Nineteen"]
    tens = ["", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", 
            "Eighty", "Ninety"]
    scales = ["", "Thousand", "Million", "Billion"]
    
    def helper(n):
        if n == 0:
            return ""
        if n < 10:
            return ones[n] + " "
        if n < 20:
            return teens[n - 10] + " "
        if n < 100:
            return tens[n // 10] + " " + helper(n % 10)
        return ones[n // 100] + " Hundred " + helper(n % 100)
    
    res = ""
    scale = 0
    while num > 0:
        chunk = num % 1000
        if chunk != 0:
            res = helper(chunk) + scales[scale] + " " + res
        num //= 1000
        scale += 1
    
    return res.strip()`,
      complexity: "Time Complexity: O(1), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/integer-to-english-words/"
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
        String Problems with Solutions
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

export default Narray2;