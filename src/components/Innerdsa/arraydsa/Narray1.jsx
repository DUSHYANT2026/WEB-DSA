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

function Narray1() {
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
      title: "Second Largest Element of Array",
      description: "Find the second largest element in an unsorted array in a single pass, returning -1 if no valid second largest exists.",
      approach: `
1. Works on both sorted or unsorted arrays
2. Scans Through the Array Once
3. Updates the Largest and Second Largest Elements Dynamically
4. Handles Duplicate and Single Element Cases
5. Continues Until All Elements Are Processed`,
      algorithm: `
• Time complexity: O(n)
• Space complexity: O(1)
• No need of sorted input array`,
      cppcode: `#include <bits/stdc++.h>
using namespace std;

int getSecondLargest(vector<int> &arr) {
    int max1 = INT_MIN, max2 = INT_MIN;

    // Find the largest and second largest
    for (int it : arr) {
        if (it > max1) {
            max2 = max1;
            max1 = it;
        } else if (it > max2 && it != max1) {
            max2 = it;
        }
    }
    
    // Check if second largest is found
    return (max2 == INT_MIN) ? -1 : max2;
}`,
      javacode: `import java.util.*;

public class Main {
    public static int getSecondLargest(int[] arr) {
        int max1 = Integer.MIN_VALUE, max2 = Integer.MIN_VALUE;

        // Find the largest and second largest
        for (int it : arr) {
            if (it > max1) {
                max2 = max1;
                max1 = it;
            } else if (it > max2 && it != max1) {
                max2 = it;
            }
        }

        // Check if second largest is found
        return (max2 == Integer.MIN_VALUE) ? -1 : max2;
    }
}`,
      pythoncode: `def getSecondLargest(arr):
    max1, max2 = float('-inf'), float('-inf')

    # Find the largest and second largest
    for it in arr:
        if it > max1:
            max2, max1 = max1, it
        elif it > max2 and it != max1:
            max2 = it

    return -1 if max2 == float('-inf') else max2`,
      complexity: "Time Complexity: O(n), Space Complexity: O(1)",
      link: "https://www.geeksforgeeks.org/problems/second-largest3735/1",
    },
    {
      title: "Move all zeros to end",
      description: "Move all zeros in an array to the end while maintaining the relative order of non-zero elements.",
      approach: `
1. Initialize a pointer for non-zero elements
2. Traverse the array with another pointer
3. For each non-zero element, place it at the non-zero pointer position
4. Increment the non-zero pointer
5. After traversal, fill remaining positions with zeros`,
      algorithm: `
• Time complexity: O(n)
• Space complexity: O(1)
• Maintains relative order of non-zero elements
• Single pass through the array
• In-place operation`,
      cppcode: `#include <vector>
using namespace std;

void moveZeroes(vector<int>& nums) {
    int nonZero = 0;
    for (int i = 0; i < nums.size(); i++) {
        if (nums[i] != 0) {
            nums[nonZero++] = nums[i];
        }
    }
    while (nonZero < nums.size()) {
        nums[nonZero++] = 0;
    }
}`,
      javacode: `public class Solution {
    public void moveZeroes(int[] nums) {
        int nonZero = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                nums[nonZero++] = nums[i];
            }
        }
        while (nonZero < nums.length) {
            nums[nonZero++] = 0;
        }
    }
}`,
      pythoncode: `def moveZeroes(nums):
    nonZero = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[nonZero] = nums[i]
            nonZero += 1
    while nonZero < len(nums):
        nums[nonZero] = 0
        nonZero += 1`,
      complexity: "Time Complexity: O(n), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/move-zeroes/",
    },
    {
      title: "Reverse an Array",
      description: "Reverse the elements of an array in-place.",
      approach: `
1. Initialize two pointers: start at the beginning and end at the end of the array
2. Swap elements at these pointers
3. Move the start pointer forward and end pointer backward
4. Continue until start pointer crosses end pointer`,
      algorithm: `
• Time complexity: O(n)
• Space complexity: O(1)
• In-place operation
• Simple two-pointer approach`,
      cppcode: `#include <vector>
using namespace std;

void reverseArray(vector<int>& arr) {
    int start = 0, end = arr.size() - 1;
    while (start < end) {
        swap(arr[start], arr[end]);
        start++;
        end--;
    }
}`,
      javacode: `public class Main {
    public static void reverseArray(int[] arr) {
        int start = 0, end = arr.length - 1;
        while (start < end) {
            int temp = arr[start];
            arr[start] = arr[end];
            arr[end] = temp;
            start++;
            end--;
        }
    }
}`,
      pythoncode: `def reverse_array(arr):
    start, end = 0, len(arr) - 1
    while start < end:
        arr[start], arr[end] = arr[end], arr[start]
        start += 1
        end -= 1`,
      complexity: "Time Complexity: O(n), Space Complexity: O(1)",
      link: "https://www.geeksforgeeks.org/write-a-program-to-reverse-an-array-or-string/"
    },
    {
      title: "Rotate Array",
      description: "Rotate an array to the right by k steps where k is non-negative.",
      approach: `
1. Reverse the entire array
2. Reverse the first k elements
3. Reverse the remaining elements
4. Handle cases where k is larger than array length`,
      algorithm: `
• Time complexity: O(n)
• Space complexity: O(1)
• In-place rotation
• Uses reversal algorithm`,
      cppcode: `#include <vector>
#include <algorithm>
using namespace std;

void rotate(vector<int>& nums, int k) {
    k = k % nums.size();
    reverse(nums.begin(), nums.end());
    reverse(nums.begin(), nums.begin() + k);
    reverse(nums.begin() + k, nums.end());
}`,
      javacode: `public class Solution {
    public void rotate(int[] nums, int k) {
        k = k % nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }
    
    private void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
        }
    }
}`,
      pythoncode: `def rotate(nums, k):
    k = k % len(nums)
    nums.reverse()
    nums[:k] = reversed(nums[:k])
    nums[k:] = reversed(nums[k:])`,
      complexity: "Time Complexity: O(n), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/rotate-array/"
    },
    {
      title: "Next Permutation",
      description: "Find the next lexicographically greater permutation of numbers.",
      approach: `
1. Find the first decreasing element from the end (pivot)
2. Find the smallest element greater than pivot to its right
3. Swap these two elements
4. Reverse the suffix after the pivot position`,
      algorithm: `
• Time complexity: O(n)
• Space complexity: O(1)
• In-place operation
• Handles duplicate elements`,
      cppcode: `#include <vector>
#include <algorithm>
using namespace std;

void nextPermutation(vector<int>& nums) {
    int n = nums.size(), i = n - 2;
    while (i >= 0 && nums[i] >= nums[i + 1]) i--;
    
    if (i >= 0) {
        int j = n - 1;
        while (nums[j] <= nums[i]) j--;
        swap(nums[i], nums[j]);
    }
    reverse(nums.begin() + i + 1, nums.end());
}`,
      javacode: `public class Solution {
    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) i--;
        
        if (i >= 0) {
            int j = nums.length - 1;
            while (nums[j] <= nums[i]) j--;
            swap(nums, i, j);
        }
        reverse(nums, i + 1);
    }
    
    private void reverse(int[] nums, int start) {
        int end = nums.length - 1;
        while (start < end) {
            swap(nums, start, end);
            start++;
            end--;
        }
    }
    
    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}`,
      pythoncode: `def nextPermutation(nums):
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    
    if i >= 0:
        j = len(nums) - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    
    nums[i+1:] = reversed(nums[i+1:])`,
      complexity: "Time Complexity: O(n), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/next-permutation/"
    },
    {
      title: "Majority Element II",
      description: "Find all elements that appear more than ⌊n/3⌋ times in an array.",
      approach: `
1. Use Boyer-Moore Voting Algorithm extended for two candidates
2. Track two potential candidates and their counts
3. Verify if candidates actually appear more than n/3 times`,
      algorithm: `
• Time complexity: O(n)
• Space complexity: O(1)
• Extended Boyer-Moore algorithm
• Handles cases with 0, 1, or 2 majority elements`,
      cppcode: `#include <vector>
using namespace std;

vector<int> majorityElement(vector<int>& nums) {
    int count1 = 0, count2 = 0;
    int candidate1 = INT_MIN, candidate2 = INT_MIN;
    
    for (int num : nums) {
        if (num == candidate1) count1++;
        else if (num == candidate2) count2++;
        else if (count1 == 0) { candidate1 = num; count1 = 1; }
        else if (count2 == 0) { candidate2 = num; count2 = 1; }
        else { count1--; count2--; }
    }
    
    // Verification
    count1 = count2 = 0;
    for (int num : nums) {
        if (num == candidate1) count1++;
        else if (num == candidate2) count2++;
    }
    
    vector<int> result;
    if (count1 > nums.size() / 3) result.push_back(candidate1);
    if (count2 > nums.size() / 3) result.push_back(candidate2);
    return result;
}`,
      javacode: `import java.util.*;

public class Solution {
    public List<Integer> majorityElement(int[] nums) {
        int count1 = 0, count2 = 0;
        Integer candidate1 = null, candidate2 = null;
        
        for (int num : nums) {
            if (candidate1 != null && num == candidate1) count1++;
            else if (candidate2 != null && num == candidate2) count2++;
            else if (count1 == 0) { candidate1 = num; count1 = 1; }
            else if (count2 == 0) { candidate2 = num; count2 = 1; }
            else { count1--; count2--; }
        }
        
        // Verification
        count1 = count2 = 0;
        for (int num : nums) {
            if (num == candidate1) count1++;
            else if (candidate2 != null && num == candidate2) count2++;
        }
        
        List<Integer> result = new ArrayList<>();
        if (count1 > nums.length / 3) result.add(candidate1);
        if (count2 > nums.length / 3) result.add(candidate2);
        return result;
    }
}`,
      pythoncode: `def majorityElement(nums):
    count1, count2 = 0, 0
    candidate1, candidate2 = None, None
    
    for num in nums:
        if num == candidate1:
            count1 += 1
        elif num == candidate2:
            count2 += 1
        elif count1 == 0:
            candidate1 = num
            count1 = 1
        elif count2 == 0:
            candidate2 = num
            count2 = 1
        else:
            count1 -= 1
            count2 -= 1
    
    # Verification
    result = []
    for candidate in [candidate1, candidate2]:
        if nums.count(candidate) > len(nums) // 3:
            result.append(candidate)
    
    return result`,
      complexity: "Time Complexity: O(n), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/majority-element-ii/"
    },
    {
      title: "Stock Buy and Sell (Multiple Transactions)",
      description: "Find the maximum profit by buying and selling stocks multiple times.",
      approach: `
1. Buy when current price is lower than next day's price
2. Sell when current price is higher than next day's price
3. Accumulate all possible profits from consecutive increases`,
      algorithm: `
• Time complexity: O(n)
• Space complexity: O(1)
• Greedy approach
• Captures all increasing sequences`,
      cppcode: `#include <vector>
using namespace std;

int maxProfit(vector<int>& prices) {
    int profit = 0;
    for (int i = 1; i < prices.size(); i++) {
        if (prices[i] > prices[i - 1]) {
            profit += prices[i] - prices[i - 1];
        }
    }
    return profit;
}`,
      javacode: `public class Solution {
    public int maxProfit(int[] prices) {
        int profit = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1]) {
                profit += prices[i] - prices[i - 1];
            }
        }
        return profit;
    }
}`,
      pythoncode: `def maxProfit(prices):
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            profit += prices[i] - prices[i-1]
    return profit`,
      complexity: "Time Complexity: O(n), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/"
    },
    {
      title: "Minimize the Heights II",
      description: "Minimize the maximum difference between heights of towers after increasing or decreasing each tower by k.",
      approach: `
1. Sort the array
2. Initialize result as difference between first and last element
3. Consider all possible splits where we add k to first i elements and subtract k from rest
4. Track minimum possible maximum difference`,
      algorithm: `
• Time complexity: O(n log n)
• Space complexity: O(1)
• Requires sorting
• Handles edge cases where k is larger than some differences`,
      cppcode: `#include <vector>
#include <algorithm>
using namespace std;

int getMinDiff(int arr[], int n, int k) {
    sort(arr, arr + n);
    int ans = arr[n - 1] - arr[0];
    
    for (int i = 1; i < n; i++) {
        if (arr[i] - k < 0) continue;
        int curr_max = max(arr[i - 1] + k, arr[n - 1] - k);
        int curr_min = min(arr[0] + k, arr[i] - k);
        ans = min(ans, curr_max - curr_min);
    }
    return ans;
}`,
      javacode: `import java.util.Arrays;

public class Solution {
    public int getMinDiff(int[] arr, int n, int k) {
        Arrays.sort(arr);
        int ans = arr[n - 1] - arr[0];
        
        for (int i = 1; i < n; i++) {
            if (arr[i] - k < 0) continue;
            int currMax = Math.max(arr[i - 1] + k, arr[n - 1] - k);
            int currMin = Math.min(arr[0] + k, arr[i] - k);
            ans = Math.min(ans, currMax - currMin);
        }
        return ans;
    }
}`,
      pythoncode: `def getMinDiff(arr, n, k):
    arr.sort()
    ans = arr[-1] - arr[0]
    
    for i in range(1, n):
        if arr[i] - k < 0:
            continue
        curr_max = max(arr[i-1] + k, arr[-1] - k)
        curr_min = min(arr[0] + k, arr[i] - k)
        ans = min(ans, curr_max - curr_min)
    return ans`,
      complexity: "Time Complexity: O(n log n), Space Complexity: O(1)",
      link: "https://www.geeksforgeeks.org/minimize-the-maximum-difference-between-the-heights/"
    },
    {
      title: "Kadane's Algorithm",
      description: "Find the maximum sum of a contiguous subarray.",
      approach: `
1. Track maximum sum ending at current position
2. Track overall maximum sum
3. Reset current sum if it becomes negative`,
      algorithm: `
• Time complexity: O(n)
• Space complexity: O(1)
• Handles all negative numbers case
• Single pass through the array`,
      cppcode: `#include <vector>
#include <climits>
using namespace std;

int maxSubArray(vector<int>& nums) {
    int max_sum = INT_MIN;
    int current_sum = 0;
    
    for (int num : nums) {
        current_sum += num;
        if (current_sum > max_sum) {
            max_sum = current_sum;
        }
        if (current_sum < 0) {
            current_sum = 0;
        }
    }
    return max_sum;
}`,
      javacode: `public class Solution {
    public int maxSubArray(int[] nums) {
        int maxSum = Integer.MIN_VALUE;
        int currentSum = 0;
        
        for (int num : nums) {
            currentSum += num;
            if (currentSum > maxSum) {
                maxSum = currentSum;
            }
            if (currentSum < 0) {
                currentSum = 0;
            }
        }
        return maxSum;
    }
}`,
      pythoncode: `def maxSubArray(nums):
    max_sum = float('-inf')
    current_sum = 0
    
    for num in nums:
        current_sum += num
        if current_sum > max_sum:
            max_sum = current_sum
        if current_sum < 0:
            current_sum = 0
    return max_sum`,
      complexity: "Time Complexity: O(n), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/maximum-subarray/"
    },
    {
      title: "Maximum Product Subarray",
      description: "Find the contiguous subarray within an array that has the largest product.",
      approach: `
1. Track both maximum and minimum product at each position
2. Update these values based on current element
3. Handle negative numbers by swapping max and min when negative number is encountered`,
      algorithm: `
• Time complexity: O(n)
• Space complexity: O(1)
• Handles negative numbers
• Tracks both max and min products`,
      cppcode: `#include <vector>
#include <climits>
using namespace std;

int maxProduct(vector<int>& nums) {
    int max_prod = nums[0];
    int min_prod = nums[0];
    int result = nums[0];
    
    for (int i = 1; i < nums.size(); i++) {
        if (nums[i] < 0) {
            swap(max_prod, min_prod);
        }
        
        max_prod = max(nums[i], max_prod * nums[i]);
        min_prod = min(nums[i], min_prod * nums[i]);
        
        result = max(result, max_prod);
    }
    return result;
}`,
      javacode: `public class Solution {
    public int maxProduct(int[] nums) {
        int maxProd = nums[0];
        int minProd = nums[0];
        int result = nums[0];
        
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] < 0) {
                int temp = maxProd;
                maxProd = minProd;
                minProd = temp;
            }
            
            maxProd = Math.max(nums[i], maxProd * nums[i]);
            minProd = Math.min(nums[i], minProd * nums[i]);
            
            result = Math.max(result, maxProd);
        }
        return result;
    }
}`,
      pythoncode: `def maxProduct(nums):
    max_prod = min_prod = result = nums[0]
    
    for num in nums[1:]:
        if num < 0:
            max_prod, min_prod = min_prod, max_prod
        
        max_prod = max(num, max_prod * num)
        min_prod = min(num, min_prod * num)
        
        result = max(result, max_prod)
    return result`,
      complexity: "Time Complexity: O(n), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/maximum-product-subarray/"
    },
    {
      title: "Max Circular Subarray Sum",
      description: "Find the maximum sum of a circular subarray (subarrays can wrap around the end).",
      approach: `
1. Compute maximum subarray sum using Kadane's algorithm (non-circular case)
2. Compute minimum subarray sum using modified Kadane's
3. Total sum minus minimum sum gives circular maximum
4. Return maximum of non-circular and circular cases`,
      algorithm: `
• Time complexity: O(n)
• Space complexity: O(1)
• Combines standard Kadane's with circular case
• Handles all negative numbers case`,
      cppcode: `#include <vector>
#include <climits>
using namespace std;

int maxSubarraySumCircular(vector<int>& nums) {
    int total = 0, max_sum = INT_MIN, min_sum = INT_MAX;
    int current_max = 0, current_min = 0;
    
    for (int num : nums) {
        total += num;
        
        // Kadane's for max subarray
        current_max = max(num, current_max + num);
        max_sum = max(max_sum, current_max);
        
        // Modified Kadane's for min subarray
        current_min = min(num, current_min + num);
        min_sum = min(min_sum, current_min);
    }
    
    if (max_sum > 0) {
        return max(max_sum, total - min_sum);
    }
    return max_sum;
}`,
      javacode: `public class Solution {
    public int maxSubarraySumCircular(int[] nums) {
        int total = 0, maxSum = Integer.MIN_VALUE, minSum = Integer.MAX_VALUE;
        int currentMax = 0, currentMin = 0;
        
        for (int num : nums) {
            total += num;
            
            // Kadane's for max subarray
            currentMax = Math.max(num, currentMax + num);
            maxSum = Math.max(maxSum, currentMax);
            
            // Modified Kadane's for min subarray
            currentMin = Math.min(num, currentMin + num);
            minSum = Math.min(minSum, currentMin);
        }
        
        if (maxSum > 0) {
            return Math.max(maxSum, total - minSum);
        }
        return maxSum;
    }
}`,
      pythoncode: `def maxSubarraySumCircular(nums):
    total = 0
    max_sum = float('-inf')
    min_sum = float('inf')
    current_max = current_min = 0
    
    for num in nums:
        total += num
        
        # Kadane's for max subarray
        current_max = max(num, current_max + num)
        max_sum = max(max_sum, current_max)
        
        # Modified Kadane's for min subarray
        current_min = min(num, current_min + num)
        min_sum = min(min_sum, current_min)
    
    if max_sum > 0:
        return max(max_sum, total - min_sum)
    return max_sum`,
      complexity: "Time Complexity: O(n), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/maximum-sum-circular-subarray/"
    },
    {
      title: "Smallest Positive Missing Number",
      description: "Find the smallest missing positive integer in an unsorted array.",
      approach: `
1. Segregate positive numbers
2. Mark presence of numbers by making corresponding indices negative
3. Scan to find first positive index which indicates missing number`,
      algorithm: `
• Time complexity: O(n)
• Space complexity: O(1)
• In-place operation using array indices
• Handles duplicates and large numbers`,
      cppcode: `#include <vector>
using namespace std;

int firstMissingPositive(vector<int>& nums) {
    int n = nums.size();
    
    // Step 1: Mark non-positive numbers as n+1 (irrelevant)
    for (int i = 0; i < n; i++) {
        if (nums[i] <= 0) {
            nums[i] = n + 1;
        }
    }
    
    // Step 2: Mark indices for numbers present
    for (int i = 0; i < n; i++) {
        int num = abs(nums[i]);
        if (num <= n) {
            nums[num - 1] = -abs(nums[num - 1]);
        }
    }
    
    // Step 3: Find first positive index
    for (int i = 0; i < n; i++) {
        if (nums[i] > 0) {
            return i + 1;
        }
    }
    
    return n + 1;
}`,
      javacode: `public class Solution {
    public int firstMissingPositive(int[] nums) {
        int n = nums.length;
        
        // Step 1: Mark non-positive numbers as n+1 (irrelevant)
        for (int i = 0; i < n; i++) {
            if (nums[i] <= 0) {
                nums[i] = n + 1;
            }
        }
        
        // Step 2: Mark indices for numbers present
        for (int i = 0; i < n; i++) {
            int num = Math.abs(nums[i]);
            if (num <= n) {
                nums[num - 1] = -Math.abs(nums[num - 1]);
            }
        }
        
        // Step 3: Find first positive index
        for (int i = 0; i < n; i++) {
            if (nums[i] > 0) {
                return i + 1;
            }
        }
        
        return n + 1;
    }
}`,
      pythoncode: `def firstMissingPositive(nums):
    n = len(nums)
    
    # Step 1: Mark non-positive numbers as n+1 (irrelevant)
    for i in range(n):
        if nums[i] <= 0:
            nums[i] = n + 1
    
    # Step 2: Mark indices for numbers present
    for i in range(n):
        num = abs(nums[i])
        if num <= n:
            nums[num - 1] = -abs(nums[num - 1])
    
    # Step 3: Find first positive index
    for i in range(n):
        if nums[i] > 0:
            return i + 1
    
    return n + 1`,
      complexity: "Time Complexity: O(n), Space Complexity: O(1)",
      link: "https://leetcode.com/problems/first-missing-positive/"
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
        Array Problems with Solutions
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

export default Narray1;