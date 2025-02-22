import React, { useState } from "react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { tomorrow } from "react-syntax-highlighter/dist/esm/styles/hljs";

function Algo1() {
  const [visibleCodeIndex, setVisibleCodeIndex] = useState(null);

  const toggleCodeVisibility = (index) => {
    if (visibleCodeIndex === index) {
      setVisibleCodeIndex(null); 
    } else {
      setVisibleCodeIndex(index); 
    }
  };

  const codeExamples = [

    {

        title: "Bubble Sort",
        description: "Bubble Sort is a simple sorting algorithm that repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order. This process is repeated until the list is sorted.",
        code: `
    #include <iostream>
    #include <vector>
    using namespace std;
    void bubbleSort(vector<int>& arr) {
      int n = arr.size();
      bool swapped;
      
      for (int i = 0; i < n - 1; ++i) {
          swapped = false;
          for (int j = 0; j < n - i - 1; ++j) {
              if (arr[j] > arr[j + 1]) {
                  swap(arr[j], arr[j + 1]);
                  swapped = true;
              }
          }
          if (!swapped) break;
      }
    }
    void printArray(const vector<int>& arr) {
      for (int num : arr) {
          cout << num << " ";
      }
      cout << endl;
    }
    int main() {
      vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
      cout << "Original array: ";
      printArray(arr);
      bubbleSort(arr);
      cout << "Sorted array: ";
      printArray(arr);
      
      return 0;
    }
        `,
        language: "cpp",
        complexity: "Time Complexity: O(n^2), Space Complexity: O(1)",
        link: "https://www.geeksforgeeks.org/bubble-sort/"
      },
      {
        title: "Selection Sort",
        description: "Implementation of selection sort algorithm in C++.",
        code: `
    #include <iostream>
    #include <vector>
    using namespace std;
    void selectionSort(vector<int>& arr) {
      int n = arr.size();
      for (int i = 0; i < n - 1; i++) {
          int minIndex = i;
          for (int j = i + 1; j < n; j++) {
              if (arr[j] < arr[minIndex]) {
                  minIndex = j;
              }
          }
          swap(arr[i], arr[minIndex]);
      }
    }
    int main() {
      vector<int> arr = {64, 25, 12, 22, 11};
      cout << "Original array: ";
      for (int num : arr) {
          cout << num << " ";
      }
      cout << endl;
      selectionSort(arr);
      cout << "Sorted array: ";
      for (int num : arr) {
          cout << num << " ";
      }
      cout << endl;
      return 0;
    }
        `,
        language: "cpp",
        complexity: "Time Complexity: O(n^2), Space Complexity: O(1)",
        link: "https://www.geeksforgeeks.org/selection-sort/"
      },
      {
        title: "Insertion Sort",
        description: "Implementation of insertion sort algorithm in C++.",
        code: `
    #include <iostream>
    #include <vector>
    using namespace std;
    void insertionSort(vector<int>& arr) {
      int n = arr.size();
      for (int i = 1; i < n; i++) {
          int key = arr[i];
          int j = i - 1;
          while (j >= 0 && arr[j] > key) {
              arr[j + 1] = arr[j];
              j = j - 1;
          }
          arr[j + 1] = key;
      }
    }
    int main() {
      vector<int> arr = {64, 25, 12, 22, 11};
      cout << "Original array: ";
      for (int num : arr) {
          cout << num << " ";
      }
      cout << endl;
      insertionSort(arr);
      cout << "Sorted array: ";
      for (int num : arr) {
          cout << num << " ";
      }
      cout << endl;
      return 0;
    }
        `,
        language: "cpp",
        complexity: "Time Complexity: O(n^2), Space Complexity: O(1)",
        link: "https://www.geeksforgeeks.org/insertion-sort/"
      },
      {
        title: "Merge Sort",
        description: "Merge Sort is a divide-and-conquer algorithm that divides the array into halves, recursively sorts them, and then merges the sorted halves.",
        code: `
    #include <iostream>
    #include <vector>
    using namespace std;
    void merge(vector<int>& arr, int left, int mid, int right) {
      int n1 = mid - left + 1;
      int n2 = right - mid;
      vector<int> L(n1), R(n2);
      for (int i = 0; i < n1; i++)
          L[i] = arr[left + i];
      for (int j = 0; j < n2; j++)
          R[j] = arr[mid + 1 + j];
      int i = 0, j = 0, k = left;
      while (i < n1 && j < n2) {
          if (L[i] <= R[j]) {
              arr[k] = L[i];
              i++;
          } else {
              arr[k] = R[j];
              j++;
          }
          k++;
      }
      while (i < n1) {
          arr[k] = L[i];
          i++;
          k++;
      }
      while (j < n2) {
          arr[k] = R[j];
          j++;
          k++;
      }
    }
    void mergeSort(vector<int>& arr, int left, int right) {
      if (left < right) {
          int mid = left + (right - left) / 2;
          mergeSort(arr, left, mid);
          mergeSort(arr, mid + 1, right);
          merge(arr, left, mid, right);
      }
    }
    int main() {
      vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
      cout << "Original array: ";
      for (int num : arr) {
          cout << num << " ";
      }
      cout << endl;
      mergeSort(arr, 0, arr.size() - 1);
      cout << "Sorted array: ";
      for (int num : arr) {
          cout << num << " ";
      }
      cout << endl;
      return 0;
    }
        `,
        language: "cpp",
        complexity: "Time Complexity: O(n log n), Space Complexity: O(n)",
        link: "https://www.geeksforgeeks.org/merge-sort/"
      },
      {
        title: "Quick Sort",
        description: "Quick Sort is a divide-and-conquer algorithm that picks an element as a pivot and partitions the array around the pivot.",
        code: `
    #include <iostream>
    #include <vector>
    using namespace std;
    int partition(vector<int>& arr, int low, int high) {
      int pivot = arr[high];
      int i = (low - 1);
      for (int j = low; j < high; j++) {
          if (arr[j] < pivot) {
              i++;
              swap(arr[i], arr[j]);
          }
      }
      swap(arr[i + 1], arr[high]);
      return (i + 1);
    }
    void quickSort(vector<int>& arr, int low, int high) {
      if (low < high) {
          int pi = partition(arr, low, high);
          quickSort(arr, low, pi - 1);
          quickSort(arr, pi + 1, high);
      }
    }
    int main() {
      vector<int> arr = {64, 34, 25, 12, 22, 11, 90};
      cout << "Original array: ";
      for (int num : arr) {
          cout << num << " ";
      }
      cout << endl;
      quickSort(arr, 0, arr.size() - 1);
      cout << "Sorted array: ";
      for (int num : arr) {
          cout << num << " ";
      }
      cout << endl;
      return 0;
    }
        `,
        language: "cpp",
        complexity: "Time Complexity: O(n log n), Space Complexity: O(n)",
        link: "https://www.geeksforgeeks.org/quick-sort/"
      },
      {
        title: "Heap Sort using Min-Heap and Max-Heap",
        description: "This program demonstrates sorting an array using both Min-Heap and Max-Heap algorithms. It includes separate heapify functions for each heap type and sorts the array in ascending or descending order accordingly.",
        code: `
    #include <iostream>
    #include <algorithm>
    using namespace std;
    void minHeapify(int n, int a[], int i) {
      int smallest = i;
      int left = 2 * i + 1;
      int right = 2 * i + 2;
      if (left < n && a[left] < a[smallest])
          smallest = left;
      if (right < n && a[right] < a[smallest])
          smallest = right;
      if (smallest != i) {
          swap(a[smallest], a[i]);
          minHeapify(n, a, smallest);
      }
    }
    void maxHeapify(int n, int a[], int i) {
      int largest = i;
      int left = 2 * i + 1;
      int right = 2 * i + 2;
      if (left < n && a[left] > a[largest])
          largest = left;
      if (right < n && a[right] > a[largest])
          largest = right;
      if (largest != i) {
          swap(a[largest], a[i]);
          maxHeapify(n, a, largest);
      }
    }
    void heapSortMin(int n, int a[]) {
      for (int i = n / 2 - 1; i >= 0; i--)
          minHeapify(n, a, i);
      for (int i = n - 1; i > 0; i--) {
          swap(a[0], a[i]);
          minHeapify(i, a, 0);
      }
    }
    void heapSortMax(int n, int a[]) {
      for (int i = n / 2 - 1; i >= 0; i--)
          maxHeapify(n, a, i);
      for (int i = n - 1; i > 0; i--) {
          swap(a[0], a[i]);
          maxHeapify(i, a, 0);
      }
    }
    int main() {
      int n;
      cout << "Enter the size of the array: ";
      cin >> n;
      int a[n];
      cout << "Enter the elements of the array: ";
      for (int i = 0; i < n; i++)
          cin >> a[i];
      cout << "\nArray before sorting: ";
      for (int i = 0; i < n; i++)
          cout << a[i] << " ";
      cout << endl;
      int minHeapArray[n];
      copy(a, a + n, minHeapArray);
      heapSortMin(n, minHeapArray);
      cout << "\nArray after Min-Heap Sort: ";
      for (int i = 0; i < n; i++)
          cout << minHeapArray[i] << " ";
      cout << endl;
      int maxHeapArray[n];
      copy(a, a + n, maxHeapArray);
      heapSortMax(n, maxHeapArray);
      cout << "\nArray after Max-Heap Sort: ";
      for (int i = 0; i < n; i++)
          cout << maxHeapArray[i] << " ";
      cout << endl;
      return 0;
    }
        `,
        language: "cpp",
        complexity: "O(n log n) for both Min-Heap and Max-Heap sorting",
        link: "https://www.geeksforgeeks.org/heap-sort/"
      },
      {
        title: "Convert Min Heap to Max Heap",
        description: "This program demonstrates sorting an array using both Min-Heap and Max-Heap algorithms. It includes separate heapify functions for each heap type and sorts the array in ascending or descending order accordingly.",
        code: `
    #include<bits/stdc++.h>
    using namespace std;
    class Solution {
    private:
      void heapify(int n, vector<int>& arr, int i) {
          int largest = i;
          int left = 2 * i + 1;
          int right = 2 * i + 2;
          if (left < n && arr[left] > arr[largest])
              largest = left;
          if (right < n && arr[right] > arr[largest])
              largest = right;
          if (largest != i) {
              swap(arr[i], arr[largest]);
              heapify(n, arr, largest);
          }
      }
      void buildMaxHeap(vector<int>& arr, int n) {
          for (int i = n / 2 - 1; i >= 0; i--)
              heapify(n, arr, i);
      }
    public:
      void convertMinToMaxHeap(vector<int> &arr, int N) {
          buildMaxHeap(arr, N);
      }
    };
    int main() {
      int t = 1;
      cin >> t;
      while (t--) {
          int n; cin >> n;
          vector<int> vec(n);
          for (int i = 0; i < n; i++) cin >> vec[i];
          Solution obj;
          obj.convertMinToMaxHeap(vec, n);
          for (int i = 0; i < n; i++) cout << vec[i] << " ";
          cout << endl;
          cout << "~" << "\n";
      }
      return 0;
    }
        `,
        language: "cpp",
        complexity: "O(n log n) for both Min-Heap and Max-Heap sorting",
        link: "https://www.geeksforgeeks.org/convert-min-heap-to-max-heap/"
      }
    
  ];

  return (
    <div className="container mx-auto px-6 py-12 bg-gradient-to-br from-indigo-50 to-purple-50 rounded-2xl shadow-2xl max-w-7xl">
      <h1 className="text-6xl font-extrabold text-center text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600 mb-12">
        Sorting Algorithms
      </h1>
      {codeExamples.map((example, index) => (
        <div
          key={index}
          className="mb-12 p-8 bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all transform hover:scale-102 duration-300 border border-indigo-100"
        >
          <h2 className="text-4xl font-bold text-indigo-800 mb-6">{example.title}</h2>
          <p className="text-gray-700 mb-6 text-lg leading-relaxed">{example.description}</p>
          <p className="text-gray-800 font-semibold mb-6">
            <span className="text-indigo-700 font-bold">Complexity:</span> {example.complexity}
          </p>
          <div className="flex gap-4 mb-6"> 
            <a
              href={example.link}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block bg-gradient-to-r from-indigo-600 to-purple-600 text-white px-6 py-3 rounded-lg hover:from-indigo-700 hover:to-purple-700 transition-all transform hover:scale-105"
            >
              View Problem
            </a>
            <button
              onClick={() => toggleCodeVisibility(index)}
              className="inline-block bg-gradient-to-r from-indigo-500 to-purple-500 text-white px-6 py-3 rounded-lg hover:from-indigo-600 hover:to-purple-600 transition-all transform hover:scale-105"
            >
              {visibleCodeIndex === index ? "Hide Code" : "Show Code"}
            </button>
          </div>
          {visibleCodeIndex === index && (
            <div className="rounded-lg overflow-hidden border-2 border-indigo-100">
              <SyntaxHighlighter
                language={example.language}
                style={tomorrow}
                customStyle={{
                  padding: "1.5rem",
                  fontSize: "0.95rem",
                  background: "#f9f9f9",
                  borderRadius: "0.5rem",
                }}
              >
                {example.code}
              </SyntaxHighlighter>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default Algo1;


