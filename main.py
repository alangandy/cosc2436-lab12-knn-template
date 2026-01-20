#!/usr/bin/env python3
"""
Lab 12: K-Nearest Neighbors - Interactive Tutorial
===================================================

🎯 GOAL: Implement KNN algorithm in knn.py

📚 K-NEAREST NEIGHBORS (Chapter 12):
------------------------------------
KNN is a simple but powerful machine learning algorithm!

HOW IT WORKS:
1. To classify a new point, find the K closest points in your data
2. Look at their labels
3. Vote! The most common label wins.

USE CASES:
- Classification: Is this email spam or not?
- Regression: What price should this house be?
- Recommendation: What movies would you like?

HOW TO RUN:
-----------
    python main.py           # Run this tutorial
    python -m pytest tests/ -v   # Run the grading tests
"""

import math
from knn import euclidean_distance, knn_classify, knn_regression, normalize_features


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def knn_concept() -> None:
    """Explain the KNN concept."""
    print_header("K-NEAREST NEIGHBORS CONCEPT")
    
    print("""
    INTUITION: "You are the average of your friends"
    
    To predict something about a new data point:
    1. Find the K most similar data points (neighbors)
    2. Use their values to make a prediction
    
    CLASSIFICATION (predict a category):
    - Find K nearest neighbors
    - Each neighbor votes for their class
    - Majority wins!
    
    Example: Is this fruit an apple or orange?
    - Find 3 nearest fruits in our database
    - 2 are apples, 1 is orange
    - Prediction: Apple! (2 votes to 1)
    
    REGRESSION (predict a number):
    - Find K nearest neighbors
    - Average their values
    
    Example: What should this house cost?
    - Find 3 nearest houses (by features)
    - Prices: $200k, $220k, $210k
    - Prediction: $210k (average)
    
    CHOOSING K:
    - Too small (K=1): Sensitive to noise
    - Too large (K=all): Just predicts the average
    - Rule of thumb: K = sqrt(n) where n = dataset size
    - Always use odd K for binary classification (avoids ties)
    """)


def demo_euclidean_distance() -> None:
    """Demonstrate Euclidean distance."""
    print_header("EUCLIDEAN DISTANCE")
    
    print("""
    To find "nearest" neighbors, we need to measure distance!
    
    EUCLIDEAN DISTANCE (straight-line distance):
    
    For 2D points (x1, y1) and (x2, y2):
        distance = sqrt((x2-x1)² + (y2-y1)²)
    
    For N-dimensional points:
        distance = sqrt(sum((a[i] - b[i])² for each dimension i))
    
    EXAMPLE:
    Point A = (0, 0)
    Point B = (3, 4)
    
    Distance = sqrt((3-0)² + (4-0)²)
             = sqrt(9 + 16)
             = sqrt(25)
             = 5
    
    PYTHON IMPLEMENTATION:
    ----------------------
    import math
    
    def euclidean_distance(p1, p2):
        total = 0
        for i in range(len(p1)):
            total += (p1[i] - p2[i]) ** 2
        return math.sqrt(total)
    
    # Or using list comprehension:
    def euclidean_distance(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    """)
    
    print("Testing euclidean_distance():")
    test_cases = [
        ([0, 0], [3, 4], 5.0),
        ([0, 0], [0, 0], 0.0),
        ([1, 2, 3], [4, 5, 6], 5.196),  # sqrt(27)
    ]
    
    for p1, p2, expected in test_cases:
        try:
            result = euclidean_distance(p1, p2)
            if result is not None and abs(result - expected) < 0.01:
                print(f"    distance({p1}, {p2}) = {result:.3f} ✅")
            elif result is None:
                print(f"    distance({p1}, {p2}) = None ❌")
            else:
                print(f"    distance({p1}, {p2}) = {result:.3f} ❌ (expected {expected})")
        except Exception as e:
            print(f"    distance({p1}, {p2}) ❌ Error: {e}")


def demo_knn_classify() -> None:
    """Demonstrate KNN classification."""
    print_header("KNN CLASSIFICATION")
    
    print("""
    ALGORITHM:
    1. Calculate distance from new point to ALL data points
    2. Sort by distance
    3. Take the K nearest neighbors
    4. Count votes for each class
    5. Return the class with most votes
    
    EXAMPLE:
    Data points:
    - (1, 1) → Class A
    - (2, 2) → Class A
    - (8, 8) → Class B
    - (9, 9) → Class B
    
    New point: (1.5, 1.5), K=3
    
    Distances:
    - to (1,1): 0.71 → Class A
    - to (2,2): 0.71 → Class A
    - to (8,8): 9.19 → Class B
    - to (9,9): 10.61 → Class B
    
    3 nearest: (1,1), (2,2), (8,8)
    Votes: A=2, B=1
    Prediction: A ✅
    """)
    
    data = [
        ([1, 1], "A"),
        ([2, 2], "A"),
        ([8, 8], "B"),
        ([9, 9], "B"),
    ]
    
    print(f"Data: {data}")
    
    test_cases = [
        ([1.5, 1.5], 3, "A"),
        ([8.5, 8.5], 3, "B"),
        ([5, 5], 3, None),  # Could be either, depends on tie-breaking
    ]
    
    print("\nTesting knn_classify():")
    for point, k, expected in test_cases:
        try:
            result = knn_classify(data, point, k)
            if expected is None:
                print(f"    classify({point}, k={k}) = '{result}' (tie-breaker)")
            elif result == expected:
                print(f"    classify({point}, k={k}) = '{result}' ✅")
            elif result is None:
                print(f"    classify({point}, k={k}) = None ❌")
            else:
                print(f"    classify({point}, k={k}) = '{result}' ❌ (expected '{expected}')")
        except Exception as e:
            print(f"    classify({point}, k={k}) ❌ Error: {e}")


def demo_knn_regression() -> None:
    """Demonstrate KNN regression."""
    print_header("KNN REGRESSION")
    
    print("""
    For regression, instead of voting, we AVERAGE the values!
    
    EXAMPLE: Predict bread price
    
    Data (loaves sold, price):
    - (300, $1.50)
    - (225, $1.80)
    - (75, $2.00)
    - (200, $1.90)
    - (150, $1.85)
    
    New bakery sells 130 loaves. What price?
    
    Find 3 nearest by loaves sold:
    - 150 loaves → $1.85
    - 200 loaves → $1.90
    - 75 loaves → $2.00
    
    Average: ($1.85 + $1.90 + $2.00) / 3 = $1.92
    """)
    
    data = [
        ([300], 1.50),
        ([225], 1.80),
        ([75], 2.00),
        ([200], 1.90),
        ([150], 1.85),
    ]
    
    print(f"Data (features, value): {data}")
    
    print("\nTesting knn_regression():")
    test_cases = [
        ([130], 3, 1.92),  # Average of 1.85, 1.90, 2.00
        ([300], 1, 1.50),  # Exact match
    ]
    
    for point, k, expected in test_cases:
        try:
            result = knn_regression(data, point, k)
            if result is not None and abs(result - expected) < 0.1:
                print(f"    regression({point}, k={k}) = ${result:.2f} ✅")
            elif result is None:
                print(f"    regression({point}, k={k}) = None ❌")
            else:
                print(f"    regression({point}, k={k}) = ${result:.2f} ❌ (expected ${expected:.2f})")
        except Exception as e:
            print(f"    regression({point}, k={k}) ❌ Error: {e}")


def demo_normalization() -> None:
    """Demonstrate feature normalization."""
    print_header("FEATURE NORMALIZATION")
    
    print("""
    PROBLEM: Features with different scales
    
    Example: Predicting movie ratings
    - Feature 1: Age (0-100)
    - Feature 2: Income ($0-$1,000,000)
    
    Without normalization, income dominates the distance!
    A $10,000 difference matters more than a 50-year age difference.
    
    SOLUTION: Normalize all features to 0-1 range
    
    FORMULA:
    normalized = (value - min) / (max - min)
    
    EXAMPLE:
    Ages: [20, 40, 60]
    Min = 20, Max = 60
    
    Normalized:
    - 20 → (20-20)/(60-20) = 0.0
    - 40 → (40-20)/(60-20) = 0.5
    - 60 → (60-20)/(60-20) = 1.0
    """)
    
    print("Testing normalize_features():")
    data = [[20, 1000], [40, 5000], [60, 3000]]
    
    try:
        result = normalize_features(data)
        if result is None:
            print(f"    normalize({data}) = None ❌")
        else:
            print(f"    Input:  {data}")
            print(f"    Output: {result}")
            
            # Check if values are in 0-1 range
            valid = True
            for row in result:
                for val in row:
                    if val < 0 or val > 1:
                        valid = False
            
            if valid:
                print("    ✅ All values in [0, 1] range")
            else:
                print("    ❌ Some values outside [0, 1] range")
                
    except Exception as e:
        print(f"    ❌ Error: {e}")


def python_sorting_tips() -> None:
    """Tips for sorting in Python."""
    print_header("PYTHON SORTING TIPS")
    
    print("""
    KNN requires sorting by distance. Here's how in Python:
    
    SORTING BASICS:
    ---------------
    numbers = [3, 1, 4, 1, 5]
    sorted_numbers = sorted(numbers)  # [1, 1, 3, 4, 5]
    
    SORTING WITH KEY FUNCTION:
    --------------------------
    # Sort by second element of tuple
    data = [(1, 5), (2, 3), (3, 8)]
    sorted_data = sorted(data, key=lambda x: x[1])
    # Result: [(2, 3), (1, 5), (3, 8)]
    
    FOR KNN:
    --------
    # Calculate distances and sort
    distances = []
    for features, label in data:
        dist = euclidean_distance(point, features)
        distances.append((dist, label))
    
    # Sort by distance (first element)
    distances.sort()  # or sorted(distances)
    
    # Take K nearest
    k_nearest = distances[:k]
    
    COUNTING VOTES:
    ---------------
    from collections import Counter
    
    labels = ["A", "A", "B"]
    counts = Counter(labels)  # {"A": 2, "B": 1}
    most_common = counts.most_common(1)[0][0]  # "A"
    """)


def main():
    """Main entry point."""
    print("\n" + "🎯" * 30)
    print("   LAB 12: K-NEAREST NEIGHBORS")
    print("   Machine Learning Made Simple!")
    print("🎯" * 30)
    
    print("""
    📋 YOUR TASKS:
    1. Open knn.py
    2. Implement these functions:
       - euclidean_distance()
       - knn_classify()
       - knn_regression()
       - normalize_features()
    3. Run this file to test: python main.py
    4. Run pytest when ready: python -m pytest tests/ -v
    """)
    
    knn_concept()
    demo_euclidean_distance()
    demo_knn_classify()
    demo_knn_regression()
    demo_normalization()
    python_sorting_tips()
    
    print_header("CONGRATULATIONS!")
    print("""
    🎉 You've completed all 12 labs! 🎉
    
    You've learned:
    - Lab 01: Binary Search - O(log n) searching
    - Lab 02: Selection Sort - O(n²) sorting basics
    - Lab 03: Recursion - Functions calling themselves
    - Lab 04: Quicksort - Divide & Conquer, O(n log n)
    - Lab 05: Hash Tables - O(1) lookups
    - Lab 06: BFS - Shortest paths in unweighted graphs
    - Lab 07: Trees - Hierarchical data structures
    - Lab 08: AVL Trees - Self-balancing BSTs
    - Lab 09: Dijkstra - Shortest paths in weighted graphs
    - Lab 10: Greedy - Local optimum → global optimum
    - Lab 11: Dynamic Programming - Overlapping subproblems
    - Lab 12: KNN - Simple machine learning
    
    These are the building blocks of computer science!
    """)


if __name__ == "__main__":
    main()
