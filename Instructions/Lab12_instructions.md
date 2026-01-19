# Lab 12: K-Nearest Neighbors (KNN)

## Overview
In this lab, you will implement **K-Nearest Neighbors** from Chapter 12 of "Grokking Algorithms." KNN is a simple but powerful machine learning algorithm.

## Learning Objectives
- Understand classification vs regression
- Implement Euclidean distance
- Implement KNN classification (voting)
- Implement KNN regression (averaging)
- Understand feature normalization

## Background

### What is KNN?
KNN makes predictions based on the K most similar examples:
- **Classification**: Predict a category (vote among neighbors)
- **Regression**: Predict a number (average of neighbors)

### Distance Metrics
To find "nearest" neighbors, we need to measure distance. **Euclidean distance**:
```
distance = sqrt((x1-x2)² + (y1-y2)² + ...)
```

### Choosing K
- **Small K** (e.g., 1): Sensitive to noise
- **Large K** (e.g., 100): May include too many irrelevant neighbors
- **Rule of thumb**: Start with K = sqrt(n) where n is dataset size
- K should be odd for classification (avoids ties)

### Feature Normalization
When features have different scales (e.g., age 0-100, income 0-1000000), larger features dominate the distance. **Normalize** to 0-1 range:
```
normalized = (value - min) / (max - min)
```

---

## Complete Solutions

### Task 1: `euclidean_distance()` - Complete Implementation

```python
import math
from typing import List, Tuple
from collections import Counter

def euclidean_distance(p1: List[float], p2: List[float]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    From Chapter 12: sqrt(sum((a-b)² for each dimension))
    
    Example:
        >>> euclidean_distance([0, 0], [3, 4])
        5.0
    """
    # Sum of squared differences for each dimension
    squared_sum = sum((a - b) ** 2 for a, b in zip(p1, p2))
    
    # Return square root
    return math.sqrt(squared_sum)
```

**How it works:**
1. For each dimension, calculate the difference between coordinates
2. Square each difference
3. Sum all squared differences
4. Take the square root

Example: Distance from (0,0) to (3,4):
- Differences: (3-0)=3, (4-0)=4
- Squared: 9, 16
- Sum: 25
- Square root: 5.0 (the famous 3-4-5 triangle!)

---

### Task 2: `knn_classify()` - Complete Implementation

```python
def knn_classify(data: List[Tuple[List[float], str]], point: List[float], k: int = 3) -> str:
    """
    Classify a point using K-Nearest Neighbors.
    
    From Chapter 12: Find k nearest neighbors and vote on class.
    
    Args:
        data: List of (features, label) tuples
        point: Features of point to classify
        k: Number of neighbors to consider
    
    Returns:
        Predicted class label
    """
    # Calculate distance from point to all data points
    distances = []
    for features, label in data:
        dist = euclidean_distance(point, features)
        distances.append((dist, label))
    
    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[0])
    
    # Take k nearest neighbors
    k_nearest = distances[:k]
    
    # Vote: count labels and return most common
    labels = [label for dist, label in k_nearest]
    label_counts = Counter(labels)
    
    # Return the most common label
    most_common_label = label_counts.most_common(1)[0][0]
    return most_common_label
```

**How it works:**
1. Calculate distance from the query point to every point in the dataset
2. Sort all points by distance (closest first)
3. Take the K nearest neighbors
4. Count the labels among those K neighbors
5. Return the most common label (majority vote)

---

### Task 3: `knn_regression()` - Complete Implementation

```python
def knn_regression(data: List[Tuple[List[float], float]], point: List[float], k: int = 3) -> float:
    """
    Predict a value using KNN regression.
    
    From Chapter 12: Average the values of k nearest neighbors.
    
    Args:
        data: List of (features, value) tuples
        point: Features of point to predict
        k: Number of neighbors
    
    Returns:
        Predicted value (average of k nearest)
    """
    # Calculate distance from point to all data points
    distances = []
    for features, value in data:
        dist = euclidean_distance(point, features)
        distances.append((dist, value))
    
    # Sort by distance (ascending)
    distances.sort(key=lambda x: x[0])
    
    # Take k nearest neighbors
    k_nearest = distances[:k]
    
    # Average the values of k nearest neighbors
    values = [value for dist, value in k_nearest]
    average = sum(values) / len(values)
    
    return average
```

**How it works:**
1. Calculate distance from the query point to every point in the dataset
2. Sort all points by distance (closest first)
3. Take the K nearest neighbors
4. Calculate the average of their values
5. Return the average as the prediction

---

### Task 4: `normalize_features()` - Complete Implementation

```python
def normalize_features(data: List[List[float]]) -> List[List[float]]:
    """
    Normalize features to 0-1 range.
    
    From Chapter 12: Important when features have different scales.
    """
    if not data or not data[0]:
        return data
    
    num_features = len(data[0])
    num_samples = len(data)
    
    # Find min and max for each feature (column)
    mins = []
    maxs = []
    
    for feature_idx in range(num_features):
        feature_values = [data[row][feature_idx] for row in range(num_samples)]
        mins.append(min(feature_values))
        maxs.append(max(feature_values))
    
    # Normalize each value
    normalized = []
    for row in data:
        normalized_row = []
        for feature_idx, value in enumerate(row):
            min_val = mins[feature_idx]
            max_val = maxs[feature_idx]
            
            # Handle edge case where min == max (all values are the same)
            if max_val == min_val:
                normalized_row.append(0.0)
            else:
                normalized_value = (value - min_val) / (max_val - min_val)
                normalized_row.append(normalized_value)
        
        normalized.append(normalized_row)
    
    return normalized
```

**How it works:**
1. For each feature (column), find the minimum and maximum values
2. For each value, apply the formula: `(value - min) / (max - min)`
3. This transforms all values to the range [0, 1]
4. Handle edge case: if min == max, all values are the same, so normalize to 0

---

## Example Usage

```python
# Euclidean Distance
>>> euclidean_distance([0, 0], [3, 4])
5.0

>>> euclidean_distance([1, 2, 3], [4, 5, 6])
5.196...  # sqrt(9 + 9 + 9) = sqrt(27)


# KNN Classification
data = [
    ([1, 1], "A"),
    ([2, 2], "A"),
    ([3, 3], "A"),
    ([8, 8], "B"),
    ([9, 9], "B"),
    ([10, 10], "B")
]

>>> knn_classify(data, [1.5, 1.5], k=3)
'A'

# Step-by-step:
# Distances from [1.5, 1.5]:
#   [1,1]: sqrt(0.25 + 0.25) = 0.707 → A
#   [2,2]: sqrt(0.25 + 0.25) = 0.707 → A
#   [3,3]: sqrt(2.25 + 2.25) = 2.12  → A
#   [8,8]: sqrt(42.25 + 42.25) = 9.19 → B
#   ...
# 3 nearest: A, A, A
# Vote: A wins!


# KNN Regression
data = [
    ([1], 10),
    ([2], 20),
    ([3], 30),
    ([4], 40)
]

>>> knn_regression(data, [2.5], k=2)
25.0

# Step-by-step:
# Distances from [2.5]:
#   [1]: 1.5 → value 10
#   [2]: 0.5 → value 20
#   [3]: 0.5 → value 30
#   [4]: 1.5 → value 40
# 2 nearest: [2]→20, [3]→30
# Average: (20 + 30) / 2 = 25.0


# Feature Normalization
data = [
    [0, 0],
    [50, 100],
    [100, 200]
]

>>> normalize_features(data)
[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]

# Feature 0: min=0, max=100
#   0 → (0-0)/(100-0) = 0.0
#   50 → (50-0)/(100-0) = 0.5
#   100 → (100-0)/(100-0) = 1.0
#
# Feature 1: min=0, max=200
#   0 → 0.0
#   100 → 0.5
#   200 → 1.0
```

---

## Testing
```bash
python -m pytest tests/ -v
```

## Submission
Commit and push your completed `knn.py` file.
