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

## Your Tasks

### Task 1: Implement `euclidean_distance()`
Calculate distance between two points:
```python
distance = sqrt(sum((a - b)² for each dimension))
```

### Task 2: Implement `knn_classify()`
Classify a point using KNN:
1. Calculate distance from point to all data points
2. Sort by distance
3. Take K nearest neighbors
4. Vote: return the most common label

### Task 3: Implement `knn_regression()`
Predict a value using KNN:
1. Calculate distance from point to all data points
2. Sort by distance
3. Take K nearest neighbors
4. Return the average of their values

### Task 4: Implement `normalize_features()`
Normalize all features to 0-1 range:
1. For each feature (column):
   - Find min and max values
   - Transform: `(value - min) / (max - min)`
2. Return normalized data

## Example

```python
# Classification
data = [
    ([1, 1], "A"),
    ([2, 2], "A"),
    ([8, 8], "B"),
    ([9, 9], "B")
]
>>> knn_classify(data, [1.5, 1.5], k=3)
'A'  # 3 nearest are all 'A'

# Regression
data = [
    ([1], 10),
    ([2], 20),
    ([3], 30)
]
>>> knn_regression(data, [2.5], k=2)
25.0  # Average of 20 and 30

# Distance
>>> euclidean_distance([0, 0], [3, 4])
5.0  # 3-4-5 triangle!
```

## Testing
```bash
python -m pytest tests/ -v
```

## Hints

### Euclidean Distance
```python
import math
def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
```

### Sorting by Distance
```python
# Create list of (distance, label) tuples
distances = [(euclidean_distance(point, features), label) 
             for features, label in data]
distances.sort()  # Sorts by first element (distance)
```

### Voting (Classification)
```python
from collections import Counter
labels = [label for _, label in nearest_k]
most_common = Counter(labels).most_common(1)[0][0]
```

### Normalization
Handle the edge case where min == max (all values are the same).

## Real-World Applications
- **Recommendation systems**: Find similar users/items
- **Image recognition**: Classify images by similar examples
- **Medical diagnosis**: Predict based on similar patient cases

## Submission
Commit and push your completed `knn.py` file.
