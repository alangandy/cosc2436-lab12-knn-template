# Lab 12: K-Nearest Neighbors

## 1. Introduction and Objectives

### Overview
Implement the K-Nearest Neighbors (KNN) algorithm for classification and regression. Use Texas city data to build a recommendation system and explore machine learning fundamentals.

### Learning Objectives
- Understand the KNN algorithm for classification and regression
- Implement distance calculations and feature extraction
- Build a simple recommendation system
- Understand the basics of machine learning

### Prerequisites
- Complete Labs 1-11
- Read Chapter 12 in "Grokking Algorithms" (pages 229-246)

---

## 2. Algorithm Background

### What is KNN?
From Chapter 12: KNN is used for **classification** and **regression**:
- **Classification**: Categorizing into groups (e.g., comedy vs action movie)
- **Regression**: Predicting a number (e.g., how many stars a user will give)

### How KNN Works
1. Find the **K nearest neighbors** to your data point
2. For **classification**: Take a vote (majority wins)
3. For **regression**: Take the average

### Feature Extraction
From Chapter 12: Converting items into **numbers** that can be compared.

Example - Rating movies by:
- Comedy (1-5)
- Action (1-5)
- Drama (1-5)
- Horror (1-5)
- Romance (1-5)

### Distance Calculation
**Pythagorean formula** (Euclidean distance):
```
distance = √[(a₁-b₁)² + (a₂-b₂)² + ... + (aₙ-bₙ)²]
```

### Choosing K
From Chapter 12:
- Too small K → noisy, overfitting
- Too large K → too general
- Common choice: Start with K = 5

### Cosine Similarity (for recommendations)
Sometimes **angle** matters more than distance:
- Users who rate everything high vs low
- Cosine similarity measures the angle between vectors

---

## 3. Project Structure

```
lab12_knn/
├── knn.py             # KNN implementation
├── distance.py        # Distance calculations
├── main.py            # Main program
└── README.md          # Your lab report
```

---

## 4. Step-by-Step Implementation

### Step 1: Create `distance.py`

```python
"""
Lab 12: Distance Calculations
Feature extraction and distance metrics for KNN.
"""
from typing import List, Dict
import math


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    """
    Calculate Euclidean distance (straight line).
    
    From Chapter 12: The Pythagorean formula
    distance = √[(a₁-b₁)² + (a₂-b₂)² + ...]
    """
    if len(point1) != len(point2):
        raise ValueError("Points must have same number of features")
    
    squared_sum = sum((a - b) ** 2 for a, b in zip(point1, point2))
    return math.sqrt(squared_sum)


def manhattan_distance(point1: List[float], point2: List[float]) -> float:
    """
    Calculate Manhattan distance (grid-based).
    
    Sum of absolute differences.
    """
    if len(point1) != len(point2):
        raise ValueError("Points must have same number of features")
    
    return sum(abs(a - b) for a, b in zip(point1, point2))


def cosine_similarity(point1: List[float], point2: List[float]) -> float:
    """
    Calculate cosine similarity.
    
    From Chapter 12: Measures angle between vectors.
    Useful when magnitude doesn't matter (e.g., rating scales).
    
    Returns value between -1 and 1:
    - 1 = identical direction
    - 0 = perpendicular
    - -1 = opposite direction
    """
    if len(point1) != len(point2):
        raise ValueError("Points must have same number of features")
    
    dot_product = sum(a * b for a, b in zip(point1, point2))
    magnitude1 = math.sqrt(sum(a ** 2 for a in point1))
    magnitude2 = math.sqrt(sum(b ** 2 for b in point2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    return dot_product / (magnitude1 * magnitude2)


def normalize_features(data: List[List[float]]) -> List[List[float]]:
    """
    Normalize features to 0-1 range.
    
    Important because features with larger ranges
    would dominate the distance calculation.
    
    Example: Population (millions) vs Rating (1-5)
    Without normalization, population dominates!
    """
    if not data:
        return []
    
    n_features = len(data[0])
    
    # Find min and max for each feature
    mins = [min(row[i] for row in data) for i in range(n_features)]
    maxs = [max(row[i] for row in data) for i in range(n_features)]
    
    # Normalize each value
    normalized = []
    for row in data:
        norm_row = []
        for i, val in enumerate(row):
            if maxs[i] - mins[i] == 0:
                norm_row.append(0)
            else:
                norm_row.append((val - mins[i]) / (maxs[i] - mins[i]))
        normalized.append(norm_row)
    
    return normalized


def extract_city_features(city: Dict) -> List[float]:
    """
    Extract features from a city for comparison.
    
    From Chapter 12: Feature extraction is converting
    an item into a list of numbers.
    """
    return [
        city.get('population', 0) / 1000000,  # Scale to millions
        city.get('lat', 0),
        city.get('lon', 0),
    ]


def demonstrate_distance_metrics():
    """Show different distance metrics."""
    print("DISTANCE METRICS COMPARISON")
    print("=" * 50)
    
    # Two users' movie ratings [comedy, action, drama, horror, romance]
    user_a = [5, 1, 2, 1, 5]  # Likes comedy and romance
    user_b = [4, 2, 1, 1, 4]  # Similar to user A
    user_c = [1, 5, 4, 5, 1]  # Likes action and horror
    
    print("\nMovie ratings [comedy, action, drama, horror, romance]:")
    print(f"User A: {user_a}")
    print(f"User B: {user_b}")
    print(f"User C: {user_c}")
    
    print("\nEuclidean distances:")
    print(f"  A to B: {euclidean_distance(user_a, user_b):.2f}")
    print(f"  A to C: {euclidean_distance(user_a, user_c):.2f}")
    
    print("\nCosine similarities:")
    print(f"  A to B: {cosine_similarity(user_a, user_b):.3f}")
    print(f"  A to C: {cosine_similarity(user_a, user_c):.3f}")
    
    print("\nUser B is more similar to User A (smaller distance, higher cosine)")
```

### Step 2: Create `knn.py`

```python
"""
Lab 12: K-Nearest Neighbors Implementation
Classification and regression using KNN.
"""
from typing import List, Tuple, Dict, Any
from collections import Counter
from distance import euclidean_distance, normalize_features


class KNN:
    """
    K-Nearest Neighbors for classification and regression.
    
    From Chapter 12:
    - Classification: Categorizing into groups
    - Regression: Predicting a number
    """
    
    def __init__(self, k: int = 5):
        """
        Initialize KNN with k neighbors.
        
        From Chapter 12: k is how many neighbors to look at.
        """
        self.k = k
        self.data: List[Tuple[List[float], Any]] = []
    
    def fit(self, features: List[List[float]], labels: List[Any]) -> None:
        """
        Store training data.
        
        KNN is a "lazy learner" - it just stores the data!
        No actual training happens.
        """
        self.data = list(zip(features, labels))
        print(f"KNN fitted with {len(self.data)} data points, k={self.k}")
    
    def _find_neighbors(self, point: List[float]) -> List[Tuple[float, Any]]:
        """
        Find k nearest neighbors to a point.
        
        Returns list of (distance, label) tuples.
        """
        distances = []
        for features, label in self.data:
            dist = euclidean_distance(point, features)
            distances.append((dist, label))
        
        # Sort by distance and return k nearest
        distances.sort(key=lambda x: x[0])
        return distances[:self.k]
    
    def classify(self, point: List[float], verbose: bool = False) -> Any:
        """
        Classify a point using majority vote.
        
        From Chapter 12: "Take a vote" among k neighbors.
        """
        neighbors = self._find_neighbors(point)
        
        if verbose:
            print(f"\n{self.k} nearest neighbors:")
            for dist, label in neighbors:
                print(f"  Distance: {dist:.2f}, Label: {label}")
        
        # Majority vote
        labels = [label for _, label in neighbors]
        vote = Counter(labels).most_common(1)[0][0]
        
        if verbose:
            print(f"Classification: {vote}")
        
        return vote
    
    def regress(self, point: List[float], verbose: bool = False) -> float:
        """
        Predict a value using average of neighbors.
        
        From Chapter 12: "Take the average" of k neighbors.
        """
        neighbors = self._find_neighbors(point)
        
        if verbose:
            print(f"\n{self.k} nearest neighbors:")
            for dist, value in neighbors:
                print(f"  Distance: {dist:.2f}, Value: {value}")
        
        # Average of neighbor values
        values = [value for _, value in neighbors]
        prediction = sum(values) / len(values)
        
        if verbose:
            print(f"Prediction (average): {prediction:.2f}")
        
        return prediction


def movie_recommendation_example():
    """
    Build a movie recommendation system.
    
    From Chapter 12: Netflix example - predict how a user
    will rate a movie based on similar users.
    """
    print("\n" + "=" * 50)
    print("MOVIE RECOMMENDATION SYSTEM")
    print("=" * 50)
    
    # Users and their ratings [comedy, action, drama, horror, romance]
    users = {
        "Priyanka": [5, 1, 2, 1, 5],
        "Justin": [4, 2, 1, 1, 4],
        "Morpheus": [1, 5, 4, 5, 1],
        "Trinity": [2, 5, 5, 4, 1],
        "Neo": [1, 4, 4, 5, 2],
    }
    
    # Their rating for a specific movie
    movie_ratings = {
        "Priyanka": 4.5,
        "Justin": 4.0,
        "Morpheus": 2.0,
        "Trinity": 2.5,
        "Neo": 2.0,
    }
    
    print("\nUser preferences [comedy, action, drama, horror, romance]:")
    for name, prefs in users.items():
        print(f"  {name}: {prefs} → rated movie: {movie_ratings[name]}")
    
    # New user wants a recommendation
    new_user = [4, 1, 1, 1, 5]  # Similar to Priyanka and Justin
    print(f"\nNew user preferences: {new_user}")
    
    # Build KNN model
    knn = KNN(k=3)
    features = list(users.values())
    labels = [movie_ratings[name] for name in users.keys()]
    knn.fit(features, labels)
    
    # Predict rating
    predicted_rating = knn.regress(new_user, verbose=True)
    print(f"\nPredicted rating for new user: {predicted_rating:.1f} stars")


def city_classification_example():
    """
    Classify cities by region using KNN.
    
    Features: latitude, longitude
    Labels: North/South/Central Texas
    """
    print("\n" + "=" * 50)
    print("CITY CLASSIFICATION BY REGION")
    print("=" * 50)
    
    # Training data: cities with known regions
    cities = [
        # (lat, lon, region)
        (32.78, -96.80, "North"),    # Dallas
        (33.02, -96.70, "North"),    # Plano
        (32.76, -97.33, "North"),    # Fort Worth
        (29.76, -95.37, "South"),    # Houston
        (27.80, -97.40, "South"),    # Corpus Christi
        (25.90, -97.50, "South"),    # Brownsville
        (30.27, -97.74, "Central"),  # Austin
        (29.42, -98.49, "Central"),  # San Antonio
        (31.12, -97.73, "Central"),  # Killeen
    ]
    
    print("\nTraining cities:")
    for lat, lon, region in cities:
        print(f"  ({lat:.2f}, {lon:.2f}) → {region}")
    
    # Build KNN model
    knn = KNN(k=3)
    features = [[lat, lon] for lat, lon, _ in cities]
    labels = [region for _, _, region in cities]
    knn.fit(features, labels)
    
    # Classify new cities
    test_cities = [
        ("Lubbock", 33.58, -101.86),
        ("Laredo", 27.53, -99.48),
        ("El Paso", 31.76, -106.49),
    ]
    
    print("\nClassifying new cities:")
    for name, lat, lon in test_cities:
        region = knn.classify([lat, lon], verbose=False)
        print(f"  {name} ({lat:.2f}, {lon:.2f}) → {region}")


def picking_good_features():
    """
    Discuss feature selection from Chapter 12.
    """
    print("\n" + "=" * 50)
    print("PICKING GOOD FEATURES (from Chapter 12)")
    print("=" * 50)
    print("""
    From Chapter 12: "Picking the right features is important"
    
    GOOD FEATURES:
    - Directly relate to what you're predicting
    - Don't have bias
    - Are on similar scales (or normalize!)
    
    EXAMPLE - Recommending movies:
    
    BAD features:
    - User's age (not directly related to taste)
    - User's location (doesn't affect movie preference)
    
    GOOD features:
    - Ratings for comedy movies
    - Ratings for action movies
    - Ratings for drama movies
    
    EXAMPLE - Classifying cities:
    
    BAD features:
    - City name length (irrelevant)
    - Alphabetical order (irrelevant)
    
    GOOD features:
    - Latitude (relates to geography)
    - Longitude (relates to geography)
    - Population (relates to city type)
    """)
```

### Step 3: Create `main.py`

```python
"""
Lab 12: Main Program
Demonstrates K-Nearest Neighbors from Chapter 12.
"""
from distance import demonstrate_distance_metrics
from knn import (
    KNN,
    movie_recommendation_example,
    city_classification_example,
    picking_good_features
)


def main():
    # =========================================
    # PART 1: Distance Metrics
    # =========================================
    print("=" * 60)
    print("PART 1: DISTANCE METRICS")
    print("=" * 60)
    
    demonstrate_distance_metrics()
    
    # =========================================
    # PART 2: Classification
    # =========================================
    print("\n" + "=" * 60)
    print("PART 2: KNN CLASSIFICATION")
    print("=" * 60)
    
    city_classification_example()
    
    # =========================================
    # PART 3: Regression (Recommendations)
    # =========================================
    print("\n" + "=" * 60)
    print("PART 3: KNN REGRESSION (RECOMMENDATIONS)")
    print("=" * 60)
    
    movie_recommendation_example()
    
    # =========================================
    # PART 4: Feature Selection
    # =========================================
    picking_good_features()
    
    # =========================================
    # PART 5: Machine Learning Overview
    # =========================================
    print("\n" + "=" * 60)
    print("PART 5: INTRODUCTION TO MACHINE LEARNING")
    print("=" * 60)
    print("""
    From Chapter 12: KNN is your introduction to machine learning!
    
    MACHINE LEARNING APPLICATIONS:
    
    1. OCR (Optical Character Recognition)
       - Extract features from images (lines, curves, points)
       - Find nearest neighbors to classify characters
       - Used by Google to digitize books
    
    2. SPAM FILTERS (Naive Bayes)
       - Train on known spam/not-spam emails
       - Calculate probability based on words
       - "Million dollars" → probably spam!
    
    3. RECOMMENDATIONS (Netflix, Amazon)
       - Extract user preferences as features
       - Find similar users
       - Recommend what similar users liked
    
    ML TRAINING STEPS (from Chapter 12):
    
    1. Gather data
    2. Clean the data (remove bad data)
    3. Extract features
    4. Train the model (90% of data)
    5. Validate/test (10% of data)
    6. Tune parameters (like k in KNN)
    
    WHAT ML CAN'T DO WELL:
    - Predict stock market (too many variables)
    - Anything without good training data
    - Tasks requiring true understanding
    """)
    
    # =========================================
    # PART 6: Key Concepts Summary
    # =========================================
    print("\n" + "=" * 60)
    print("PART 6: KEY CONCEPTS FROM CHAPTER 12")
    print("=" * 60)
    print("""
    KNN SUMMARY:
    
    ┌─────────────────┬─────────────────────────────────┐
    │ Classification  │ Categorize into groups          │
    │                 │ → Majority vote of k neighbors  │
    ├─────────────────┼─────────────────────────────────┤
    │ Regression      │ Predict a number                │
    │                 │ → Average of k neighbors        │
    ├─────────────────┼─────────────────────────────────┤
    │ Feature         │ Convert items to numbers        │
    │ Extraction      │ → Choose features carefully!    │
    ├─────────────────┼─────────────────────────────────┤
    │ Distance        │ Euclidean (straight line)       │
    │                 │ Cosine (angle between vectors)  │
    └─────────────────┴─────────────────────────────────┘
    
    CHOOSING K:
    - Small k → sensitive to noise
    - Large k → too general
    - Try different values and test!
    
    FEATURE SELECTION:
    - Features should relate to prediction
    - Normalize if scales differ
    - More features isn't always better
    """)


if __name__ == "__main__":
    main()
```

---

## 5. Lab Report Template

```markdown
# Lab 12: K-Nearest Neighbors

## Student Information
- **Name:** [Your Name]
- **Date:** [Date]

## KNN Concepts

### Classification vs Regression
| Type | Purpose | How it works |
|------|---------|--------------|
| Classification | | |
| Regression | | |

### Distance Metrics
[Explain Euclidean distance and when to use cosine similarity]

## City Classification Results

### Training Data
[List the cities and their regions]

### Test Results
| City | Coordinates | Predicted Region | Correct? |
|------|-------------|------------------|----------|
| | | | |

## Movie Recommendation Results

### User Preferences
[List user preferences and ratings]

### Prediction for New User
- New user preferences: [list]
- K nearest neighbors: [list with distances]
- Predicted rating: [value]

## Feature Selection

### Good vs Bad Features
| Feature Type | Example | Why Good/Bad |
|--------------|---------|--------------|
| Good | | |
| Bad | | |

## Reflection Questions

1. Why is KNN called a "lazy learner"?

2. How does changing k affect predictions?

3. Why is feature selection important?

4. When would you use cosine similarity instead of Euclidean distance?

5. What are the limitations of KNN?

## Machine Learning Connection

### How does KNN relate to machine learning?
[Explain based on Chapter 12]

### Real-world applications of KNN
[List 3 applications from the chapter]
```

---

## 6. Submission
Save files in `lab12_knn/`, complete README, commit and push.

---

## Congratulations!

You have completed all 12 labs covering the algorithms in "Grokking Algorithms, Second Edition":

| Lab | Chapter | Topic |
|-----|---------|-------|
| 1 | 1 | Binary Search & Big O |
| 2 | 2 | Selection Sort |
| 3 | 3 | Recursion |
| 4 | 4 | Quicksort |
| 5 | 5 | Hash Tables |
| 6 | 6 | Breadth-First Search |
| 7 | 7 | Trees & Huffman Coding |
| 8 | 8 | Balanced Trees |
| 9 | 9 | Dijkstra's Algorithm |
| 10 | 10 | Greedy Algorithms |
| 11 | 11 | Dynamic Programming |
| 12 | 12 | K-Nearest Neighbors |

You now have a solid foundation in data structures and algorithms!
