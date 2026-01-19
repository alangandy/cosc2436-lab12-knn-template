"""
Lab 12: K-Nearest Neighbors
Implement KNN from Chapter 12.

Chapter 12 covers:
- Classification and regression
- Feature extraction
- Distance metrics (Euclidean, cosine)
- Choosing K
"""
from typing import List, Tuple, Dict
import math


def euclidean_distance(p1: List[float], p2: List[float]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    From Chapter 12: sqrt(sum((a-b)² for each dimension))
    
    Example:
        >>> euclidean_distance([0, 0], [3, 4])
        5.0
    """
    # TODO: Implement Euclidean distance
    pass


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
    
    Example:
        >>> data = [([1, 1], "A"), ([2, 2], "A"), ([8, 8], "B"), ([9, 9], "B")]
        >>> knn_classify(data, [1.5, 1.5], k=3)
        'A'
    """
    # TODO: Implement KNN classification
    # 1. Calculate distance from point to all data points
    # 2. Sort by distance
    # 3. Take k nearest neighbors
    # 4. Vote on class (most common label wins)
    
    pass


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
    # TODO: Implement KNN regression
    pass


def normalize_features(data: List[List[float]]) -> List[List[float]]:
    """
    Normalize features to 0-1 range.
    
    From Chapter 12: Important when features have different scales.
    """
    # TODO: Implement feature normalization
    pass
