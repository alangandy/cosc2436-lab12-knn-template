"""Lab 12: Test Cases for KNN"""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knn import euclidean_distance, knn_classify, knn_regression, normalize_features


class TestEuclideanDistance:
    def test_basic(self):
        assert euclidean_distance([0, 0], [3, 4]) == 5.0
    
    def test_same_point(self):
        assert euclidean_distance([1, 2, 3], [1, 2, 3]) == 0.0
    
    def test_1d(self):
        assert euclidean_distance([0], [5]) == 5.0


class TestKNNClassify:
    def test_basic(self):
        data = [
            ([1, 1], "A"), ([2, 2], "A"), ([1.5, 1.5], "A"),
            ([8, 8], "B"), ([9, 9], "B"), ([8.5, 8.5], "B")
        ]
        assert knn_classify(data, [1.2, 1.2], k=3) == "A"
        assert knn_classify(data, [8.2, 8.2], k=3) == "B"
    
    def test_tie_breaker(self):
        data = [([0, 0], "A"), ([1, 1], "B")]
        # With k=1, should pick closest
        result = knn_classify(data, [0.1, 0.1], k=1)
        assert result == "A"


class TestKNNRegression:
    def test_basic(self):
        data = [
            ([1], 10.0), ([2], 20.0), ([3], 30.0)
        ]
        result = knn_regression(data, [2], k=3)
        assert result == 20.0  # Average of 10, 20, 30
    
    def test_k1(self):
        data = [([0], 100.0), ([10], 200.0)]
        result = knn_regression(data, [1], k=1)
        assert result == 100.0


class TestNormalize:
    def test_basic(self):
        data = [[0, 0], [5, 10], [10, 20]]
        result = normalize_features(data)
        assert result[0] == [0.0, 0.0]
        assert result[2] == [1.0, 1.0]
    
    def test_empty(self):
        assert normalize_features([]) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
