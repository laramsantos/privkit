import numpy as np
import osmnx as ox
import pandas as pd
import geopy.distance
import config
from haversine import haversine, Unit

def improved_adaptive_epsilon_formula_pred(predictability_value, min_epsilon=config.min_eps, max_epsilon=config.max_eps, beta=2):
    predictability_risk = predictability_value

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-2 * predictability_risk)

    return epsilon

def compute_proximity_between_points(trajectory):
    """
    Compute the proximity between points in a trajectory.

    Parameters:
    - trajectory: List of (latitude, longitude) tuples.

    Returns:
    - A list of distances between consecutive points in meters.
    """
    distances = []
    for i in range(len(trajectory) - 1):
        point1 = trajectory[i]
        point2 = trajectory[i + 1]
        distance = haversine(point1, point2, unit=Unit.METERS)
        distances.append(distance)
    return np.mean(distances)

def euclidian_distance(trajectory):
    """
    Compute the Euclidean distance between consecutive points in a
    trajectory.
    Parameters:
    - trajectory: List of (latitude, longitude) tuples.
    Returns:
    - A list of Euclidean distances between consecutive points.
    """
    distances = []
    for i in range(len(trajectory) - 1):
        point1 = trajectory[i]
        point2 = trajectory[i + 1]
        distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        distances.append(distance)
    return np.mean(distances)

def get_predictability_score_per_point(variance_x, variance_y):
    """
    Computes a predictability score between 0 and 1 per point 
    based on variance in x and y directions.

    Parameters:
    variance_x (ndarray): Variance along x-axis (shape: [batch_size, steps])
    variance_y (ndarray): Variance along y-axis (shape: [batch_size, steps])

    Returns:
    ndarray: Predictability scores (shape: [batch_size, steps])
    """
    # Step 1: Aggregate variance per point
    total_variance = variance_x + variance_y  # shape: (batch_size, steps)

    scale_factor = 1 / (np.median(total_variance) + 1e-8) ** 0.5
    predictability = 1 / (1 + scale_factor * total_variance)

    return predictability  # shape: (batch_size, steps)

def get_predictability_by_distance(predicted, actual):
    """
    Compute predictability based on distance between predicted and actual points.

    Parameters:
    - predicted: Predicted point
    - actual: Actual point
    """
    distance = haversine(predicted, actual, unit=Unit.METERS)
    predictability = 1 / (1 + distance) 
    return predictability