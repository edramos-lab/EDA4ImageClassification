#!/usr/bin/env python3
"""
Simple demonstration of the updated Balancing Efficiency calculation
"""

def old_be(class_counts):
    """Old formula: only considers max deviation"""
    total = sum(class_counts)
    num_classes = len(class_counts)
    ideal = total / num_classes
    max_dev = max(abs(count - ideal) for count in class_counts)
    return (1 - (max_dev / ideal)) * 100

def new_be(class_counts):
    """New formula: considers all deviations with proper weighting"""
    total = sum(class_counts)
    num_classes = len(class_counts)
    ideal = total / num_classes
    sum_abs_dev = sum(abs(count - ideal) for count in class_counts)
    return (1 - (1 / num_classes) * (sum_abs_dev / ideal)) * 100

# Example with imbalanced dataset
example_counts = [100, 50, 200, 25]  # Very imbalanced
print("Example: Imbalanced dataset with counts [100, 50, 200, 25]")
print(f"Old BE: {old_be(example_counts):.2f}%")
print(f"New BE: {new_be(example_counts):.2f}%")
print()

# Example with balanced dataset
balanced_counts = [100, 100, 100, 100]  # Perfectly balanced
print("Example: Balanced dataset with counts [100, 100, 100, 100]")
print(f"Old BE: {old_be(balanced_counts):.2f}%")
print(f"New BE: {new_be(balanced_counts):.2f}%")
print()

# Example with slightly imbalanced dataset
slight_imbalance = [95, 105, 98, 102]  # Slightly imbalanced
print("Example: Slightly imbalanced dataset with counts [95, 105, 98, 102]")
print(f"Old BE: {old_be(slight_imbalance):.2f}%")
print(f"New BE: {new_be(slight_imbalance):.2f}%")