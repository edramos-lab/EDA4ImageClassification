#!/usr/bin/env python3
"""
Test script to demonstrate the updated Balancing Efficiency calculation
"""

import pandas as pd
import numpy as np

def calculate_balancing_efficiency_old(class_counts):
    """Old Balancing Efficiency calculation (simplified)"""
    total_samples = sum(class_counts)
    num_classes = len(class_counts)
    ideal_count = total_samples / num_classes
    max_deviation = max(abs(count - ideal_count) for count in class_counts)
    return (1 - (max_deviation / ideal_count)) * 100

def calculate_balancing_efficiency_new(class_counts):
    """New Balancing Efficiency calculation (correct formula)"""
    total_samples = sum(class_counts)
    num_classes = len(class_counts)
    ideal_count = total_samples / num_classes
    # BE = (1 - (1/C) * Σ|n_i - n̄|/n̄) × 100
    sum_abs_deviations = sum(abs(count - ideal_count) for count in class_counts)
    return (1 - (1 / num_classes) * (sum_abs_deviations / ideal_count)) * 100

def main():
    # Load the sample dataset
    print("Loading dataset...")
    df = pd.read_csv('dataset_paths.csv')
    
    # Get class distribution
    class_counts = df['label'].value_counts()
    print(f"\nDataset Overview:")
    print(f"Total samples: {len(df)}")
    print(f"Number of classes: {len(class_counts)}")
    print(f"\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    # Calculate both old and new Balancing Efficiency
    counts_list = class_counts.tolist()
    
    old_be = calculate_balancing_efficiency_old(counts_list)
    new_be = calculate_balancing_efficiency_new(counts_list)
    
    print(f"\nBalancing Efficiency Comparison:")
    print(f"Old formula: {old_be:.2f}%")
    print(f"New formula: {new_be:.2f}%")
    print(f"Difference: {new_be - old_be:.2f} percentage points")
    
    # Show the calculation details for the new formula
    total_samples = sum(counts_list)
    num_classes = len(counts_list)
    ideal_count = total_samples / num_classes
    
    print(f"\nNew Formula Details:")
    print(f"Ideal count per class: {ideal_count:.2f}")
    print(f"Deviations from ideal:")
    sum_abs_deviations = 0
    for i, count in enumerate(counts_list):
        deviation = abs(count - ideal_count)
        sum_abs_deviations += deviation
        print(f"  Class {i+1}: {count} samples, deviation: {deviation:.2f}")
    
    print(f"Sum of absolute deviations: {sum_abs_deviations:.2f}")
    print(f"Average deviation: {sum_abs_deviations / num_classes:.2f}")
    print(f"Normalized average deviation: {(sum_abs_deviations / num_classes) / ideal_count:.4f}")
    print(f"BE = (1 - {1/num_classes:.4f} × {sum_abs_deviations/ideal_count:.4f}) × 100 = {new_be:.2f}%")

if __name__ == "__main__":
    main()