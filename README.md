# K-Nearest Neighbors (KNN) Implementation

A simple implementation of the K-Nearest Neighbors algorithm for predicting House Price Index (HPI) based on Age and Loan amount.

## Features

- Dataset with Age, Loan amount, and HPI values
- Min-max normalization of features
- Euclidean distance calculation between unknown case and all data points
- Detailed step-by-step distance calculations
- Normalized data visualization

## How it works

The algorithm:
1. Takes a dataset of known cases with Age, Loan, and HPI values
2. Normalizes all numerical features using min-max scaling
3. Calculates the Euclidean distance between an unknown case and all data points
4. Can be used to predict the HPI of the unknown case using the nearest neighbors

## Usage

Run the script to see the detailed calculations and the final distance table:

```
python knn.py
```

The closest neighbors (smallest distance values) can be used to predict the HPI for the new entry.
