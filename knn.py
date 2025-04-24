import pandas as pd
import numpy as np

# Step 1: Original data
data = [
    [25, 40000, 135],
    [35, 60000, 256],
    [45, 80000, 231],
    [20, 20000, 267],
    [35, 120000, 139],
    [52, 18000, 150],
    [23, 95000, 127],
    [40, 62000, 216],
    [60, 100000, 139],
    [48, 220000, 250],
    [33, 150000, 264]
]

unknown_case = [48, 142000]  # Age, Loan

# Create DataFrame
df = pd.DataFrame(data, columns=["Age", "Loan", "HPI"])

# Step 2: Min-max normalization
def normalize_column(col):
    return (col - col.min()) / (col.max() - col.min())

df["Age_norm"] = normalize_column(df["Age"])
df["Loan_norm"] = normalize_column(df["Loan"])
df["HPI_norm"] = normalize_column(df["HPI"])

# Normalize unknown case
age_min, age_max = df["Age"].min(), df["Age"].max()
loan_min, loan_max = df["Loan"].min(), df["Loan"].max()

unknown_age_norm = (unknown_case[0] - age_min) / (age_max - age_min)
unknown_loan_norm = (unknown_case[1] - loan_min) / (loan_max - loan_min)

print("\nUnknown case normalization:")
print(f"Age norm: ({unknown_case[0]} - {age_min}) / ({age_max} - {age_min}) = {unknown_age_norm:.4f}")
print(f"Loan norm: ({unknown_case[1]} - {loan_min}) / ({loan_max} - {loan_min}) = {unknown_loan_norm:.4f}")

# Step 3: Compute Euclidean distance using normalized Age and Loan
print("\nDistance Calculation Details:")
for i, row in df.iterrows():
    distance = np.sqrt(
        (row["Age_norm"] - unknown_age_norm) ** 2 +
        (row["Loan_norm"] - unknown_loan_norm) ** 2
    )
    
    df.at[i, "Distance"] = distance
    
    print(f"\nRow {i+1} ({row['Age']}, {row['Loan']}, {row['HPI']}):")
    print(f"  Age_norm: {row['Age_norm']:.4f}, Unknown_age_norm: {unknown_age_norm:.4f}")
    print(f"  Loan_norm: {row['Loan_norm']:.4f}, Unknown_loan_norm: {unknown_loan_norm:.4f}")
    print(f"  Distance = √[({row['Age_norm']:.4f} - {unknown_age_norm:.4f})² + ({row['Loan_norm']:.4f} - {unknown_loan_norm:.4f})²]")
    print(f"           = √[{(row['Age_norm'] - unknown_age_norm)**2:.4f} + {(row['Loan_norm'] - unknown_loan_norm)**2:.4f}]")
    print(f"           = √{(row['Age_norm'] - unknown_age_norm)**2 + (row['Loan_norm'] - unknown_loan_norm)**2:.4f}")
    print(f"           = {distance:.4f}")

# Step 4: Display table with normalized data and distance
pd.set_option("display.precision", 4)
print("\nFinal Table:")
print(df[["Age", "Loan", "HPI", "Age_norm", "Loan_norm", "HPI_norm", "Distance"]])
