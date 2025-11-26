import pandas as pd
import sys
import os

def split_csv(input_file, output_file1='part1.csv', output_file2='part2.csv'):
    """
    Split a CSV file into two parts (75/25).
    
    Args:
        input_file: Path to the input CSV file
        output_file1: Path for the first output CSV (75% of data, default: part1.csv)
        output_file2: Path for the second output CSV (25% of data, default: part2.csv)
    """
    try:
        # Read the CSV file
        print(f"Reading {input_file}...")
        df = pd.read_csv(input_file)
        
        total_rows = len(df)
        print(f"Total rows: {total_rows}")
        
        # Calculate split point (75% for first file)
        split_point = int(total_rows * 0.97)
        
        # Split the dataframe
        df1 = df.iloc[:split_point]
        df2 = df.iloc[split_point:]
        
        # Save to new CSV files
        df1.to_csv(output_file1, index=False)
        df2.to_csv(output_file2, index=False)
        
        print(f"\nSplit complete!")
        print(f"{output_file1}: {len(df1)} rows ({len(df1)/total_rows*100:.1f}%)")
        print(f"{output_file2}: {len(df2)} rows ({len(df2)/total_rows*100:.1f}%)")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file.csv> [output1.csv] [output2.csv]")
        print("Example: python script.py data.csv part1.csv part2.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file1 = sys.argv[2] if len(sys.argv) > 2 else 'model_dataset.csv'
    output_file2 = sys.argv[3] if len(sys.argv) > 3 else 'script_dataset.csv'

    split_csv(input_file, output_file1, output_file2)