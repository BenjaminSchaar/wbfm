#!/usr/bin/env python3
"""
Debug script to test CSV loading directly
"""

import pandas as pd
import sys
from pathlib import Path

def debug_csv_file(csv_path):
    """Debug a specific CSV file"""
    print(f"Debugging CSV file: {csv_path}")
    
    # Check if file exists
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"ERROR: File does not exist: {csv_path}")
        return
    
    print(f"✓ File exists: {csv_file}")
    print(f"File size: {csv_file.stat().st_size} bytes")
    
    # Try to read raw content
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            first_lines = [f.readline().strip() for _ in range(5)]
        print(f"First 5 lines (raw):")
        for i, line in enumerate(first_lines):
            print(f"  {i}: {repr(line)}")
    except Exception as e:
        print(f"ERROR reading raw file: {e}")
        return
    
    # Try to read with pandas
    try:
        df = pd.read_csv(csv_file)
        print(f"✓ Pandas loaded successfully")
        print(f"Shape: {df.shape}")
        print(f"Columns (raw): {repr(df.columns.tolist())}")
        print(f"Dtypes: {df.dtypes.to_dict()}")
        print(f"First 3 rows:\n{df.head(3)}")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        print(f"Columns (cleaned): {df.columns.tolist()}")
        
        # Check if format is correct
        if list(df.columns) == ['frame', 'value']:
            print("✓ Column names are correct")
        else:
            print(f"✗ Column names incorrect. Expected: ['frame', 'value'], got: {list(df.columns)}")
        
        # Check data types
        print(f"Frame column numeric: {pd.api.types.is_numeric_dtype(df['frame'])}")
        print(f"Value column numeric: {pd.api.types.is_numeric_dtype(df['value'])}")
        
        return df
        
    except Exception as e:
        print(f"ERROR loading with pandas: {e}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_csv.py /path/to/your/file.csv")
        print("Example: python debug_csv.py /path/to/project/behavior/custom_timeseries/worm1-2025-07-17_conc_at_0.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    df = debug_csv_file(csv_path)
    
    if df is not None:
        print("\n" + "="*50)
        print("CSV file appears to be valid!")
        print("If it's still not loading in the dashboard, run the dashboard with debug output")
        print("="*50)