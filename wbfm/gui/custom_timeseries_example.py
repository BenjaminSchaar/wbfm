#!/usr/bin/env python3
"""
Example script showing how to create and test custom timeseries functionality.

This script creates example CSV files in the correct format and provides
a test to verify that the custom timeseries loading works correctly.
"""

import os
import pandas as pd
from pathlib import Path
import numpy as np

def create_example_custom_timeseries(project_folder: str):
    """
    Create example custom timeseries CSV files for testing.
    
    Parameters
    ----------
    project_folder : str
        Path to the project folder
    """
    
    # Create the custom_timeseries folder
    custom_folder = Path(project_folder) / "behavior" / "custom_timeseries"
    custom_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating example custom timeseries in: {custom_folder}")
    
    # Example 1: Sine wave timeseries
    frames1 = np.arange(0, 1000)
    values1 = np.sin(frames1 * 0.1) + np.random.normal(0, 0.1, len(frames1))
    
    df1 = pd.DataFrame({
        'frame': frames1,
        'value': values1
    })
    
    csv_file1 = custom_folder / "sine_wave_signal.csv"
    df1.to_csv(csv_file1, index=False)
    print(f"Created: {csv_file1}")
    
    # Example 2: Step function timeseries  
    frames2 = np.arange(0, 1000)
    values2 = np.where(frames2 < 300, 1.0, 
                      np.where(frames2 < 600, 2.5, 
                              np.where(frames2 < 900, 0.5, 3.0)))
    values2 += np.random.normal(0, 0.05, len(frames2))
    
    df2 = pd.DataFrame({
        'frame': frames2,
        'value': values2
    })
    
    csv_file2 = custom_folder / "step_function.csv"
    df2.to_csv(csv_file2, index=False)
    print(f"Created: {csv_file2}")
    
    # Example 3: Linear trend with noise
    frames3 = np.arange(0, 1000)
    values3 = 0.002 * frames3 + 1.0 + np.random.normal(0, 0.1, len(frames3))
    
    df3 = pd.DataFrame({
        'frame': frames3,
        'value': values3
    })
    
    csv_file3 = custom_folder / "linear_trend.csv"
    df3.to_csv(csv_file3, index=False)
    print(f"Created: {csv_file3}")
    
    print(f"\nCreated 3 example custom timeseries CSV files:")
    print(f"- sine_wave_signal.csv: Oscillating signal")
    print(f"- step_function.csv: Step changes in signal") 
    print(f"- linear_trend.csv: Linear trend with noise")
    print(f"\nThese will appear in the behavior dropdown as:")
    print(f"- sine_wave_signal")
    print(f"- step_function") 
    print(f"- linear_trend")

def test_custom_timeseries_loading():
    """
    Test function to verify custom timeseries loading works correctly.
    """
    print("\n" + "="*50)
    print("TESTING CUSTOM TIMESERIES FUNCTIONALITY")
    print("="*50)
    
    # Import the dashboard module
    try:
        from wbfm_dashboard import _load_custom_timeseries_csvs, _downsample_custom_timeseries
        print("✓ Successfully imported custom timeseries functions")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return
    
    # Create a temporary test folder
    test_folder = Path("/tmp/test_custom_timeseries")
    test_folder.mkdir(exist_ok=True)
    
    # Create test CSV files
    create_example_custom_timeseries(str(test_folder))
    
    # Test loading
    custom_path = test_folder / "behavior" / "custom_timeseries"
    df_custom = _load_custom_timeseries_csvs(custom_path)
    
    if df_custom.empty:
        print("✗ No custom timeseries loaded")
        return
    else:
        print(f"✓ Loaded {len(df_custom.columns)} custom timeseries: {list(df_custom.columns)}")
    
    # Test downsampling
    target_length = 500  # Simulate trace length
    df_downsampled = _downsample_custom_timeseries(df_custom, target_length)
    
    if len(df_downsampled) == target_length:
        print(f"✓ Successfully downsampled from {len(df_custom)} to {target_length} frames")
    else:
        print(f"✗ Downsampling failed: got {len(df_downsampled)} frames, expected {target_length}")
    
    print("\n" + "="*50)
    print("USAGE INSTRUCTIONS")
    print("="*50)
    print("1. Place your project at: /path/to/your/project/")
    print("2. Create folder: /path/to/your/project/behavior/custom_timeseries/")
    print("3. Add CSV files with format: frame,value")
    print("4. Run: python wbfm_dashboard.py --project_path /path/to/your/project/project_config.yaml")
    print("5. Your custom timeseries will appear in behavior dropdowns!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Create example files in specified project folder
        project_folder = sys.argv[1]
        create_example_custom_timeseries(project_folder)
    else:
        # Run test
        test_custom_timeseries_loading()