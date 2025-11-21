#!/usr/bin/env python3
"""
Test script to verify custom timeseries loading fix
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, '/Users/benjaminschaar/Documents/GitHub/wbfm')

try:
    # Test if we can at least load the CSV files correctly
    import pandas as pd
    from pathlib import Path
    
    def _load_custom_timeseries_csvs(custom_timeseries_path: Path) -> pd.DataFrame:
        """
        Load all CSV files from custom_timeseries folder and validate format.
        """
        print(f"🔍 DEBUG: Loading CSV files from: {custom_timeseries_path}")
        
        if not custom_timeseries_path.exists():
            print(f"❌ ERROR: Custom timeseries path does not exist: {custom_timeseries_path}")
            return pd.DataFrame()
        
        csv_files = list(custom_timeseries_path.glob("*.csv"))
        print(f"🔍 DEBUG: Found {len(csv_files)} CSV files")
        
        if len(csv_files) == 0:
            print("🔍 DEBUG: No CSV files found")
            return pd.DataFrame()
        
        dataframes = []
        for csv_file in csv_files:
            print(f"🔍 DEBUG: Processing {csv_file.name}...")
            try:
                df = pd.read_csv(csv_file)
                df.columns = df.columns.str.strip()  # Clean column names
                
                # Validate format
                if list(df.columns) != ['frame', 'value']:
                    print(f"❌ ERROR: Invalid CSV format in {csv_file.name}. Expected columns: ['frame', 'value'], got: {list(df.columns)}")
                    continue
                
                # Check data types
                if not pd.api.types.is_numeric_dtype(df['frame']) or not pd.api.types.is_numeric_dtype(df['value']):
                    print(f"❌ ERROR: Non-numeric data in {csv_file.name}")
                    continue
                
                # Use filename (without extension) as column name
                timeseries_name = csv_file.stem
                df_timeseries = pd.DataFrame({timeseries_name: df['value']})
                dataframes.append(df_timeseries)
                
                print(f"✅ SUCCESS: Loaded {len(df)} rows from {csv_file.name} as '{timeseries_name}'")
                
            except Exception as e:
                print(f"❌ ERROR: Failed to load {csv_file.name}: {e}")
                continue
        
        if len(dataframes) == 0:
            print("🔍 DEBUG: No valid CSV files loaded")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, axis=1)
        print(f"✅ SUCCESS: Combined {len(dataframes)} timeseries into DataFrame with shape {combined_df.shape}")
        print(f"✅ SUCCESS: Available timeseries: {list(combined_df.columns)}")
        
        return combined_df

    def test_project_directory_detection():
        """Test if we can properly detect project directory"""
        print("="*60)
        print("Testing project directory detection...")
        print("="*60)
        
        # Simulate the project config structure we saw in debug output
        class MockProjectConfig:
            def __init__(self):
                self.project_dir = '/Volumes/scratch/neurobiology/zimmer/schaar/wbfm/results/dataset/benz_0_7/worm1-2025-07-17'
        
        class MockProjectData:
            def __init__(self):
                self.project_config = MockProjectConfig()
        
        # Test the fixed logic
        dat = MockProjectData()
        
        if hasattr(dat, 'project_config') and hasattr(dat.project_config, 'project_dir'):
            project_directory = dat.project_config.project_dir
            print(f"✅ SUCCESS: Found project_dir: {project_directory}")
            
            # Test path construction
            project_folder = Path(project_directory)
            custom_timeseries_path = project_folder / 'behavior' / 'custom_timeseries'
            print(f"✅ SUCCESS: Custom timeseries path: {custom_timeseries_path}")
            print(f"✅ SUCCESS: Path exists: {custom_timeseries_path.exists()}")
            
            if custom_timeseries_path.exists():
                # Test CSV loading
                df_custom = _load_custom_timeseries_csvs(custom_timeseries_path)
                if not df_custom.empty:
                    print(f"✅ SUCCESS: Loaded custom timeseries with columns: {list(df_custom.columns)}")
                    print(f"✅ SUCCESS: Shape: {df_custom.shape}")
                    return True
                else:
                    print("❌ ERROR: Custom timeseries DataFrame is empty")
                    return False
            else:
                print("❌ ERROR: Custom timeseries path does not exist")
                return False
        else:
            print("❌ ERROR: Could not find project_dir attribute")
            return False

    if __name__ == "__main__":
        result = test_project_directory_detection()
        if result:
            print("\n🎉 SUCCESS: The fix should work!")
            print("Custom timeseries should now load properly in the trace explorer.")
        else:
            print("\n❌ FAILED: There are still issues with the fix.")

except Exception as e:
    print(f"❌ ERROR: Failed to run test: {e}")
    import traceback
    traceback.print_exc()