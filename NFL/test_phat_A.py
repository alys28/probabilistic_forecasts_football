import pandas as pd
import os
from pathlib import Path


def process_game_csv_files(directory_path):
    """
    Process CSV files from a directory, keeping unique entries by timestep.
    
    Parameters:
    -----------
    directory_path : str
        Path to directory containing game CSV files
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with columns: timestep, homeWinProbability, game_id
    """
    all_games_data = []
    
    # Get all CSV files in the directory
    directory = Path(directory_path)
    csv_files = list(directory.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return pd.DataFrame()
    
    for csv_file in csv_files:
        try:
            # Extract game_id from filename (e.g., "game_401671489.csv" -> "401671489")
            filename = csv_file.stem  # Gets filename without extension
            if filename.startswith('game_'):
                game_id = int(filename.replace('game_', ''))
            else:
                game_id = filename
            
            # Read the CSV file
            df = pd.read_csv(csv_file)
            df = df.iloc[1:]  # Skip the first row if it's a header or metadata
            # Check if required columns exist
            if 'timestep' not in df.columns or 'homeWinProbability' not in df.columns:
                print(f"Skipping {csv_file.name}: missing required columns")
                continue
            
            # Keep only the last occurrence of each timestep
            df_unique = df.drop_duplicates(subset=['timestep'], keep='last')
            
            # Keep only required columns
            df_filtered = df_unique[['timestep', 'homeWinProbability']].copy()
            
            # Add game_id column
            df_filtered['game_id'] = game_id
            
            all_games_data.append(df_filtered)
           
        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")
            continue
    
    # Combine all dataframes
    if all_games_data:
        combined_df = pd.concat(all_games_data, ignore_index=True)
        print(f"\nTotal records: {len(combined_df)}")
        return combined_df
    else:
        print("No data processed")
        return pd.DataFrame()


def find_mismatches(df1, df2, tolerance=1e-9):
    """
    Find mismatches in homeWinProbability between two dataframes for the same game_id + timestep.
    
    Parameters:
    -----------
    df1 : pd.DataFrame
        First dataframe with columns: timestep, homeWinProbability, game_id
    df2 : pd.DataFrame
        Second dataframe with columns: timestep, homeWinProbability, game_id
    tolerance : float, optional
        Tolerance for floating point comparison (default: 1e-9)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing only mismatched entries with columns:
        - game_id
        - timestep
        - homeWinProbability_df1
        - homeWinProbability_df2
        - difference (absolute difference)
    """
    # Merge dataframes on game_id and timestep
    merged = pd.merge(
        df1,
        df2,
        on=['game_id', 'timestep'],
        how='inner',
        suffixes=('_df1', '_df2')
    )
    
    if merged.empty:
        return pd.DataFrame()
    
    # Calculate absolute difference
    merged['difference'] = abs(merged['homeWinProbability_df1'] - merged['homeWinProbability_df2'])
    
    # Find and return only mismatches (where difference exceeds tolerance)
    mismatches = merged[merged['difference'] > tolerance].copy()
    mismatches = mismatches.sort_values(['game_id', 'timestep']).reset_index(drop=True)
    
    return mismatches


# Example usage:
if __name__ == "__main__":
    # Example: process files from a specific directory
    directory = "/Users/aly/Documents/University_of_Waterloo/Winter 2025/Research/code/NFL/ML/dataset_interpolated_fixed/"
    years = [2024, 2025]
    # Process files for the specified years
    dfs = []
    for year in years:
        year_directory = os.path.join(directory, str(year))
        if os.path.exists(year_directory):
            print(f"Processing files in {year_directory}...")
            result_df = process_game_csv_files(year_directory)
            dfs.append(result_df)
            print(f"Processed {len(result_df)} records from {year_directory}")
        else:
                print(f"Directory {year_directory} does not exist.")
    # Combine into a single DataFrame if needed
    result_df = pd.concat(dfs, ignore_index=True)
    
    
    if not result_df.empty:
        print("\nFirst few rows:")
        print(result_df.head())
    
    results_file = "/Users/aly/Documents/University_of_Waterloo/Winter 2025/Research/code/NFL/test_8/coin_flip_model_combined_data.csv"
    if os.path.exists(results_file):
        result_df2 = pd.read_csv(results_file)
        result_df2 = result_df2.rename(columns={'phat_A': 'homeWinProbability', "game_completed": "timestep"})
        result_df2 = result_df2.drop(columns=['phat_B'])
        print("\nSecond DataFrame loaded successfully.")
        print(result_df2.head())
    
    if not result_df.empty and not result_df2.empty:
        mismatches = find_mismatches(result_df, result_df2)
        if not mismatches.empty:
            print("\nMismatches:")
            print(mismatches.head(10))
            print(f"\nTotal mismatches found: {len(mismatches)}")
        else:
            print("No mismatches found.")
