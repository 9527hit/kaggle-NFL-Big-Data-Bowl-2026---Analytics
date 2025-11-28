import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob

class NFLDataLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "nfl-big-data-bowl-2026-analytics" / "114239_nfl_competition_files_published_analytics_final" / "train"
        self.supp_file = self.data_dir / "nfl-big-data-bowl-2026-analytics" / "114239_nfl_competition_files_published_analytics_final" / "supplementary_data.csv"

    def load_supplementary_data(self):
        """Load game and play information."""
        if not self.supp_file.exists():
            raise FileNotFoundError(f"Supplementary data not found at {self.supp_file}")
        return pd.read_csv(self.supp_file)

    def load_week_data(self, week, load_output=False):
        """
        Load tracking data for a specific week.
        week: int (1-18)
        load_output: bool, whether to load and merge output data (ground truth)
        """
        week_str = f"{week:02d}"
        input_file = self.train_dir / f"input_2023_w{week_str}.csv"
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file for week {week} not found at {input_file}")
            
        print(f"Loading input data for Week {week}...")
        df_input = pd.read_csv(input_file)
        
        if load_output:
            output_file = self.train_dir / f"output_2023_w{week_str}.csv"
            if output_file.exists():
                print(f"Loading output data for Week {week}...")
                df_output = pd.read_csv(output_file)
                return self.merge_input_output(df_input, df_output)
            else:
                print(f"Output file for week {week} not found.")
                return df_input
        
        return df_input

    def merge_input_output(self, df_input, df_output):
        """
        Merge input and output dataframes to create full trajectories.
        Output frames are appended to Input frames.
        """
        # Get max frame_id from input for each play to offset output frames
        # Note: Input frame_ids might differ per play, but usually they are sequential 1..N
        # We need to be careful. Let's assume for a (game_id, play_id), input ends at N, output starts at N+1?
        # Based on check: Input ended at 26, Output started at 1. And coordinates matched.
        # So Output Frame 1 is actually Input Frame N + 1.
        
        # We need to find the max frame_id per play in input
        max_frames = df_input.groupby(['game_id', 'play_id'])['frame_id'].max().reset_index()
        max_frames.rename(columns={'frame_id': 'max_input_frame'}, inplace=True)
        
        # Merge max_frames into df_output
        df_output_merged = df_output.merge(max_frames, on=['game_id', 'play_id'], how='left')
        
        # Adjust frame_id in output
        df_output_merged['frame_id'] = df_output_merged['frame_id'] + df_output_merged['max_input_frame']
        
        # Drop the helper column
        df_output_merged.drop(columns=['max_input_frame'], inplace=True)
        
        # Now concatenate. 
        # Output has fewer columns. We should keep all columns from Input.
        # For columns missing in Output, we might forward fill? Or leave as NaN?
        # Static columns (player_name, height, etc.) should be filled.
        
        # Get static columns from input (first row per player)
        static_cols = ['game_id', 'play_id', 'nfl_id', 'player_name', 'player_role', 'player_position', 
                       'player_height', 'player_weight', 'ball_land_x', 'ball_land_y', 'play_direction']
        # Note: Not all columns might be in df_input, check intersection
        available_static = [c for c in static_cols if c in df_input.columns]
        
        player_meta = df_input[available_static].drop_duplicates()
        
        # Merge static info into output
        df_output_full = df_output_merged.merge(player_meta, on=['game_id', 'play_id', 'nfl_id'], how='left')
        
        # Concatenate
        # Ensure columns match
        common_cols = list(set(df_input.columns) & set(df_output_full.columns))
        
        df_full = pd.concat([df_input, df_output_full], ignore_index=True)
        
        # Sort
        df_full.sort_values(by=['game_id', 'play_id', 'nfl_id', 'frame_id'], inplace=True)
        
        # Calculate missing speed/acceleration for output frames
        # We group by player and play to ensure continuity
        # Note: This might be slow for large datasets, but necessary
        
        # Define a function to calculate kinematics
        def calculate_kinematics(group):
            # If 's' is missing (NaN), calculate it
            if group['s'].isnull().any():
                # Calculate dx, dy
                group['dx'] = group['x'].diff()
                group['dy'] = group['y'].diff()
                
                # Calculate speed (yards/0.1s -> yards/s, so multiply by 10)
                # Assuming 10Hz data
                calculated_s = np.sqrt(group['dx']**2 + group['dy']**2) * 10
                
                # Fill NaN 's' with calculated 's'
                group['s'] = group['s'].fillna(calculated_s)
                
                # Clean up
                group.drop(columns=['dx', 'dy'], inplace=True)
            return group

        # Apply to the whole dataframe
        # To optimize, we can only apply to rows where 's' is NaN? 
        # But we need the previous row for diff.
        # So we apply to groups that have at least one NaN in 's'
        
        # For efficiency, let's just fill NaNs using shift on the whole dataframe 
        # (assuming sorted by game, play, nfl_id, frame)
        # We need to be careful not to diff across players/plays.
        # Mask for valid transitions: same game, play, nfl_id
        
        # Vectorized approach
        mask = (df_full['game_id'] == df_full['game_id'].shift(1)) & \
               (df_full['play_id'] == df_full['play_id'].shift(1)) & \
               (df_full['nfl_id'] == df_full['nfl_id'].shift(1))
               
        dx = df_full['x'].diff()
        dy = df_full['y'].diff()
        speed = np.sqrt(dx**2 + dy**2) * 10
        
        # Only fill where mask is True (valid previous frame) and s is NaN
        df_full.loc[mask & df_full['s'].isnull(), 's'] = speed[mask & df_full['s'].isnull()]
        
        # For the very first frame of output (if it follows input), the diff works.
        # If there is a gap, it might be wrong, but we verified continuity.
        
        return df_full

    def standardize_tracking_data(self, df):
        """
        Standardize coordinates so all plays go from left to right.
        This is a standard BDB preprocessing step.
        """
        # Check if play_direction exists
        if 'play_direction' not in df.columns:
            return df

        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Identify plays going left
        to_left = df['play_direction'] == 'left'
        
        # Standardize X and Y
        # Field length is 120 (including endzones), width is 53.3
        df.loc[to_left, 'x'] = 120 - df.loc[to_left, 'x']
        df.loc[to_left, 'y'] = 53.3 - df.loc[to_left, 'y']
        
        # Standardize Ball Landing Spot if present
        if 'ball_land_x' in df.columns:
            df.loc[to_left, 'ball_land_x'] = 120 - df.loc[to_left, 'ball_land_x']
        if 'ball_land_y' in df.columns:
            df.loc[to_left, 'ball_land_y'] = 53.3 - df.loc[to_left, 'ball_land_y']
        
        # Standardize Direction (dir) and Orientation (o)
        # These are in degrees 0-360. 
        # To flip, we usually add 180 and mod 360.
        if 'dir' in df.columns:
            df.loc[to_left, 'dir'] = (df.loc[to_left, 'dir'] + 180) % 360
        if 'o' in df.columns:
            df.loc[to_left, 'o'] = (df.loc[to_left, 'o'] + 180) % 360
            
        return df

if __name__ == "__main__":
    # Example usage
    data_path = r"e:\AI\kaggle\nfl-big-data-bowl-2026-analytics\data"
    loader = NFLDataLoader(data_path)
    
    try:
        supp_df = loader.load_supplementary_data()
        print(f"Supplementary data loaded: {supp_df.shape}")
        
        week1_df = loader.load_week_data(1)
        print(f"Week 1 input data loaded: {week1_df.shape}")
        
        week1_df_std = loader.standardize_tracking_data(week1_df)
        print("Standardization complete.")
        
    except Exception as e:
        print(f"Error: {e}")
