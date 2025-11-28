import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys
import gc

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data_loader import NFLDataLoader
from src.metrics import calculate_receiver_adjustment_metrics, analyze_adjustment_efficiency

def process_weeks(weeks, data_dir, output_file):
    loader = NFLDataLoader(data_dir)
    output_path = Path(output_file)
    
    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting batch analysis for weeks: {list(weeks)}")
    print(f"Results will be saved to: {output_path}")

    for week in weeks:
        print(f"\nProcessing Week {week}...")
        try:
            # Load data
            df = loader.load_week_data(week, load_output=True)
            if df is None:
                print(f"Skipping Week {week} (No data found)")
                continue
                
            # Standardize
            df = loader.standardize_tracking_data(df)
            
            # Identify targets
            targets = df[df['player_to_predict'] == True][['game_id', 'play_id', 'nfl_id']].drop_duplicates()
            print(f"  Found {len(targets)} targets.")
            
            week_results = []
            
            # Process each target
            for _, row in tqdm(targets.iterrows(), total=len(targets), desc=f"Week {week}"):
                game_id = row['game_id']
                play_id = row['play_id']
                nfl_id = row['nfl_id']
                
                # Extract track
                player_track = df[(df['game_id'] == game_id) & 
                                  (df['play_id'] == play_id) & 
                                  (df['nfl_id'] == nfl_id)].copy()
                
                # Basic validation
                if len(player_track) < 5:
                    continue
                
                # Check for ball landing spot
                if pd.isna(player_track['ball_land_x'].iloc[0]):
                    continue
                    
                try:
                    metrics_df = calculate_receiver_adjustment_metrics(player_track)
                    summary = analyze_adjustment_efficiency(metrics_df)
                    
                    if summary:
                        summary['game_id'] = game_id
                        summary['play_id'] = play_id
                        summary['nfl_id'] = nfl_id
                        summary['week'] = week
                        week_results.append(summary)
                except Exception:
                    continue
            
            # Save results for this week
            if week_results:
                df_results = pd.DataFrame(week_results)
                
                # Check if file exists to determine if we need header
                header = not output_path.exists()
                
                df_results.to_csv(output_path, mode='a', header=header, index=False)
                print(f"  Saved {len(df_results)} records to {output_path}")
            
            # Clean up memory
            del df
            del targets
            del week_results
            gc.collect()
                
        except Exception as e:
            print(f"Error processing Week {week}: {e}")

if __name__ == "__main__":
    # Configuration
    DATA_PATH = r"e:\AI\kaggle\nfl-big-data-bowl-2026-analytics\data"
    OUTPUT_FILE = r"e:\AI\kaggle\nfl-big-data-bowl-2026-analytics\data\adjustment_metrics_all.csv"
    
    # Process Weeks 1-9
    WEEKS_TO_PROCESS = range(1, 10) 
    
    process_weeks(WEEKS_TO_PROCESS, DATA_PATH, OUTPUT_FILE)
