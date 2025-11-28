import pandas as pd
import numpy as np

def calculate_receiver_adjustment_metrics(df):
    """
    Calculate metrics related to receiver adjustment to the ball.
    Assumes df contains data for a SINGLE player on a SINGLE play,
    and includes both input and output frames (full trajectory).
    """
    df = df.copy()
    
    # Ensure sorted by frame
    df.sort_values('frame_id', inplace=True)
    
    # Get ball landing spot (assumed constant for the play)
    ball_x = df['ball_land_x'].iloc[0]
    ball_y = df['ball_land_y'].iloc[0]
    
    # Calculate distance to ball at each frame
    df['dist_to_ball'] = np.sqrt((df['x'] - ball_x)**2 + (df['y'] - ball_y)**2)
    
    # Calculate frames remaining until the end of the trajectory
    # We assume the last frame in the data is the catch/arrival point
    max_frame = df['frame_id'].max()
    df['frames_remaining'] = max_frame - df['frame_id']
    
    # Calculate time remaining (assuming 10Hz data, so 0.1s per frame)
    df['time_remaining'] = df['frames_remaining'] * 0.1
    
    # Calculate Required Speed (yards/sec)
    # Avoid division by zero at the last frame
    df['required_speed'] = np.where(
        df['time_remaining'] > 0,
        df['dist_to_ball'] / df['time_remaining'],
        0 # At the catch point, required speed is 0 (or undefined)
    )
    
    # Calculate Speed Difference (Current Speed - Required Speed)
    # Positive: Player has "speed buffer" (can slow down)
    # Negative: Player is "behind schedule" (needs to speed up)
    df['speed_diff'] = df['s'] - df['required_speed']
    
    # Calculate Acceleration Required (to reach required speed? or to reach target?)
    # This is more complex, maybe stick to speed diff for now.
    
    return df

def analyze_adjustment_efficiency(df_metrics):
    """
    Aggregates metrics to score the adjustment.
    """
    # Example metric: Average absolute speed difference in the last 1 second (10 frames)
    # A lower value means the player was perfectly in sync with the ball arrival.
    
    last_1_sec = df_metrics[df_metrics['time_remaining'] <= 1.0]
    
    if len(last_1_sec) == 0:
        return None
        
    avg_speed_diff = last_1_sec['speed_diff'].mean()
    abs_speed_diff = last_1_sec['speed_diff'].abs().mean()
    
    return {
        'avg_speed_diff_last_1s': avg_speed_diff,
        'abs_speed_diff_last_1s': abs_speed_diff,
        'final_dist_error': df_metrics['dist_to_ball'].iloc[-1]
    }
