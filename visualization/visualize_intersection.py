import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import imageio

def load_data(scene='IntersectionMerge', data_root='../data'):
    """Load the driving scene data."""
    data_path = os.path.join(data_root, scene, 'data.npy')
    data = np.load(data_path, allow_pickle=False)
    return data

def visualize_frames(data, output_dir='frames'):
    """
    Create visualization plots for each frame.
    
    Args:
        data: numpy array of shape (T, N, 4) where each entry is [frame, x, y, yaw]
        output_dir: directory to save frame images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_frames, num_cars, _ = data.shape
    
    # Determine axis limits from all data
    all_x = data[:, :, 1].flatten()  # x coordinates
    all_y = data[:, :, 2].flatten()  # y coordinates
    
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    
    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding_x = x_range * 0.1
    # Increase Y-axis padding significantly to provide a larger viewing window
    padding_y = y_range * 2 # significantly increased from 0.1 to 0.5 for better viewing
    
    print(f"Creating {num_frames} frame visualizations...")
    
    for frame_idx in range(num_frames):
        frame_data = data[frame_idx]  # Shape: (N, 4)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot NPC cars (indices 1 onwards)
        npc_x = frame_data[1:, 1]  # x coordinates
        npc_y = frame_data[1:, 2]  # y coordinates
        npc_yaw = frame_data[1:, 3]  # yaw angles
        
        ax.scatter(npc_x, npc_y, c='blue', s=100, alpha=0.7, 
                  marker='o', edgecolors='darkblue', linewidths=1.5,
                  label='NPC Cars')
        
        # Plot main car (index 0)
        main_x = frame_data[0, 1]
        main_y = frame_data[0, 2]
        main_yaw = frame_data[0, 3]
        
        ax.scatter(main_x, main_y, c='red', s=200, alpha=0.9,
                  marker='*', edgecolors='darkred', linewidths=2,
                  label='Main Car', zorder=5)
        
        # Add arrow to show direction (optional, based on yaw)
        # For main car
        arrow_length = max(x_range, y_range) * 0.05
        dx = arrow_length * np.cos(main_yaw)
        dy = arrow_length * np.sin(main_yaw)
        ax.arrow(main_x, main_y, dx, dy, head_width=arrow_length*0.3,
                head_length=arrow_length*0.3, fc='darkred', ec='darkred', 
                linewidth=2, zorder=6)
        
        # Set axis limits
        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)
        ax.invert_yaxis()  # Reverse the Y-axis
        
        # Labels and title
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_title(f'Frame {frame_idx + 1}/{num_frames}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        # Removed equal aspect ratio to allow independent Y-axis scaling
        
        # Save frame
        frame_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.png')
        plt.tight_layout()
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        if (frame_idx + 1) % 20 == 0:
            print(f"  Processed {frame_idx + 1}/{num_frames} frames...")
    
    print(f"All frames saved to {output_dir}/")

def create_gif(frame_dir='frames', output_gif='intersection_merge.gif', fps=10):
    """
    Create a GIF from frame images.
    
    Args:
        frame_dir: directory containing frame images
        output_gif: output GIF filename
        fps: frames per second for the GIF
    """
    print(f"Creating GIF from frames in {frame_dir}...")
    
    # Get all frame files sorted
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.startswith('frame_') and f.endswith('.png')])
    
    if not frame_files:
        raise ValueError(f"No frame images found in {frame_dir}")
    
    # Read all images
    images = []
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        images.append(imageio.v2.imread(frame_path))
    
    # Create GIF - duration is in milliseconds (1000 / fps)
    duration_ms = 1000 / fps
    gif_path = output_gif
    imageio.mimsave(gif_path, images, duration=duration_ms, loop=0)
    print(f"GIF saved to {gif_path}")

if __name__ == '__main__':
    # Load data
    print("Loading data...")
    data = load_data(scene='IntersectionMerge')
    print(f"Data shape: {data.shape} (frames, cars, features)")
    
    # Create visualizations
    frame_dir = 'frames'
    visualize_frames(data, output_dir=frame_dir)
    
    # Create GIF
    gif_name = 'intersection_merge_visualization.gif'
    create_gif(frame_dir=frame_dir, output_gif=gif_name, fps=10)
    
    print("\nVisualization complete!")
    print(f"  - Frame images: {frame_dir}/")
    print(f"  - GIF: {gif_name}")
