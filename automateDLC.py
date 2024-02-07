import deeplabcut
import argparse
from config import *


def train_network(config_path):
    deeplabcut.train_network(config_path, displayiters=1000, saveiters=20000, maxiters=100000)

def evaluate_network(config_path):
    deeplabcut.evaluate_network(config_path, plotting=True)

def analyze_videos(config_path, video_paths):
    deeplabcut.analyze_videos(config_path, video_paths, save_as_csv=True)

def plot_trajectories(config_path, video_paths):
    deeplabcut.plot_trajectories(config_path, video_paths)    

def create_labeled_video(config_path, video_paths):
    deeplabcut.create_labeled_video(config_path, video_paths)

def extract_outliers(config_path, video_paths):
    deeplabcut.extract_outlier_frames(config_path, video_paths, outlieralgorithm='jump', automatic = True)

def full_pipeline(config_path, video_paths):
    #train_network(config_path)
    evaluate_network(config_path)
    analyze_videos(config_path, video_paths)
    plot_trajectories(config_path, video_paths)
    create_labeled_video(config_path, video_paths)
    extract_outliers(config_path, video_paths)

def main():
    parser = argparse.ArgumentParser(description="DeepLabCut helper script.")
    parser.add_argument('function', type=str, choices=['train_network', 'evaluate_network', 'analyze_videos', 'plot_trajectories', 'create_labeled_video', 'extract_outliers', 'full_pipeline'], help="Function to run")

    args = parser.parse_args()

    if args.function == 'train_network':
        train_network(config_path)
    elif args.function == 'evaluate_network':
        evaluate_network(config_path)
    elif args.function in ['analyze_videos', 'plot_trajectories', 'create_labeled_video', 'extract_outliers', 'full_pipeline']:
        globals()[args.function](config_path, video_paths)


if __name__ == "__main__":
    main()