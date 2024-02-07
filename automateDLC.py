import deeplabcut
import argparse

def train_network(config_path):
    deeplabcut.train_network(config_path, displayiters=1000, saveiters=20000, maxiters=100000)

def evaluate_network(config_path):
    deeplabcut.evaluate_network(config_path, plotting=True)

def analyze_videos(config_path, video_path):
    deeplabcut.analyze_videos(config_path, [video_path], save_as_csv=True)

def plot_trajectories(config_path, video_path):
    deeplabcut.plot_trajectories(config_path, [video_path])    

def create_labeled_video(config_path, video_path):
    deeplabcut.create_labeled_video(config_path, [video_path])

def extract_outliers(config_path, video_path):
    deeplabcut.extract_outlier_frames(config_path, [video_path], outlieralgorithm='jump')

def full_pipeline(config_path, video_path):
    train_network(config_path)
    evaluate_network(config_path)
    analyze_videos(config_path, video_path)
    plot_trajectories(config_path, video_path)
    create_labeled_video(config_path, video_path)
    extract_outliers(config_path, video_path)

def main():
    parser = argparse.ArgumentParser(description="DeepLabCut helper script.")
    parser.add_argument('function', type=str, choices=['train_network', 'evaluate_network', 'analyze_videos', 'create_labeled_video', 'extract_outliers', 'full_pipeline'], help="Function to run")
    parser.add_argument('config_path', type=str, help="Path to the config file")
    parser.add_argument('video_path', type=str, nargs='?', default='', help="Path to the video file")

    args = parser.parse_args()

    if args.function == 'train_network':
        train_network(args.config_path)
    elif args.function == 'evaluate_network':
        evaluate_network(args.config_path)
    elif args.function in ['analyze_videos', 'plot_trajectories', 'create_labeled_video', 'extract_outliers']:
        if args.video_path == '':
            raise ValueError("video_path is required for {}".format(args.function))
        globals()[args.function](args.config_path, args.video_path)
    elif args.function == 'full_pipeline':
        if args.video_path == '':
            raise ValueError("video_path is required for full_pipeline")
        full_pipeline(args.config_path, args.video_path)

if __name__ == "__main__":
    main()