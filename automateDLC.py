import deeplabcut
import sys
from config import *

def train_network(config_path):
    
    deeplabcut.train_network(config_path, displayiters=1000, saveiters=20000, maxiters=100000)

def evaluate_network(config_path):
    
    deeplabcut.evaluate_network(config_path, plotting=True)  

def analyze_videos(config_path):
   
    deeplabcut.analyze_videos(config_path, video_path, save_as_csv=True)

def create_labeled_video(config_path, video_path):
    
    deeplabcut.create_labeled_video(config_path, video_path)

def extract_outliers(config_path, video_path):
    
    deeplabcut.extract_outlier_frames(config_path, video_path, outlieralgorithm='uncertain')

def full_pipeline(config_path, video_path):

    train_network(config_path)
    evaluate_network(config_path)
    analyze_videos(config_path, video_path)
    create_labeled_video(config_path, video_path)
    extract_outliers(config_path, video_path)

def main():
    if len(sys.argv) < 2 or sys.argv[1].lower() != 'run_all':
        print("Usage: python script.py run_all")
        sys.exit(1)

    full_pipeline()


if __name__ == "__main__":
    main(config_path, video_path)