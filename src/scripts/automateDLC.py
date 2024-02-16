import deeplabcut
import argparse
from config import *
import time 
import csv
from src.utils.extrafun import read_config 

def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:.2f}"

def train_network(config_path):
    deeplabcut.train_network(config_path, displayiters=1000, saveiters=20000, maxiters=max_iters)

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

def full_pipeline(config_path, video_paths, csv_path):
    tic_train = time.time()
    train_network(config_path)
    toc_train = time.time()
    tic_eval = time.time()
    evaluate_network(config_path)
    toc_eval = time.time()
    tic_analyze = time.time()
    analyze_videos(config_path, video_paths)
    toc_analyze = time.time()
    tic_plot = time.time()
    plot_trajectories(config_path, video_paths)
    toc_plot = time.time()
    tic_create = time.time()
    create_labeled_video(config_path, video_paths)
    toc_create = time.time()
    tic_extract = time.time()
    extract_outliers(config_path, video_paths)
    toc_extract = time.time()

    time_train = toc_train - tic_train
    time_eval = toc_eval - tic_eval
    time_analyze = toc_analyze - tic_analyze
    time_plot = toc_plot - tic_plot
    time_create = toc_create - tic_create
    time_extract = toc_extract - tic_extract

    total_time = time_train + time_eval + time_analyze + time_plot + time_create + time_extract 

    tasks_and_times = {
    "Train Network": {"duration_seconds": time_train, "formatted": format_duration(time_train)},
    "Evaluate Network": {"duration_seconds": time_eval, "formatted": format_duration(time_eval)},
    "Analyze Videos": {"duration_seconds": time_analyze, "formatted": format_duration(time_analyze)},
    "Plot Trajectories": {"duration_seconds": time_plot, "formatted": format_duration(time_plot)},
    "Create Labeled Video": {"duration_seconds": time_create, "formatted": format_duration(time_create)},
    "Extract Outliers": {"duration_seconds": time_extract, "formatted": format_duration(time_extract)},
    "Total": {"duration_seconds": total_time, "formatted": format_duration(total_time)},
}
    cfg = read_config(config_path)
    iteration = "iteration-" + str(cfg["iteration"])

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'Maximum Iterations', 'Task', 'Duration (seconds)', 'Duration (hh:mm:ss)'])
        for task, times in tasks_and_times.items():
            writer.writerow([iteration, max_iters, task, times['duration_seconds'], times['formatted']])
 
def main():
    parser = argparse.ArgumentParser(description="DeepLabCut helper script.")
    parser.add_argument('function', type=str, choices=['train_network', 'evaluate_network', 'analyze_videos', 'plot_trajectories', 'create_labeled_video', 'extract_outliers', 'full_pipeline'], help="Function to run")

    args = parser.parse_args()

    if args.function == 'train_network':
        train_network(config_path)
    elif args.function == 'evaluate_network':
        evaluate_network(config_path)
    elif args.function in ['analyze_videos', 'plot_trajectories', 'create_labeled_video', 'extract_outliers', 'full_pipeline']:
        globals()[args.function](config_path, video_paths, csv_path)

if __name__ == "__main__":
    main()
