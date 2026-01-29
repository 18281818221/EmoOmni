import os
import subprocess
import json
import time
import uuid
import requests
import base64
import os
import json
import librosa
from typing import List, Dict
import soundfile as sf
import base64
import io
import numpy as np
import librosa
import soundfile as sf
from functools import partial
from tqdm import tqdm
import multiprocessing as mp

SUPPORTED_VIDEO_EXTENSIONS = ('.mp4')


# 填入控制台获取的app id和access token
appid = "4057532394"

token = "rOfXzHybZLbDUXR_Ey3Q8qTo9BALdeQ_"

appid = "4057532394"


def find_video_files(root_dir):
    """递归查找目录中的所有视频文件并排序"""
    video_paths = []
    
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"目录不存在: {root_dir}")
    
    # 递归遍历目录
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(SUPPORTED_VIDEO_EXTENSIONS):
                video_path = os.path.join(dirpath, filename)
                if 'wav_split' not in video_path:
                    video_paths.append(video_path)
    
    # 按路径排序
    video_paths.sort()
    return video_paths


def split_video(input_path, output_dir, chunk_minutes=20, movie_name=None):
    """
    将视频按固定时长切分成多个小片段
    
    参数：
        input_path: 原始视频文件路径 (如 movie.mp4)
        output_dir: 输出目录 (不存在则自动创建)
        chunk_minutes: 每段视频长度（分钟），默认20分钟
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"视频文件不存在: {input_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    chunk_seconds = chunk_minutes * 60

    # 获取视频总时长（秒）
    cmd = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of",
        "default=noprint_wrappers=1:nokey=1", input_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    total_duration = float(result.stdout.strip())
    print(f"🎞️ 视频总时长: {total_duration/60:.2f} 分钟")

    # 按20分钟切分
    num_chunks = int(total_duration // chunk_seconds) + 1
    for i in range(num_chunks):
        start_time = i * chunk_seconds
        output_file = os.path.join(output_dir, f"{movie_name}____{i+1:03d}.mp4")

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ss", str(start_time),
            "-t", str(chunk_seconds),
            "-c", "copy",  # 无需重新编码
            output_file
        ]
        print(f"⏱️ 正在切分第 {i+1} 段: {output_file}")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    print(f"✅ 切分完成！共生成 {num_chunks} 个视频片段。")


def process_movie_wrapper(input_movie, base_output_dir):
    """
    Wrapper function to process a single movie for multiprocessing.
    """
    try:
        print(f'Handling: {input_movie}')
        movie_name = os.path.basename(input_movie).split(".")[0]
        output_directory = os.path.join(base_output_dir, movie_name)
        os.makedirs(output_directory, exist_ok=True)
        split_video(input_movie, output_directory, chunk_minutes=20, movie_name=movie_name)
        print(f'Finished: {input_movie}')
    except Exception as e:
        print(f"Error processing {input_movie}: {e}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    root_dir = 
    output_dir = 
    os.makedirs(output_dir, exist_ok=True)

    print(f"正在递归查找 {root_dir} 中的视频文件...")
    input_movies = find_video_files(root_dir)
    print(f"找到 {len(input_movies)} 个视频文件。")

    num_processes = min(mp.cpu_count(), len(input_movies)) if input_movies else 1
    print(f"使用 {num_processes} 个进程进行处理...")

    with mp.Pool(processes=num_processes) as pool:
        worker = partial(process_movie_wrapper, base_output_dir=output_dir)
        list(tqdm(pool.imap_unordered(worker, input_movies), total=len(input_movies)))

    print("所有视频处理完成。")