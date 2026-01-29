import argparse
import yaml
import time
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from tqdm import tqdm
import sys
import os
import glob
import torch
import soundfile as sf
import torch.nn as nn
from utils import demix_track, get_model_from_config
import numpy as np
import warnings
import librosa
import sys

print('sys.path: ', sys.path)
warnings.filterwarnings("ignore")
def sum_to_mono(data):
    """
    将多通道音频相加为单通道
    :param data: 多通道音频数据，形状为 (n_channels, n_samples)
    :return: 单通道音频数据，形状为 (n_samples,)
    """
    return np.sum(data, axis=0)  # 对所有通道相加

def normalize_audio(audio):
    """
    归一化音频数据，避免 clipping
    :param audio: 单通道音频数据
    :return: 归一化后的音频数据
    """
    max_abs = np.max(np.abs(audio))
    if max_abs > 1.0:  # 如果超出范围，进行归一化
        audio = audio / max_abs
    return audio

def worker(device_id, file_list, args, config):
    """
    Worker process for multi-GPU inference.
    """
    device = torch.device(f'cuda:{device_id}')
    
    model = get_model_from_config(args.model_type, config)
    if args.model_path != '':
        print(f'GPU {device_id}: Loading model: {args.model_path}')
        model.load_state_dict(
            torch.load(args.model_path, map_location=torch.device('cpu'))
        )
    model = model.to(device)
    
    print(f"GPU {device_id}: Processing {len(file_list)} files.")
    run_folder(model, args, config, device, file_list, verbose=False)


def run_folder(model, args, config, device, all_mixtures_path, verbose=False):
    start_time = time.time()
    model.eval()

    total_tracks = len(all_mixtures_path)
    print('Total tracks found: {}'.format(total_tracks))
    print('args.input_folder: ', args.input_folder)


    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    if not verbose:
        all_mixtures_path = tqdm(all_mixtures_path)

    first_chunk_time = None



    for track_number, path in enumerate(all_mixtures_path, 1):
        try:    
            path = path.strip()  # 去除行末的换行符
            if '演唱会' in path:
                continue
            filename = os.path.basename(path)[:-4]
            input_dir = os.path.dirname(path)
            instrumental_path = os.path.join(input_dir, "background", f"{filename}.wav")
            if os.path.exists(instrumental_path):
                continue
            # mix, sr = sf.read(path)
            # if len(mix.shape) == 1: 
            #     mix = np.stack((mix, mix), axis=-1)
            

            mix, sr = librosa.load(path, sr=None, mono=False)  # sr=None 保留原始采样率
            print('mix.shape: ', mix.shape)
            if len(mix.shape) == 1: 
                mix = np.stack((mix, mix), axis=-1)
            else:
                mix = mix.T

            mixture = torch.tensor(mix.T, dtype=torch.float32)

            res = {}
            with torch.no_grad():
                chunk_duration = 30  # seconds
                if sr is not None and sr > 0:
                    chunk_samples = int(chunk_duration * sr)
                    num_channels, total_samples = mixture.shape
                    final_results = {}

                    if total_samples > 0:
                        for start_sample in range(0, total_samples, chunk_samples):
                            end_sample = min(start_sample + chunk_samples, total_samples)
                            chunk = mixture[:, start_sample:end_sample]
                            
                            res_chunk, first_chunk_time = demix_track(config, model, chunk, device, first_chunk_time)
                            # print('res_chunk shape: ', res_chunk[instruments[0]].shape)
                            if not final_results:
                                final_results = res_chunk
                            else:
                                for instr in instruments:
                                    final_results[instr] = torch.cat((torch.tensor(final_results[instr], dtype=torch.float32), torch.tensor(res_chunk[instr], dtype=torch.float32)), dim=1)
                    res = final_results
                else:
                    res, first_chunk_time = demix_track(config, model, mixture, device, first_chunk_time)


            for ind, instr in enumerate(instruments):
                if ind == 0:
                    vocals_path = os.path.join(input_dir, "vocal", f"{filename}.wav")

                    os.makedirs(os.path.dirname(vocals_path), exist_ok=True)
                    sf.write(vocals_path, res[instr].T, sr, subtype='FLOAT')
                else:
                        
                    vocals_path = os.path.join(input_dir, "vocal", f"{filename}_{ind}.wav")
                    sf.write(vocals_path, res[instr].T, sr, subtype='FLOAT')

            vocals = res[instruments[0]].cpu().numpy().T
            instrumental = mix - vocals
            os.makedirs(os.path.dirname(instrumental_path), exist_ok=True)

            sf.write(instrumental_path, instrumental, sr, subtype='FLOAT')
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
        # raise "!"

    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))


def proc_folder(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mel_band_roformer')
    parser.add_argument("--config_path", type=str, help="path to config yaml file")
    parser.add_argument("--model_path", type=str, default='./MelBandRoformer.ckpt', help="Location of the model")
    parser.add_argument("--input_folder", type=str, help="folder with songs to process")
    parser.add_argument("--store_dir", default="", type=str, help="path to store model outputs")
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    # device_ids = [0]

    torch.backends.cudnn.benchmark = True

    with open("./configs/config_vocals_mel_band_roformer.yaml") as f:
      config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    all_mixtures_path = glob.glob(os.path.join(args.input_folder, '**', '*.wav'), recursive=True)
    
    unprocessed_paths = []
    for path in all_mixtures_path:
        path = path.strip()
        if "演唱会" in path:
            continue
        if 'vocal' in path:
            continue
        if 'background' in path:
            continue
        
        filename = os.path.basename(path)[:-4]
        input_dir = os.path.dirname(path)
        instrumental_path = os.path.join(input_dir, "background", f"{filename}.wav")

        if os.path.exists(instrumental_path):
            print(f"Skipping {os.path.basename(path)} as instrumental file already exists.")
            continue
        unprocessed_paths.append(path)

    all_mixtures_path = unprocessed_paths
    total_tracks = len(all_mixtures_path)
    print(f'Total tracks to process: {total_tracks}')
    if total_tracks == 0:
        return

    if torch.cuda.is_available():
        if not isinstance(device_ids, list):
            device_ids = [device_ids]

        if len(device_ids) > 1:
            import torch.multiprocessing as mp
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass

            num_gpus = len(device_ids)
            chunks = [[] for _ in range(num_gpus)]
            for i, path in enumerate(all_mixtures_path):
                chunks[i % num_gpus].append(path)

            procs = []
            for i, device_id in enumerate(device_ids):
                if not chunks[i]:
                    continue
                proc = mp.Process(target=worker, args=(device_id, chunks[i], args, config))
                procs.append(proc)
                proc.start()

            for proc in procs:
                proc.join()
        else:
            device_id = device_ids[0] if device_ids else 0
            device = torch.device(f'cuda:{device_id}')
            model = get_model_from_config(args.model_type, config)
            if args.model_path != '':
                print('Using model: {}'.format(args.model_path))
                model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
            model.to(device)
            run_folder(model, args, config, device, all_mixtures_path, verbose=False)
    else:
        device = 'cpu'
        print('CUDA is not available. Run inference on CPU. It will be very slow...')
        model = get_model_from_config(args.model_type, config)
        if args.model_path != '':
            print('Using model: {}'.format(args.model_path))
            model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        model.to(device)
        run_folder(model, args, config, device, all_mixtures_path, verbose=False)


if __name__ == "__main__":
    proc_folder(None)