import json
import os
import json
import sys
import os
import re
import unicodedata
import sys
import multiprocessing
import fcntl
import sys
import transformers
import wandb
import numpy as np
import random
import math
from multiprocessing import Lock
import json
from tqdm import tqdm
import jiwer
import torch
from zhon.hanzi import punctuation
import string
from jiwer import compute_measures
import numpy as np

punctuation_all = punctuation + string.punctuation


def clean_punctuation(text, keep_puncts=None):
    """
    清理文本中的符号，只保留指定的关键标点
    
    参数：
        text: 原始文本字符串
        keep_puncts: 保留的标点集合，默认保留中英文句号、逗号、问号、感叹号
    返回：
        清理后的文本
    """
    if keep_puncts is None:
        keep_puncts = {'。', '，', '？', '！', '.', ',', '?', '!', ' '}

    escaped_puncts = [re.escape(p) for p in keep_puncts]
    
    # 注意加了空格
    pattern = f"[^\u4e00-\u9fa5a-zA-Z0-9 {''.join(escaped_puncts)}]"

    cleaned = re.sub(pattern, '', text)
    
    # 多个标点合并成一个
    for p in keep_puncts:
        cleaned = re.sub(f"{re.escape(p)}+", p, cleaned)

    return cleaned
    
from modelscope.utils.constant import Tasks
# import whisper

JSONL_FILE_PATH = ""
# swift 
WHISPER_MODEL_SIZE = "large"  # Whisper模型大小：tiny/base/small/medium/large
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择GPU/CPU
# ==================================================

def is_chinses(text):
    """判断文本是否包含中文字符"""
    return any('\u4e00' <= char <= '\u9fff' for char in text)
def init_models():
    """初始化Whisper（英语）和Paraformer（中文）模型"""
    print("正在加载模型...")
    
    # 初始化Whisper模型（英语识别）
    # whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)
    # print(f"Whisper模型 {WHISPER_MODEL_SIZE} 加载完成（设备：{DEVICE}）")
    from transformers import pipeline
    whisper_model = pipeline(
        "automatic-speech-recognition",
        model="./hf_cache/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1",
        tokenizer="./hf_cache/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1",
        device=DEVICE
    )
    from modelscope.pipelines import pipeline
    # 初始化Paraformer模型（中文识别）
    paraformer_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        device=DEVICE
    )
    print("Paraformer模型加载完成（设备：{DEVICE}）")
    
    return whisper_model, paraformer_pipeline

def recognize_audio(audio_path, lang, whisper_model, paraformer_pipeline):
    """
    根据语言类型调用对应模型识别音频
    Args:
        audio_path: 音频文件路径
        lang: 语言标识（en/zh）
        whisper_model: Whisper模型实例
        paraformer_pipeline: Paraformer管道实例
    Returns:
        str: 识别出的文本（识别失败返回空字符串）
    """
    if not os.path.exists(audio_path):
        print(f"警告：音频文件不存在 {audio_path}")
        return ""
    
    try:
        if lang.lower() == "en":
            # Whisper识别英语音频
            result = whisper_model(audio_path, language="en")
            print(result.keys())
            recognized_text = result["text"].strip()
        elif lang.lower() == "zh":
            # Paraformer识别中文音频
            result = paraformer_pipeline(audio_path)
            recognized_text = result[0]["text"].strip()
        else:
            print(f"警告：不支持的语言 {lang}，跳过识别")
            recognized_text = ""
        
        return recognized_text
    except Exception as e:
        print(f"错误：识别音频 {audio_path} 失败 - {str(e)}")
        import traceback
        traceback.print_exc()
        return ""


def process_one(hypo, truth, lang):
    raw_truth = truth
    raw_hypo = hypo

    for x in punctuation_all:
        if x == '\'':
            continue
        truth = truth.replace(x, '')
        hypo = hypo.replace(x, '')

    truth = truth.replace('  ', ' ')
    hypo = hypo.replace('  ', ' ')

    if lang == "zh":
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo])
    elif lang == "en":
        truth = truth.lower()
        hypo = hypo.lower()
    else:
        raise NotImplementedError


    truth = clean_punctuation(truth)
    hypo = clean_punctuation(hypo)

    measures = compute_measures(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures["wer"]
    subs = measures["substitutions"] / len(ref_list)
    dele = measures["deletions"] / len(ref_list)
    inse = measures["insertions"] / len(ref_list)
    return (truth, hypo, wer, subs, dele, inse)


def process_jsonl(jsonl_path, whisper_model, paraformer_pipeline):
    """
    处理JSONL文件，批量识别音频并计算WER
    Args:
        jsonl_path: JSONL文件路径
        whisper_model: Whisper模型实例
        paraformer_pipeline: Paraformer管道实例
    Returns:
        dict: 统计结果（总条数、成功条数、平均WER等）
    """
    if not os.path.exists(jsonl_path):
        print(f"错误：JSONL文件不存在 {jsonl_path}")
        return {}
    # 初始化统计变量
    total_count = 0
    success_count = 0
    total_wer = 0.0
    results = []
    from tqdm import tqdm
    import json
    
    print("\n开始处理音频识别和WER计算...")

    # 一边处理一边写入output.jsonl
    output_file = jsonl_path.replace(".jsonl", "_output.jsonl")
    with open(jsonl_path, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out_f:
        for line_num, line in tqdm(enumerate(f, 1), desc="处理进度"):
            line = line.strip()
            if not line:
                continue
            
            total_count += 1
            # 解析JSON行
            data = json.loads(line)
            audio_path = data.get("generated_audio")
            print(audio_path)
            # reference_text = data.get("text_to_speak")


            reference_text = data.get("response")
            if isinstance(reference_text, list):
                reference_text = reference_text[0]
            if reference_text.startswith("system\nYou are Qwe"):
                # 过滤掉system prompt
                reference_text = reference_text.split("\nassistant\n")[1]
            if "所以我的回复应该是:" in reference_text:
                reference_text = reference_text.split("所以我的回复应该是:")[1].strip()



            if reference_text is None or reference_text.strip() == "":
                continue
            
            lang = "zh" if is_chinses(reference_text) else "en"
            # 校验必要字段
            if not audio_path or not reference_text or not lang:
                print(f"第{line_num}行：缺少必要字段（audio_path/reference_text/lang）")
                continue
            
            # 音频识别
            recognized_text = recognize_audio(audio_path, lang, whisper_model, paraformer_pipeline)
            if not recognized_text:
                continue


            processed_truth, processed_hypo, wer, subs, dele, inse = process_one(recognized_text, reference_text, lang=lang)
            # 计算WER
            # wer = process_one(reference_text, recognized_text)
            total_wer += wer
            success_count += 1
            
            # 保存结果
            result = {
                "line_num": line_num,
                "audio_path": audio_path,
                "lang": lang,
                "reference_text": reference_text,
                "recognized_text": recognized_text,
                "wer": round(wer, 4)
            }
            results.append(result)
            
            # 打印单条结果
            print(f"\n第{line_num}行处理完成：")
            print(f"  音频路径：{audio_path}")
            print(f"  语言：{lang}")
            print(f"  参考文本：{reference_text}")
            print(f"  识别文本：{recognized_text}")
            print(f"  WER值：{wer:.4f}")

            # 写入output.jsonl
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
            # except json.JSONDecodeError:
            #     print(f"第{line_num}行：JSON格式错误，跳过")
            # except Exception as e:
            #     print(f"第{line_num}行：处理失败 - {str(e)}")
    
    # 计算平均WER
    avg_wer = total_wer / success_count if success_count > 0 else 1.0
    
    # 输出汇总结果
    print("\n" + "="*80)
    print("处理汇总：")
    print(f"  总数据条数：{total_count}")
    print(f"  成功处理条数：{success_count}")
    print(f"  平均WER值：{avg_wer:.4f}")
    print("="*80)
    
    # 保存结果到JSON文件
    with open("wer_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_count": total_count,
                "success_count": success_count,
                "avg_wer": round(avg_wer, 4)
            },
            "details": results
        }, f, ensure_ascii=False, indent=2)
    print("\n详细结果已保存到 wer_results.json")
    
    return {
        "total_count": total_count,
        "success_count": success_count,
        "avg_wer": avg_wer
    }

if __name__ == "__main__":
    # 安装依赖（首次运行可取消注释）
    # os.system("pip install torch modelscope[audio] openai-whisper jiwer")
    
    # 初始化模型
    whisper_model, paraformer_pipeline = init_models()
    
    # 处理JSONL文件
    process_jsonl(JSONL_FILE_PATH, whisper_model, paraformer_pipeline)