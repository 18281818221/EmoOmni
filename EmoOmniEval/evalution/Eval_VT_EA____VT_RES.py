import openai
import json
import os
import base64
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

import re
import json
import openai

base_url = 
api_version = 
ak = 
model_name = "gemini-2.5-pro-preview-06-05"
max_tokens = 8192



def extract_and_clean_json(response_content: str) -> (str | None, str | None):
    """
    从原始响应文本中提取并清理 JSON 字符串。
    
    Args:
        response_content (str): 大模型返回的完整文本。

    Returns:
        tuple[str | None, str | None]:
        - 第一个元素是清理后的 JSON 字符串，如果提取失败则为 None。
        - 第二个元素是原始提取但未清理的 JSON 字符串，用于调试，提取失败则为 None。
    """
    json_str = None
    
    # 策略 1: 优先匹配 Markdown 代码块 ```json ... ```
    match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_content, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # 策略 2: 备用方案，匹配裸露的 JSON 对象 { ... }
        match = re.search(r'^\s*(\{.*\})', response_content, re.DOTALL)
        if match:
            json_str = match.group(1)

    if not json_str:
        return None, None # 无法从响应中提取任何 JSON 内容

    original_extracted_str = json_str # 保存一份原始提取的副本
    
    # 清理步骤：移除 } 或 ] 前的拖尾逗号，这是最常见的 LLM JSON 错误
    cleaned_json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
    
    return cleaned_json_str, original_extracted_str

def save_results_to_json(results, output_path):
    """
    将结果保存为JSON文件
    """

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"结果已保存到: {output_path}")


import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import pandas as pd



score_keys =  ["Emotion_Analysis", "Response_Emotional_Strategy", "Response_Content"]


def analyze_jsonl_entry(entry, client, gt_dict_map_video_item):

    response_str = entry.get("response", None)
    if response_str is list:
        response_str = response_str[0]

    if response_str.startswith("system\nYou are Qwe"):
        # 过滤掉system prompt
        response_str = response_str.split("\nassistant\n")[1]

    video_path = entry['videos'][0]
    max_tokens = 8192

    try:
        with open(video_path, "rb") as video_file:
            video_data = video_file.read()
        encoded_string = base64.b64encode(video_data).decode('utf-8')
    except Exception as e:
        return "ERROR", f"Key: {video_path} - 文件读取失败: {e}"


    if len(response_str) > 8192:
        response_str = response_str[:8192]
        return {"response": response_str, "labels": None, "answer": None, "scores": {}, "audios": entry.get("audios"), "videos": entry.get("videos")}

    Model_Analysis = ""

    item_PROMPT = """
    # Role
    你是一位精通多模态交互与情感计算的评估专家。你的任务是基于原始视频输入，评估一个“带有情感思维链对话模型”的性能。

    # Input Data
    1. **<Video_Input>**: 用户输入的原始视频片段。这是判断的 Ground Truth。
    2. **<Model_Response>**: 模型生成的回复，包含对当前用户输入状态的情感分析和回复策略分析以及最终使用的对话回复文本。

    # Evaluation Task
    请基于视频内容，从以下三个维度对模型表现进行评分（0-2分）和点评。

    ## 维度一：输入情感特征分析 (Emotion_Analysis)
    **目标**：评估 `<Model_Response>` 相对于 `<Video_Input>` 中真实情感特征描述的准确性和深度。
    **评分标准 (0-2分)**：
    * **2分 (卓越)**：情绪判断精准，能识别复合情绪或情绪的动态变化。分析逻辑清晰，明确结合了音频（如语调、停顿、颤音）和视频（如微表情、视线方向、肢体动作）的具体证据来支撑结论，并能深入解释情绪产生的潜在原因。
    * **1分 (良好)**：情绪识别基本正确，但仅依赖单一模态（仅看文本、仅看图像或仅看音频）进行推断，缺乏多模态综合分析，或者分析过于浅显。
    * **0分 (错误)**：核心情绪判断错误; 或引用的证据在视频中不存在（幻觉）; 或分析结论与引用证据无逻辑关联。

    ## 维度二：回复情感策略适宜性 (Response_Emotional_Strategy)
    **目标**：评估 `<Model_Response>` 所采取的情感基调和应对策略是否与 `<Video_Input>` 中用户表现出的情绪状态（类型及强度）相匹配。
    **评分标准 (0-2分)**：
    * **2分 (精准/高情商)**：策略得当且**强度匹配**。回复的情感浓度与视频中观察到的**情绪强度**高度契合（例如：面对激动的用户，回复有足够的安抚力度；面对开心的用户，回复有足够的感染力），并能提供额外的情绪价值。
    * **1分 (合格/安全)**：情感基调大体正确，符合基本礼仪，但较为**公式化**或**平淡**。未能根据视频中的微表情或语气细微变化调整语气，属于“安全但缺乏温度”的回复。
    * **0分 (冲突/冷漠)**：情感基调与用户情绪**冲突**（如用户悲伤时模型轻浮），毫无关联的情感表现; 或面对强烈情绪信号时表现得**过度冷漠**和机械，严重破坏对话体验。

    ## 维度三：回复内容相关性与逻辑 (Response_Content Relevance & Logic)
    **目标**：评估 `<Model_Response>` 的回复文本内容是否紧扣 `<Video_Input>` 中的语义信息，以及逻辑是否通顺。
    **评分标准 (0-2分)**：
    * **2分 (高质量)**：回复内容紧扣上下文，逻辑严密，语义通顺。不仅回答了用户的问题或回应了话题，还根据情感分析提供了有价值的信息、建议或引导，推动对话深入。
    * **1分 (合格)**：回复内容相关且逻辑基本通顺，能完成基本的对话任务，但内容较为平庸、通用，啰嗦;或缺乏针对性（“万金油”回复），未考虑到用户的情感状态。
    * **0分 (不可用)**：回复内容离题，与上文没有关联度; 或者出现严重的逻辑错误、事实性错误、存在答非所问的情况; 或产生严重的幻觉。

    # Output Format
    请严格按照以下 JSON 格式输出评估结果，不要包含其他废话：

    ```json
    {
        "Emotion_Analysis": {
            "score": <0, 1, or 2>,
            "reason": "<简短评语，指出对多模态细节的捕捉情况>"
        },
        "Response_Emotional_Strategy": {
            "score": <0, 1, or 2>,
            "reason": "<简短评语，解释情感策略和强度是否匹配>"
        }
        "Response_Content": {
            "score": <0, 1, or 2>,
            "reason": "<简短评语，评价内容的逻辑和相关性>"
        },
    }

    现在我把数据给你，请你开始分析:
    <Model_Response>:{<Model_Response>},
    """



    try:

        # item_PROMPT = item_PROMPT.replace("{<Model_Analysis>}", Model_Analysis)
        item_PROMPT = item_PROMPT.replace("{<Model_Response>}", response_str)

        print('message:', item_PROMPT)

        api_response = client.chat.completions.create(
            model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": item_PROMPT },
                            {"type": "image_url", "image_url": {"url": f"data:video/mp4;base64,{encoded_string}"}},

                        ]
                    },
                ],
            stream=False
        )
        answer = api_response.choices[0].message.content.strip()
        try:
            evaluation_dict = json.loads(answer)
        except:
            try:
                answer, raw = extract_and_clean_json(answer)
                evaluation_dict = json.loads(answer)
            except Exception as e:
                import traceback
                print(traceback.format_exc())


        print(f"Result: {answer}")


        return {"response": response_str, "labels": response_str, "answer": answer, "scores": evaluation_dict, "audios": entry.get("audios"), "videos": entry.get("videos")}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(f"API error: {e}")
        return {"response": response_str, "labels": None, "answer": None, "scores": {}, "audios": entry.get("audios"), "videos": entry.get("videos")}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="批量处理JSONL并多线程可视化")
    # parser.add_argument("--jsonl", type=str, default="/mnt/bn/twj-data-multimodal/20250806-135830.jsonl", help="输入的jsonl文件路径")
    parser.add_argument("--jsonl", default='', help="输入的jsonl文件或目录路径(支持一个或多个)")
    parser.add_argument("--max-workers", type=int, default=32, help="最大线程数")
    args = parser.parse_args()


# 支持dir，也支持单个file
    args.jsonl = [

'/mnt/bn/twj-data-multimodal2/workspace/swift_training/ckpt_output/lm_output_dialogure/0112-7B-multitask-pretrained_stage1-50k_all/CH_SIMSv2_MMLA_test-for_EmoOmni-0112-7B-multitask-pretrained_stage1-50k_all-3500.jsonl'

""
    ] 

    gt_jsonl = "/mnt/bn/twj-data-multimodal2/workspace/swift_training/tools/dialogue_1229/MELD_test_with_cot_sentiment_new-chunk1.1226-gpt4o-new_data-filtered_data_used_for_inference.jsonl"

    back_fix = '.v0116-Gemini-Video_text-output2.jsonl'
    gt_dict_map_video_item = {}

    with open(gt_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                video_id = entry.get("videos")[0]
                if video_id:
                    gt_dict_map_video_item[video_id] = entry
            except Exception as e:
                print(f"跳过无效行: {e}")
    
    for jsonl_dir in args.jsonl :

        # 判断是文件还是dir，如果是文件则直接处理，如果是dir，则先 for jsonl in os.listdir(jsonl_dir):

        try:
            jsonl_item = jsonl_dir

            if (not jsonl_item.endswith(back_fix) ) and jsonl_item.endswith('.jsonl'):
                print(f'jsonl_item {jsonl_item}')

                output_jsonl_path = os.path.splitext(jsonl_item)[0] + back_fix
                # output_jsonl_path = jsonl_item
                print('output_jsonl_path', output_jsonl_path)
                results = []
                if not os.path.exists(output_jsonl_path):
                    # 初始化API
                    client = openai.AzureOpenAI(
                        azure_endpoint=base_url,
                        api_version=api_version,
                        api_key=ak,
                    )
                    # 读取jsonl

                    entries = []
                    with open(jsonl_item, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                entry = json.loads(line)
                                entries.append(entry)
                            except Exception as e:
                                print(f"跳过无效行: {e}")

                    print(f"共读取{len(entries)}条记录")

                    if len(entries) <= 300:
                        print(f'jsonl_item {jsonl_item} 记录数小于300，跳过')
                        continue


                    basename = os.path.basename(jsonl_item).split('.')[0]
                    
                    # 多线程处理，处理一条立即写入一条
                    with ThreadPoolExecutor(max_workers=args.max_workers) as executor, open(output_jsonl_path, "w", encoding="utf-8") as f_out:
                        future_to_entry = {executor.submit(analyze_jsonl_entry, entry, client, gt_dict_map_video_item): entry for entry in entries}
                        for future in tqdm(as_completed(future_to_entry), total=len(future_to_entry), desc="处理中", ncols=80):
                            result = future.result()
                            if result.get("answer", None) is None:
                                continue
                            results.append(result.get("scores", ""))
                            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                            f_out.flush()
                    print(f"结果已保存到: {output_jsonl_path}")
                else:
                    print(f"输出文件 {output_jsonl_path} 已存在，跳过保存步骤。")

                    with open(output_jsonl_path, "r", encoding="utf-8") as f_in:
                        for line in f_in:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                entry = json.loads(line)
                                results.append(entry.get("scores", ""))
                            except Exception as e:
                                print(f"跳过无效行: {e}")
                    

                # 可视化分数分布
                # 更新评分项键名以匹配新结构
                global score_keys
                score_data = {k: [] for k in score_keys}
                for r in results:
                    for k in score_keys:
                        # 从嵌套字典中提取Score值
                        try:
                            score = r[k]['score']
                            if isinstance(score, (int, float)):
                                score_data[k].append(score)
                        except Exception as e:
                            print(f"跳过无效行: {e}")



                # 计算每个描述的平均分
                averages = {k: sum(v)/len(v) if v else 0 for k, v in score_data.items()}
                print("各描述的平均分:", averages)

                print('score_data', score_data)


            # 3. 整合 score data 和 averages 为一个完整数据字典
                saved_data = {
                    "average_score": averages,  # 整体平均分
                    "score_detailed_data": score_data,  # 各维度详细评分数据
                }

                # 4. 保存到 JSON 文件（路径可自定义，如"./evaluation_results.json"）
                save_path = output_jsonl_path.replace('.jsonl', '---avgscore.json')
                # 单独序列化两个字段，应用不同格式
                # average_score 带缩进
                avg_part = json.dumps(
                    {"average_score": saved_data["average_score"]},
                    ensure_ascii=False,
                    indent=4
                )
                # score_detailed_data 无缩进
                # 处理score_detailed_data部分
                # 先将内部value转为字符串，去掉缩进
                detailed_data = saved_data["score_detailed_data"]
                formatted_detailed = {}
                for key, value in detailed_data.items():
                    # 将每个value序列化为无缩进的字符串
                    formatted_detailed[key] = json.dumps(value, ensure_ascii=False)

                # 序列化整个score_detailed_data，key之间有缩进
                detailed_part = json.dumps(
                    {"score_detailed_data": formatted_detailed},
                    ensure_ascii=False,
                    indent=4
                )
                # 修复引号转义问题并拼接
                detailed_part = detailed_part.replace('"\{', '{').replace('\}"', '}').replace('\\', '')
                # 移除各自的外层花括号后拼接成完整JSON
                # 拼接成完整的JSON
                with open(save_path, "w", encoding="utf-8") as f:
                    content = "{\n" + avg_part[1:-1] + ",\n" + detailed_part[1:-1] + "\n}"
                    f.write(content)
                print(f"各描述的平均分数据保存成功！文件路径: {save_path}")


                # 画柱状图并标注比例和平均分
                plt.figure(figsize=(18, 5))
                for i, k in enumerate(score_keys):
                    plt.subplot(1, 3, i+1)
                    data = score_data[k]
                    bins = range(0,5)  # 分数范围0-5
                    counts, _, patches = plt.hist(data, bins=bins, align='left', rwidth=0.8)
                    total = sum(counts)
                    # 在每个柱子上标注比例
                    for count, patch, score in zip(counts, patches, bins):
                        if total > 0 and count > 0:
                            percent = count / total * 100
                            plt.text(patch.get_x() + patch.get_width()/2, count, f"{percent:.1f}%", 
                                    ha='center', va='bottom', fontsize=10)
                    # 标题中显示平均分
                    plt.title(f"{k}\nAvg: {averages[k]:.2f}")
                    plt.xlabel('Score')
                    plt.ylabel('Count')
                plt.tight_layout()


                fig_path = output_jsonl_path.replace('.jsonl', '.png')
                plt.savefig(fig_path)   
                print(f" 分数分布图已保存为 {fig_path}")
        except Exception as e:
            print(f'处理 {jsonl_item} 时出错: {e}')
            continue
if __name__ == "__main__":
    main()

    # 处理视频数据并获取结果
    



# "python /mnt/bn/twj-data-multimodal/twj/workspace/deepseek_test_VideoAudio-forjsonl.py --jsonl /mnt/bn/multimodal-emo-llm-data/mlx/users/zhaozhixian.zzx/workspace/ms-swift/output-all/output-0804-full/v1-20250805-230015/checkpoint-2600/infer_result/20250806-135830.jsonl"