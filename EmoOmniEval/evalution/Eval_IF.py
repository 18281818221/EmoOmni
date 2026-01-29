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

import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import pandas as pd


base_url = 
api_version = 
ak = 
model_name = "gemini-2.5-pro-preview-06-05"
max_tokens = 8192


score_keys =  ["一致性"]

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


def analyze_jsonl_entry(entry, client, gt_dict_map_video_item):

    if "instruction" in entry:
        instruct_string = entry["instruction"]
    elif "answer" in entry:
        instruct_string = entry["answer"]


    generated_audio_path = entry.get("generated_audio")
    print('generated_audio_path', generated_audio_path)
    with open(generated_audio_path, "rb") as audio_file:
        audio_data = audio_file.read()
    speech_encoded_string = base64.b64encode(audio_data).decode('utf-8')



    item_PROMPT = """ # 角色
你是具有丰富声学知识的专家。请根据以下维度描述一段语音，并判断语音与给出的描述是否相符，在一致性维度上输出0/1/2三个等级的分数，忽略非风格因素（音质、自然度等）。
---
## 评价维度
- 音高：声音的感知频率，决定声音是高还是低。通常男性声音音高较低，女性声音音高较高。可基于性别表达相对音高水平，例如：“女性高音”，“男性低沉稳定音”。
- 语速：说话的快慢程度，常在对话中变化。若说话者表现出特定的节奏模式，请指明。
- 音量：说话的响度或轻柔程度，变化可能较大。示例包括低语、正常对话音量或喊叫。
- 音色质感：声音的音色质感，包括甜美、沙哑、深沉、明亮、温暖、鼻音、柔和、粗砺或纤细等描述。这些属性反映生理特征（如声带结构）和风格细微差别，可用于区分说话者或分析情感/表达倾向。
- 情绪：说话时表达的情感，可能在对话中变化。例如，一个人可能开始平静地说，但逐渐变得沮丧，或在同一句话中从悲伤转为笑声。
- 语调：通过声音抑扬传递的情感或态度质感，包括音高变化模式，表达细微差别如讽刺、正式、热情或冷漠。
---
# 评判标准
# 分数层级：
# - 2: 语音与描述的风格完全一致，没有明显偏差。
# - 1: 语音与描述在部分维度上存在差异，但整体风格仍然可以接受。
# - 0: 语音与描述的风格明显冲突，差异较大，无法匹配描述的风格。
---
## 注意事项
* 描述有很大的可能与语音有明显冲突、程度上不符或者完全不匹配，请不要轻易相信给出的描述，应该先对语音保留自己的理解
* 语音与描述的风格在多个维度上应相互匹配。如果描述中提到的是“激动”而语音却没有表现出强烈的情绪变化，应降低一致性评分。
* 若描述中提到的是某个非常明显的特征（如某种特定的音高或情绪），但语音的表现与此差异较大，应给予低评分（0或1）。
* 仅评价**风格一致性**，忽略发音准确性、自然度等非风格因素
* 以描述为唯一依据，不带个人主观偏好
* 对于描述中没有提及的特征，则对于该特征没有限制，不应影响判断
* 当描述仅关注某一维度（如情绪），应侧重该维度判断
---
## 输出格式要求：
请严格使用 JSON 格式，包含一个字典，结构如下：
```json
{
    "音高": ...,
    "语速": ...,
    "情绪": ...,
    ...
    "一致性": 0/1/2
}
```
---
## 待评测的描述及语音：
{<Model_Response>}
"""



    try:

        # item_PROMPT = item_PROMPT.replace("{<Model_Analysis>}", Model_Analysis)
        item_PROMPT = item_PROMPT.replace("{<Model_Response>}", instruct_string)

        print('message:', item_PROMPT)

        api_response = client.chat.completions.create(
            model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": item_PROMPT },
                            {"type": "image_url", "image_url": {"url": f"data:audio/wav;base64,{speech_encoded_string}"}},

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


        return {"response": answer, "labels": answer, "answer": answer, "scores": evaluation_dict, "audios": entry.get("audios"), "videos": entry.get("videos"), "generated_audio": generated_audio_path, "instruction": instruct_string}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(f"API error: {e}")
        return {"response": None, "labels": None, "answer": None, "scores": {}, "audios": entry.get("audios"), "videos": entry.get("videos")}
def main():
    import argparse
    parser = argparse.ArgumentParser(description="批量处理JSONL并多线程可视化")
    parser.add_argument("--jsonl", default='', help="输入的jsonl文件或目录路径(支持一个或多个)")
    parser.add_argument("--max-workers", type=int, default=32, help="最大线程数")
    args = parser.parse_args()


# 支持dir，也支持单个file
    args.jsonl = [

    ] 

    gt_jsonl =

    back_fix = '.v0116-Gemini-Instruct_Speech-output_v1.jsonl'
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

                if len(entries) <= 200:
                    print(f'jsonl_item {jsonl_item} 记录数小于200，跳过')
                    continue
                
                processed_entries_count = 0
                if os.path.exists(output_jsonl_path):
                    with open(output_jsonl_path, "r", encoding="utf-8") as f_in:
                        for line in f_in:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                entry = json.loads(line)
                                results.append(entry.get("scores", ""))
                                processed_entries_count += 1
                            except Exception as e:
                                print(f"跳过无效行: {e}")
                    print(f"输出文件 {output_jsonl_path} 已存在，已加载 {processed_entries_count} 条结果。")

                if processed_entries_count < len(entries):
                    unprocessed_entries = entries[processed_entries_count:]
                    print(f"发现 {len(unprocessed_entries)} 条未处理记录，开始处理...")
                    
                    # 初始化API
                    client = openai.AzureOpenAI(
                        azure_endpoint=base_url,
                        api_version=api_version,
                        api_key=ak,
                    )

                    basename = os.path.basename(jsonl_item).split('.')[0]
                    
                    # 多线程处理，处理一条立即写入一条
                    with ThreadPoolExecutor(max_workers=args.max_workers) as executor, open(output_jsonl_path, "a", encoding="utf-8") as f_out:
                        future_to_entry = {executor.submit(analyze_jsonl_entry, entry, client, gt_dict_map_video_item): entry for entry in unprocessed_entries}
                        for future in tqdm(as_completed(future_to_entry), total=len(future_to_entry), desc="处理中", ncols=80):
                            result = future.result()
                            if result.get("answer", None) is None:
                                continue
                            results.append(result.get("scores", ""))
                            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                            f_out.flush()
                    print(f"结果已保存到: {output_jsonl_path}")
                else:
                    print(f"所有记录均已处理，无需额外操作。")
                    

                # 可视化分数分布
                # 更新评分项键名以匹配新结构
                global score_keys
                score_data = {k: [] for k in score_keys}
                for r in results:
                    for k in score_keys:
                        # 从嵌套字典中提取Score值
                        try:
                            score = r[k]
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

