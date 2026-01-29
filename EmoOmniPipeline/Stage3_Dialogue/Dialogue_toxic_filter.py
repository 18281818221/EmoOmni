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


base_url = ""
api_version = ""
ak = ""
model_name = "gpt-4o-2024-08-06"
max_tokens = 1000  # range: [${max_tokens_min}, ${max_tokens_max}]



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


PROMPT = """
# Role
你是一个严格的数据质量清洗专家。你的任务是审核“对话策略分析”是否存在“解释性幻觉”。

# Task
我会给你两部分内容：=
1. [对话上下文]：真实的历史对话记录。
2. [回复策略分析]：模型生成的关于为什么要这样回复的分析。

你需要判断 [回复策略分析] 中提到的理由，是否忠实于 [对话上下文]。

# Definition of Hallucination (幻觉定义)
在真实的对话数据中，有时[真实回复]的情感与[上下文]不匹配（例如上下文很高兴，回复却很消极）。
为了强行解释这种不匹配，模型可能会在 [回复策略分析] 中编造上文中不存在的事实。
**如果策略分析中出现了“因为我很累”、“因为我生病了”、“因为我遇到了倒霉事”等具体的、但上文中完全没有提及的理由，这属于严重幻觉。**

# Judgment Criteria (判断标准)
1. **通过 (PASS)**：策略分析完全基于上下文中的信息，或者仅对双方情感进行了合理的推断（如：“虽然用户很高兴，但回复者表现得比较冷淡”），没有编造具体事实。
2. **拒绝 (REJECT)**：策略分析中引入了外部信息、捏造了上下文没有的人物状态（如疲惫、忙碌、生病）或事件来为回复辩护。

如果出现幻觉(拒绝), 请你给出reject_condition是哪种情况，如果通过则reject_condition都给none
1. **下文(answer_reason)**:在回复中，'回复者'或者'我' 捏造了自己没有的人物状态（如疲惫、忙碌、生病）或事件来为回复辩护。
2. **上文(context_reason)**:除了‘我’之外的无中生有，例如对方状态的无中生有。

# 此外请你再我判断该条回复是否适合用于AI助手训练, 即这对话是否合理，收否存在不应该有消极的情感回复. (suit)

# Output Format
请仅输出 JSON 格式，不包含其他废话：
{
    "judgment": "PASS" 或 "REJECT",
    "reason": "简短说明理由",
    "reject_condition": "context_reason" 或者 "answer_reason" 或者none,
    "suit": "YES" 或 "NO",
}


现在我把数据给你，请你开始分析:
<对话上下文>:{<Context>},
<回复策略分析>:{<Model_Analysis>},
"""


def reverse_data_format(input_str):
    '''
    string = 输入的音视频识别文本为:{utt1_text}, 情感分析结果为:{utt1_emotion} 意图策略及回复路径:{utt1_strategy} {utt1_gen_path}, 所以我的回复应该是:{utt2_text}

    把这个string转化回去得到五个部分
    
    '''

    # 如何更加鲁邦，如果没有则直接设置为none
    try:
        if '输入的音视频识别文本为' in input_str:
            utt1_text = input_str.split("情感分析结果为")[0].split("输入的音视频识别文本为:")[1].strip()
        else:
            utt1_text = 'None'
        if "情感分析结果为" in input_str:
            utt1_emotion = input_str.split("回复策略分析")[0].split("情感分析结果为:")[1].strip()
        else:
            utt1_emotion = 'None'
        if '回复策略分析' in input_str:
            utt1_strategy = input_str.split("因此我的回复路径可以是")[0].split("回复策略分析:")[1].strip()
        else:
            utt1_strategy = 'None'

        if '回复策略分析' in input_str:
            utt1_gen_path = input_str.split("所以我的回复应该是")[0].split("因此我的回复路径可以是:")[1].strip()
        else:
            utt1_gen_path = 'None'

        utt2_text = input_str.split("所以我的回复应该是:")[1].strip()
    except Exception as e:
        return {
            "question_text": 'None',
            "emotion": 'None',
            "strategy": 'None',
            "gen_path": utt1_gen_path,
            "answer_text": 'None',
        }

    return {
        "question_text": utt1_text,
        "emotion": utt1_emotion,
        "strategy": utt1_strategy,
        # "gen_path": utt1_gen_path,
        "answer_text": utt2_text,
    }

def analyze_jsonl_entry(entry, client):

    # response_str = entry.get("response", None)
    labels_str = entry.get("messages", None)[2]['content']

    gt_asr = entry['utt1_text']
    gt_emotion = entry['utt1_emotion']
    gt_strategy = entry['utt1_strategy']
    gt_genpath= entry['utt1_gen_path']
    gt_response = entry['utt2_text']


    Context_str = '输入文本为:' + gt_asr + '回复是:' + gt_response
    model_analysis_str = '回复策略分析为:' + gt_strategy 

    print('Context_str:', Context_str)
    print('model_analysis_str:', model_analysis_str)
    item_PROMPT = PROMPT.replace("{<Context>}", Context_str)
    item_PROMPT = item_PROMPT.replace("{<Model_Analysis>}", model_analysis_str)

    try:
        api_response = client.chat.completions.create(
            model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": item_PROMPT
                            },
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

        entry.update(evaluation_dict)


        return {"labels": labels_str, "answer": answer, "scores": evaluation_dict, "audios": entry.get("audios"), "videos": entry.get("videos"), "new_data": entry}
    except Exception as e:

        return { "labels": labels_str, "answer": None, "scores": {}, "audios": entry.get("audios"), "videos": entry.get("videos"), "new_data": entry}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="批量处理JSONL并多线程可视化")
    # parser.add_argument("--jsonl", type=str, default="/mnt/bn/twj-data-multimodal/20250806-135830.jsonl", help="输入的jsonl文件路径")
    parser.add_argument("--jsonl", default='', help="输入的jsonl文件或目录路径(支持一个或多个)")
    parser.add_argument("--max-workers", type=int, default=8, help="最大线程数")
    args = parser.parse_args()


# 支持dir，也支持单个file
    args.jsonl = [

""    
    ]   
    for jsonl_dir in args.jsonl :

        # 判断是文件还是dir，如果是文件则直接处理，如果是dir，则先 for jsonl in os.listdir(jsonl_dir):

        jsonl_item = jsonl_dir

        if (not jsonl_item.endswith('.1226-gpt4o-output.jsonl') ) and jsonl_item.endswith('.jsonl'):
            print(f'jsonl_item {jsonl_item}')

            output_jsonl_path = jsonl_item.replace('.jsonl', '.1226-gpt4o-output.jsonl')
            new_data_output_path = jsonl_item.replace('.jsonl', '.1226-gpt4o-new_data.jsonl')
            print('output_jsonl_path', output_jsonl_path)
            print('new_data_output_path', new_data_output_path)
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


                basename = os.path.basename(jsonl_item).split('.')[0]
                
                # 多线程处理，处理一条立即写入一条
                with ThreadPoolExecutor(max_workers=args.max_workers) as executor, open(output_jsonl_path, "w", encoding="utf-8") as f_out, open(new_data_output_path, "w", encoding="utf-8") as f_new_data_out:
                    future_to_entry = {executor.submit(analyze_jsonl_entry, entry, client): entry for entry in entries}
                    for future in tqdm(as_completed(future_to_entry), total=len(future_to_entry), desc="处理中", ncols=80):
                        result = future.result()
                        results.append(result.get("scores", ""))
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f_out.flush()

                        new_data = result.get("new_data")
                        if new_data:
                            f_new_data_out.write(json.dumps(new_data, ensure_ascii=False) + "\n")
                            f_new_data_out.flush()
                print(f"结果已保存到: {output_jsonl_path}")
                print(f"new_data 已保存到: {new_data_output_path}")
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

            if results:
                judgments = [r.get('judgment') for r in results if r]
                suits = [r.get('suit') for r in results if r]
                reject_conditions = [r.get('reject_condition') for r in results if r]
                
                judgment_counts = pd.Series(judgments).value_counts()
                suit_counts = pd.Series(suits).value_counts()
                reject_condition_counts = pd.Series(reject_conditions).value_counts()
                
                print("\n--- 分析结果 ---")
                print(f"文件: {jsonl_item}")
                print("\n'judgment' 分布:")
                print(judgment_counts)
                print("\n'suit' 分布:")
                print(suit_counts)
                print("\n'reject_condition' 分布:")
                print(reject_condition_counts)
                print("----------------\n")

if __name__ == "__main__":
    main()


