# filter_dialogues_azure_api_fixed.py
import os
import json
import argparse
import openai
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
import asyncio
import re

# --- 1. [不变] 系统和用户Prompt模板 ---
SYSTEM_PROMPT = """
### 角色 ###
你是一位顶尖的数据标注专家，专门为训练大型语言模型筛选高质量的对话数据。你的核心任务是评估对话片段的内部逻辑连贯性和信息价值。

### 任务 ###
分析下面提供的对话片段，判断它是否构成了一个逻辑连贯、有意义的对话交流单元。这个单元将被用来训练一个大型语言模型，因此它需要展示出自然的语言交流模式。

### 核心评估标准 ###
1.  **逻辑连贯性 (最重要的标准)**：对话中的每一句话是否都是对前文的合理延续、回答或反应？整个片段是否围绕一个或多个相关联的主题展开？是否存在突然的、无逻辑的话题跳跃？
2.  **对话交互性**：片段中是否存在至少一次有意义的“问与答”、“请求与回应”或“陈述与评论”这样的多轮交互？一个简单的陈述后接另一个无关的陈述，价值较低。
3.  **信息完整性**：虽然片段可能只是更长对话的一部分，但它本身是否传达了一个相对完整的意图或信息点？例如，一个完整的问路、一次简短的争论、一次任务的交接等。

### 数据是“好”的 (我们需要的) ###
- 包含多轮问答的片段。
- 角色之间围绕一个主题进行讨论或争论。
- 即使开头是“所以...”、“但是...”，只要后续的对话是连贯的，我们也认为它是好的。
- 结尾即便是一个开放性问题（如“那边的情况怎么样？”），只要它逻辑上承接了前面的内容，我们也认为它是好的。

### 数据是“坏”的 (需要被过滤掉的) ###
- 独白：只有一个人在连续说话，没有互动。
- 逻辑断裂：几句话之间毫无关联，像是随机拼接的。
- 意义不明：对话内容过于破碎，无法理解其主旨。例如，只有“嗯”、“啊”、“好的”这类无意义的应答。

### 输出格式 ###
必须严格按照下面的JSON格式提供你的分析结果，不要添加任何额外的解释或文字。
{
  "is_coherent_and_valuable": <true 或 false>,
  "confidence": "<High/Medium/Low>",
  "reason": "<请用一句话简要说明你的判断理由。例如：该片段围绕询问情况展开，包含多轮问答，逻辑连贯。或者：该片段仅包含无意义的应答，缺乏信息价值。>"
}

### 示例1 (坏的数据)
对话文本:
Speaker 1: 哈哈哈哈。
Speaker 2: 嗯。
Speaker 1: 走。

输出:
{
  "is_coherent_and_valuable": false,
  "confidence": "High",
  "reason": "该片段由无明确意义的感叹词和单个指令组成，缺乏连贯的对话上下文和信息价值。"
}
"""
USER_PROMPT_TEMPLATE = "### 对话文本 ###\n{dialogue_text}"


def format_dialogue_from_jsonl(filepath):
    """从jsonl文件中读取内容并格式化为对话字符串。"""
    lines = []
    speakers = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    lines.append(f"Speaker {data['speaker']}: {data['text']}")
                    speakers.add(data['speaker'])
                except json.JSONDecodeError:
                    continue
        has_interaction = len(speakers) > 1
        return "\n".join(lines), len(lines), has_interaction
    except Exception as e:
        print(f"  警告：读取文件 {os.path.basename(filepath)} 时出错: {e}")
        return None, 0, False

async def call_custom_api(dialogue_text, client, model_name, max_tokens):
    """
    [异步] 使用 openai.AsyncAzureOpenAI 客户端调用API。
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(dialogue_text=dialogue_text)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
    ]

    retries = 3
    for i in range(retries):
        try:
            completion = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"  API调用失败 (尝试 {i+1}/{retries}): {e}")
            await asyncio.sleep(5)
    return None

def parse_llm_response(response_text):
    """
    [不变] 增强版JSON解析函数。
    """
    if not response_text:
        return None
    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            return None
    except json.JSONDecodeError:
        print(f"  警告：JSON解析失败，即使在提取后。内容: {response_text}")
        return None

def append_result_to_csv(result_data, output_filepath):
    """[不变] 追加单条结果到CSV文件。"""
    file_exists = os.path.isfile(output_filepath)
    df = pd.DataFrame([result_data])
    df.to_csv(output_filepath, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')

async def process_single_file(filepath, args, client, semaphore):
    """
    [新增] 异步处理单个文件的核心逻辑。
    """
    async with semaphore:
        filename = os.path.relpath(filepath, args.input_dir)
        dialogue_text, line_count, has_interaction = format_dialogue_from_jsonl(filepath)

        if not dialogue_text or not (args.min_lines <= line_count <= args.max_lines) or not has_interaction:
            return None

        response_text = await call_custom_api(dialogue_text, client, args.model_name, args.max_tokens)
        
        result_to_save = None
        if not response_text:
            result_to_save = {'file': filename, 'is_coherent_and_valuable': 'API_Error', 'confidence': 'N/A', 'reason': 'API call failed or returned empty response.'}
        else:
            parsed_result = parse_llm_response(response_text)
            if parsed_result:
                result_to_save = {
                    'file': filename,
                    'is_coherent_and_valuable': parsed_result.get('is_coherent_and_valuable'),
                    'confidence': parsed_result.get('confidence'),
                    'reason': parsed_result.get('reason')
                }
            else:
                sanitized_reason = response_text.replace('\n', ' ').replace('"', "'")
                result_to_save = {
                    'file': filename, 
                    'is_coherent_and_valuable': 'Parse_Error', 
                    'confidence': 'N/A', 
                    'reason': f'Failed to parse. Raw response: {sanitized_reason}'
                }
        return result_to_save

async def main():
    parser = argparse.ArgumentParser(description="[并发版] 使用自定义Azure-compatible API筛选逻辑连贯的对话片段。")
    parser.add_argument('-i', '--input_dir', required=True, help='包含切分后场景jsonl文件的文件夹路径。')
    parser.add_argument('-o', '--output_csv', default='dialogue_coherence_report.csv', help='保存分析结果的CSV文件名。')
    parser.add_argument('--min_lines', type=int, default=2, help='要处理的对话片段的最小行数（句子数）。')
    parser.add_argument('--max_lines', type=int, default=10, help='要处理的对话片段的最大行数（句子数）。')
    
    # API相关参数
    parser.add_argument('--ak', default="",help='API Key (ak)。')
    parser.add_argument('--base_url', default="", help='API的base_url。')
    parser.add_argument('--api_version', default="", help='API的版本。')
    parser.add_argument('--model_name', default="gemini-2.5-pro-preview-06-05", help='要调用的模型名称。')
    parser.add_argument('--max_tokens', type=int, default=8000, help='响应的最大token数。')
    parser.add_argument('--concurrency', type=int, default=10, help='并发请求的数量。')
    
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_files = set()
    if os.path.exists(args.output_csv):
        print(f"输出文件 '{args.output_csv}' 已存在，将读取已处理文件列表并跳过它们。")
        try:
            df_existing = pd.read_csv(args.output_csv)
            if 'file' in df_existing.columns:
                processed_files = set(df_existing['file'])
                print(f"已从CSV中加载 {len(processed_files)} 个已处理文件的记录。")
        except Exception as e:
            print(f"警告：读取现有CSV文件失败: {e}。将继续执行，但可能产生重复记录。")

    # [异步] 初始化API客户端
    try:
        client = openai.AsyncAzureOpenAI(
            azure_endpoint=args.base_url,
            api_version=args.api_version,
            api_key=args.ak,
        )
    except Exception as e:
        print(f"致命错误：创建API客户端失败: {e}")
        return

    all_files = [os.path.join(root, file) for root, _, files in os.walk(args.input_dir) for file in files if file.endswith('.jsonl')]
    files_to_process = [f for f in all_files if os.path.relpath(f, args.input_dir) not in processed_files]
    
    print(f"总共发现 {len(all_files)} 个 .jsonl 文件。")
    print(f"已处理 {len(processed_files)} 个文件。")
    print(f"本次将处理 {len(files_to_process)} 个新文件 (并发数: {args.concurrency})。")
    print(f"处理条件: 对话行数在 [{args.min_lines}, {args.max_lines}] 之间，且包含多方交互。")

    if not files_to_process:
        print("\n处理完成，没有发现任何需要处理的新文件。")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [process_single_file(filepath, args, client, semaphore) for filepath in files_to_process]

    evaluated_count = 0
    for future in tqdm_asyncio.as_completed(tasks, desc="评估对话连贯性"):
        result = await future
        if result:
            append_result_to_csv(result, args.output_csv)
            evaluated_count += 1

    print(f"\n处理完成！")
    print(f"本次运行评估并保存了 {evaluated_count} 个符合条件的文件。")
    
    # 最终文件总数
    total_records = 0
    if os.path.exists(args.output_csv):
        total_records = len(pd.read_csv(args.output_csv))
    print(f"输出文件 '{args.output_csv}' 中现在总共有 {total_records} 条记录。")


if __name__ == '__main__':
    asyncio.run(main())