import json
import os
import time
import re
import base64
from collections import defaultdict
from threading import Semaphore
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import openai

# ==============================================================================
# 1. 配置与限速器
# ==============================================================================

API_CONFIG = {
    "base_url": "",
    "api_version": "",
    "ak": "",
    "model_name": "gemini-2.5-pro-preview-06-05",
    "max_tokens": 4096
}

class QPMLimiter:
    def __init__(self, qpm: int):
        self.interval = 60.0 / qpm
        self.lock = Semaphore(1)
        self.last_time = time.time()

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_time
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_time = time.time()

# 全局限速器实例
qpm_limiter = None

# ==============================================================================
# 2. Prompt 模板 (保持不变)
# ==============================================================================

COT_PROMPT_TEMPLATE = """# Role
你是一个高情商、善于沟通的 AI 对话专家。
你的任务是根据提供的对话上下文和对方的情感状态，**先进行深度的内心思考（CoT），然后生成一句得体的回复。**

# Input Data Provided
- **对方的话语 (Immediate Trigger)**: {prev_text}
- **对方情感描述 (Contextual Sentiment)**: {prev_sentiment}

# Critical Rule: The "Direct Perception" Simulation
**非常重要**：在编写思考过程（`intent_strategy` 和 `thought_process`）时，**绝对不能**在文字中提及“情感描述”、“标签”、“标注”或“Dataset”等词汇。

你需要把`对方情感描述`中的信息，转化为你**第一人称的直接感知体验**。

# Output Requirements (JSON)
请输出一个 JSON 对象，严格包含以下三个字段：

1.  **intent_strategy (意图规划)**
    - **任务**: 分析对方的深层意图。在感知到对方意图后，规划你的回复策略。
    - **内容**: 分析对方的深层意图。描述为了达成好的沟通效果，你“打算”采取什么策略（如共情、反问、幽默等）。
    - **约束**: 
        - 分析意图必须写得像是当下正在发生。
        - 策略回复必须使用**将来时**或**意愿词**（"我应该..."，"我计划..."）。
        - 你的计划必须逻辑顺承地指向`目标文本`的内容，但不能直接抄写`目标文本`。

2.  **path (生成执行)**
    - **任务**: 将策略转化为具体文本的每一步，用自然语言描述，将策略转化为具体回复的逐步构建过程。
    - **格式**: 自然语言描述，展示你的思考和逻辑连接。
    - **逻辑**: 将`目标文本`拆解。“我现在的生成意图是什么”，然后对应的`目标文本`原文。

3.  **response (最终回复)**
    - **任务**: 基于上述策略和思维过程，输出最终通过嘴巴说出来的回复文本。
    - **要求**: 回复必须自然、流畅，且完美符合你在 `intent_strategy` 中设定的策略。

# One-Shot Example
**Input:**
- 上文文本: "I've rewritten this paragraph five times and it still sounds complete garbage."
- 上文情感描述: "The speaker rubs their temples, sighs heavily, and has a tone of exhaustion mixed with deep frustration." (说话者揉着太阳穴，重重地叹气，语气中夹杂着精疲力竭和深深的挫败感。)

**Output (JSON):**

# Example Output Format (JSON)
{{
  "intent_strategy": "我观察到对方揉太阳穴和重重叹气的动作，这表明他不仅仅是在批评自己的作品，更多的是处于一种极度脑力透支和自我怀疑的状态。这时候如果我直接提供修改建议（如“试试这个词...”），反而会增加他的压力。我的最佳策略是：先进行‘情绪急救’，承认并共情他的疲惫感，建议他暂停休息，从而打破这个死循环。我要从‘解决问题’转向‘关怀人’。",
  "path": "首先，我不能顺着他说‘是啊，有时候写东西就是很难’，这样不够有力。我需要根据他揉太阳穴的动作，直接提出‘休息’的建议,比如->"Hey, take a breath."。第一步，先用温和的语气安抚他，让他从屏幕前移开->"You're burning the candle at both ends. Why not step away from the screen for 15 minutes?”。第二步，给他一个合理的心理台阶——告诉他‘由于疲劳产生的隧道视野’是正常的，不是能力问题。最后，用鼓励性的口吻结尾->"Sometimes the best writing happens after you let your brain reset."。",
  "response": "Hey, take a breath. You're burning the candle at both ends. Why not step away from the screen for 15 minutes? Sometimes the best writing happens after you let your brain reset."
}}

# Now, process the specific data below:
"""

def call_gemini_cot(text, sentiment):
    """调用 API 生成 CoT"""
    if not text or not sentiment:
        return None, "Skipped: Missing text or sentiment"

    # 填充 Prompt
    prompt_text = COT_PROMPT_TEMPLATE.format(
        prev_text=text,
        prev_sentiment=sentiment,
    )

    try:
        client = openai.AzureOpenAI(
            azure_endpoint=API_CONFIG["base_url"],
            api_version=API_CONFIG["api_version"],
            api_key=API_CONFIG["ak"],
        )
        
        # 等待限速
        if qpm_limiter:
            qpm_limiter.wait()

        completion = client.chat.completions.create(
            model=API_CONFIG["model_name"],
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=API_CONFIG["max_tokens"],
            response_format={"type": "json_object"} 
        )
        
        response_content = completion.choices[0].message.content
        # 清理可能存在的 Markdown 代码块标记
        cleaned_str = re.sub(r'```json\s*|\s*```', '', response_content).strip()
        cot_data = json.loads(cleaned_str)
        return cot_data, "SUCCESS"

    except Exception as e:
        return None, f"LLM Error: {str(e)}"
    
def process_single_item(item, save_dir):
    """处理单条数据并保存"""
    video_path = item.get('videos')
    if isinstance(video_path, list):
        video_path = video_path[0]

    if not video_path:
        return 0

    key = os.path.splitext(os.path.basename(video_path))[0]
    
    output_path = os.path.join(save_dir, f"{key}.json")
    
    # 断点续跑：如果文件已存在，直接跳过
    if os.path.exists(output_path):
        return 1
    
    text = ""
    sentiment = ""

    if "utt1_text" in item and item["utt1_text"]:
        # 格式1: 包含 utt1_text 和 utt1_emotion
        text = item.get("utt1_text", "")
        sentiment = item.get("utt1_emotion", "")
    else:
        # 格式2: 需要从外部JSON文件读取
        supplementary_json_dir = ""
        supplementary_json_path = os.path.join(supplementary_json_dir, f"{key}.json")

        if os.path.exists(supplementary_json_path):
            try:
                with open(supplementary_json_path, 'r', encoding='utf-8') as f:
                    sup_data = json.load(f)
                text = sup_data.get("audio_desc", {}).get("speech_content", "")
                sentiment = sup_data.get("sentiment_analysis", "")
            except Exception as e:
                print(f"Error reading supplementary file for {key}: {e}")
                return 0
        else:
            # 如果补充文件不存在，则跳过
            return 0

    response_str = item.get("response", None)
    
    if isinstance(response_str, list):
        response_str = response_str[0]

    if response_str and response_str.startswith("system\nYou are Qwe"):
        # 过滤掉system prompt
        parts = response_str.split("\nassistant\n")
        if len(parts) > 1:
            response_str = parts[1]

    response = response_str

    # 调用大模型
    cot_result, status = call_gemini_cot(text, sentiment)
    
    if status == "SUCCESS" and cot_result:
        # 将原始 Key 信息也放进去，方便后续追踪（可选）
        cot_result["key"] = key
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cot_result, f, ensure_ascii=False, indent=2)
        return 1
    else:
        # 失败时可以选择记录日志，这里简单打印
        # print(f"Failed {key}: {status}")
        return 0
    
def generate_cot_dataset(input_jsonl_path, output_base_dir, qpm=150, max_workers=10):
    global qpm_limiter
    qpm_limiter = QPMLimiter(qpm)

    # 1. 确定具体的输出子文件夹
    # 获取文件名（不带后缀），例如 "step1_jsonl_part1"
    filename_stem = os.path.splitext(os.path.basename(input_jsonl_path))[0]
    specific_output_dir = os.path.join(output_base_dir, filename_stem)

    if not os.path.exists(specific_output_dir):
        os.makedirs(specific_output_dir, exist_ok=True)
        print(f"创建输出目录: {specific_output_dir}")
    else:
        print(f"输出目录已存在，将执行增量更新: {specific_output_dir}")

    # 2. 读取所有数据
    all_items = []
    print(f"正在读取文件: {input_jsonl_path}")
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                all_items.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    print(f"共加载 {len(all_items)} 条数据。开始并发处理...")
    
    total_success = 0
    
    # 3. 线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_item = {
            executor.submit(process_single_item, item, specific_output_dir): item 
            for item in all_items
        }
        
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(future_to_item), total=len(all_items), desc="Processing"):
            try:
                result = future.result()
                total_success += result
            except Exception as e:
                print(f"未捕获的线程异常: {e}")

    print(f"\n全部完成！共确保 {total_success}/{len(all_items)} 个文件存在于 {specific_output_dir}")

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":

    INPUT_FILE = ""

    # 输出根目录
    OUTPUT_BASE_DIR = ""

    QPM_SETTING = 120  # 根据你的 Key 限速情况调整
    MAX_WORKERS = 20   # 并发线程数
    
    # 检查输入文件是否存在
    if os.path.exists(INPUT_FILE):
        generate_cot_dataset(INPUT_FILE, OUTPUT_BASE_DIR, qpm=QPM_SETTING, max_workers=MAX_WORKERS)
