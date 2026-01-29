from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = '''
我能感觉到她语气里的那种轻微的恼火和防御感，尤其是她紧锁的眉头和紧绷的嘴角。她不是在简单地纠正我，而是在回应我之前可能存在的误解，她感觉自己被我小看了。我的首要任务是立刻消除她的这种感觉。我不能只是简单地说声“哦，抱歉”，那太敷衍了。我计划用一种轻松、甚至带点自嘲的方式来回应，承认我的错误，但要把这个错误描绘成一个无伤大雅的、无心的笑话。这样既能化解尴尬，又能让她觉得我是个有趣的人，而不是一个固执己见的人。"

请你根据我给你的回复策略，用一段话写出回复的语音情感和风格应该是什么样的，不用解释原因。
'''
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
