import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# 切换工作目录到存放模型的上一级目录
os.chdir(r"C:\Users\qin21\Desktop\Graded Exercise #3\SCSE_26_GEX3")

# 使用相对路径，不要写完整C:\绝对路径
MODEL_DIR = "qwen3-0.6b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype="auto"
)

def rag_answer(user_query: str, context_text: str):
    prompt = """You are university IT support assistant.
Answer ONLY based on the reference context given below.
If the context does NOT contain the answer to user question, reply exactly: I do not know this information.
DO NOT invent any facts, phone numbers or other information that is not present in context.

Reference context:
{context_text}

User question: {user_query}
Answer:"""
    prompt = prompt.format(context_text=context_text, user_query=user_query)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False
    )
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_answer = ans.split("Answer:")[-1].strip()
    return final_answer
