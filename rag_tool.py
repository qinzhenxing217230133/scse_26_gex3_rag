import sys
sys.path.append(".")

from policy_loader import load_policy_documents

def retrieve_relevant_docs(query: str):
    # 按照作业要求，只能使用这个函数加载文档，不要改动policy_loader.py
    documents = load_policy_documents()

    # 简易关键词匹配检索
    query_words = set(query.lower().split())
    scored = []
    for doc in documents:
        text = doc.page_content.lower()
        count = sum(1 for w in query_words if w in text)
        scored.append((count, doc.page_content))

    # 按匹配分数从高到低排序，取前2段作为上下文
    scored.sort(reverse=True, key=lambda x: x[0])
    top_texts = [item[1] for item in scored[:2]]
    return "\n\n".join(top_texts)
