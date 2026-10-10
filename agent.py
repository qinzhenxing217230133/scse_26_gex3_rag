from rag_tool import retrieve_relevant_docs
from rag import rag_answer

def run_agent(question: str):
    # 1.检索相关文档片段
    context = retrieve_relevant_docs(question)
    # 2.调用RAG得到回答
    result = rag_answer(question, context)
    print(f"Question: {question}")
    print(f"Answer: {result}")

if __name__ == "__main__":
    # 作业指定测试问题
    test_question = "What exact phone number should I call for the University IT Service Desk?"
    run_agent(test_question)
