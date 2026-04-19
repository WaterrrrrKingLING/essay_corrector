import os
from typing import TypedDict, List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph
from dotenv import load_dotenv
import json

load_dotenv()
# testing
# ====== 初始化模型 ======
llm = ChatOpenAI(
    temperature=0.3,
    model="gpt-4o-mini"  # 或 deepseek-chat
)

# ====== 定义状态 ======
class EssayState(TypedDict):
    essay: str
    corrected_text: str
    corrections: List[Dict]
    explanations: List[str]
    scores: Dict
    suggestions: str


# ====== 1️⃣ 纠错节点 ======
def correct_node(state: EssayState):
    prompt = f"""
You are an English teacher.

Correct the following essay.

IMPORTANT:
- You must be deterministic.
- Always choose the most common correction.
- Do not generate alternative expressions.
- Keep the original paragraph structure EXACTLY the same
- Do NOT merge paragraphs
- Do NOT split paragraphs
- Preserve line breaks (\n)
- Use advanced vocabulary and natural collocations
- Increase sentence complexity (use clauses, varied structures)
- Improve coherence and logical flow
- Strengthen arguments with clearer reasoning
- Keep original meaning

Return JSON format:
{{
  "corrected_text": "...",
  "corrections": [
    {{
      "original": "...",
      "corrected": "...",
      "type": "grammar/vocab/spelling"
    }}
  ]
}}

Essay:
{state["essay"]}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        data = json.loads(response.content)
    except:
        data = {
            "corrected_text": response.content,
            "corrections": []
        }

    return {
        "corrected_text": data["corrected_text"],
        "corrections": data["corrections"]
    }


# ====== 2️⃣ 错误解释节点 ======
def explain_node(state: EssayState):
    prompt = f"""
Explain each correction briefly.

Corrections:
{json.dumps(state["corrections"], indent=2)}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "explanations": response.content.split("\n")
    }


# ====== 3️⃣ 评分节点 ======
def score_node(state: EssayState):
    prompt = f"""
Score this essay (0-10):

Return JSON:
{{
  "grammar": x.float,
  "vocabulary": x.float,
  "coherence": x.float,
  "overall": x.float
}}

Essay:
{state["corrected_text"]}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        scores = json.loads(response.content)
    except:
        scores = {"grammar": 5, "vocabulary": 5, "coherence": 5, "overall": 5}

    return {"scores": scores}


# ====== 4️⃣ 建议节点 ======
def improve_node(state: EssayState):
    prompt = f"""
Give suggestions to improve this essay.

Essay:
{state["corrected_text"]}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"suggestions": response.content}


# ====== 构建 LangGraph ======
builder = StateGraph(EssayState)

builder.add_node("correct", correct_node)
builder.add_node("explain", explain_node)
builder.add_node("score", score_node)
builder.add_node("improve", improve_node)

builder.set_entry_point("correct")

builder.add_edge("correct", "explain")
builder.add_edge("explain", "score")
builder.add_edge("score", "improve")

graph = builder.compile()


# ====== 运行测试 ======
if __name__ == "__main__":
    try:
        essay = open("D:\\pytorch深度学习\\langchain\\essay.txt", "r", encoding="utf-8").read()
    except FileNotFoundError:
        print("找不到 essay.txt 文件，请确认路径正确")
        exit()

    print("===== 原文预览 =====")
    print(essay[:200], "...\n")

    result = graph.invoke({"essay": essay})

    print("\n====== 修改后作文 ======")
    print(result["corrected_text"])

    print("\n====== 错误说明 ======")
    for e in result["explanations"]:
        print("-", e)

    print("\n====== 评分 ======")
    print(result["scores"])

    print("\n====== 提升建议 ======")
    print(result["suggestions"])