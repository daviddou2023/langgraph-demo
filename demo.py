import os
from typing import Annotated
from typing_extensions import TypedDict

# 导入 LangChain 的标准用户消息类
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv


load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


# 2. 通过指定提供商和模型名称进行初始化
llm = init_chat_model(
    model="deepseek-v4-flash", 
    model_provider="deepseek"
)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# 添加节点与边
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# 编译图
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    # 【核心修改点】：将字典改为用 HumanMessage 对象包装
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    
    for event in graph.stream(initial_state):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


# 交互循环
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except Exception as e:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
