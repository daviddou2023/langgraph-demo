import json
import os
from typing import Annotated
from typing_extensions import TypedDict

# 1. 导入 dotenv 工具包并加载 .env 文件（必须在加载工具前执行）
from dotenv import load_dotenv
load_dotenv()

# 导入 LangChain 和 LangGraph 的核心组件
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# 导入内存持久化检查点（记忆功能）
from langgraph.checkpoint.memory import InMemorySaver

# 2. 定义状态结构
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# 3. 初始化工具
tool = TavilySearch(max_results=2)
tools = [tool]

# 4. 通过指定提供商和模型名称进行初始化
llm = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="deepseek"
)
# 将工具绑定到语言模型中，生成一个支持工具调用的新模型实例
llm_with_tools = llm.bind_tools(tools)

# 5. 定义 Chatbot 节点
def chatbot(state: State):
    # 【核心修改点 1】：必须使用绑定了工具的 llm_with_tools.invoke 替换原来的 llm.invoke
    # 否则大模型在遇到需要联网的问题时无法生成 tool_calls，从而破坏状态图的完整运行生命周期
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 添加节点
graph_builder.add_node("chatbot", chatbot)

# 6. 定义工具执行节点
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# 7. 定义条件路由逻辑
def route_tools(state: State):
    """
    Use in the conditional_edge to route to the ToolNode if the last message has tool calls.
    Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

# 8. 构建图的条件边和固定边
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)
# 任何时候调用完工具，都必须返回 chatbot 让大模型整合搜索结果并给出最终回答
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# ==================== 增加记忆部分 ====================

# 9. 实例化内存检查点并将图编译
memory = InMemorySaver()
# 在编译时传入 checkpointer 从而激活记忆机制
graph = graph_builder.compile(checkpointer=memory)

# 10. 配置会话 ID (Thread ID)
# 相同的 thread_id 代表同一个会话，Graph 会自动读取该会话的历史记忆
config = {"configurable": {"thread_id": "1"}}

# ==================== 运行与交互循环 ====================

def run_interactive_loop():
    """交互式终端对话逻辑（带记忆）。"""
    print("\n--- 进入实时交互模式 (输入 'q' 退出) ---")
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            # 使用带记忆的 config 流式获取更新
            # 将字典形式的消息列表改用标准规范的 HumanMessage 对象进行流式传入
            events = graph.stream(
                {"messages": [HumanMessage(content=user_input)]}, 
                config, 
                stream_mode="values"
            )
            
            # 用于在流式输出时，防止重复打印历史消息
            printed_messages = set()

            for event in events:
                # 获取该状态更新事件中的最新一条消息
                msg = event["messages"][-1]
                
                # 只有当消息是 AI 说的、并且有内容、且当前事件中还没打印过时才进行打印
                if msg.type == "ai" and msg.content and msg.id not in printed_messages:
                    print(f"Ai: {msg.content}")
                    printed_messages.add(msg.id)
                    
        except Exception as e:
            # 异常降级处理
            print(f"\n[系统提示] 发生异常: {e}，将触发自动演示...")
            fallback_input = "What do you know about LangGraph?"
            print("User: " + fallback_input)
            events = graph.stream(
                {"messages": [HumanMessage(content=fallback_input)]}, 
                config, 
                stream_mode="values"
            )
            for event in events:
                event["messages"][-1].pretty_print()
            break

if __name__ == "__main__":
    # 进入终端实时交互
    run_interactive_loop()
