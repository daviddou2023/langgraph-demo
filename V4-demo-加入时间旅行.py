import json
import os
from typing import Annotated
from typing_extensions import TypedDict

# 1. 导入 dotenv 工具包并加载 .env 文件
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

# 【新增引入】用于构建自定义工具以及处理中断和恢复
from langchain_core.tools import tool
from langgraph.types import interrupt, Command

# 2. 定义状态结构
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# ==================== 新增：人类干预工具 ====================

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human when you are unsure, need a decision, or lack information."""
    # 触发中断，挂起当前节点的执行，并将 query 传递到外部
    human_response = interrupt({"query": query})
    # 外部恢复 (resume) 执行后，返回的数据将作为此处的返回值
    return human_response["data"]

# 3. 初始化工具
# 为了避免变量名冲突，将原先的 tool 改名为 tavily_tool
tavily_tool = TavilySearch(max_results=2)
# 将联网搜索和人类干预工具组合
tools = [tavily_tool, human_assistance]

# 4. 初始化大模型并绑定工具
llm = init_chat_model(
    model="deepseek-v4-flash", 
    model_provider="deepseek"
)
llm_with_tools = llm.bind_tools(tools)

# 5. 定义 Chatbot 节点
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

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
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# 9. 实例化内存检查点并将图编译
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# 10. 配置会话 ID (Thread ID)
config = {"configurable": {"thread_id": "1"}}


# ==================== 运行与交互循环 (带时间旅行功能) ====================

def run_interactive_loop():
    print("\n--- 进入实时交互模式 (输入 'q' 退出) ---")
    print("💡 [新功能指令]:")
    print("   输入 '/history' 查看历史节点 (Checkpoints)")
    print("   输入 '/revert <id>' 穿越到指定历史节点，开启新分支")
    
    # 用于存储用户希望穿越回的特定 checkpoint_id
    target_checkpoint_id = None
    
    while True:
        try:
            # 1. 检查图的状态是否被中断挂起
            state = graph.get_state(config)
            
            if state.tasks and state.tasks[0].interrupts:
                interrupt_data = state.tasks[0].interrupts[0].value
                print(f"\n🔔 [AI 触发人工求助]: {interrupt_data.get('query', '需要你的帮助')}")
                human_input = input("🗣️ 提供协助信息 (输入 'q' 退出): ")
                
                if human_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                
                stream_input = Command(resume={"data": human_input})
                
            else:
                # 2. 正常用户交互与特殊指令解析
                user_input = input("\nUser: ").strip()
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                
                # --- 时间旅行功能：查看历史 ---
                if user_input.lower() == "/history":
                    print("\n⏳ === 会话历史节点 (Checkpoints) ===")
                    # 获取该 thread_id 下的所有历史状态
                    history_states = list(graph.get_state_history(config))
                    if not history_states:
                        print("当前没有历史记录。")
                        continue
                        
                    # 打印最近的 10 条历史状态
                    for s in history_states[:10]:
                        c_id = s.config['configurable']['checkpoint_id']
                        # 提取最后一条消息作为摘要
                        if s.values.get("messages"):
                            last_msg = s.values["messages"][-1]
                            msg_preview = f"[{last_msg.type.upper()}] {last_msg.content[:30]}..."
                        else:
                            msg_preview = "[EMPTY] 起始状态"
                        print(f"🔹 ID: {c_id}\n   摘要: {msg_preview}")
                    print("===================================\n")
                    continue
                
                # --- 时间旅行功能：穿越到特定节点 ---
                if user_input.lower().startswith("/revert"):
                    parts = user_input.split()
                    if len(parts) == 2:
                        target_checkpoint_id = parts[1]
                        print(f"\n[系统提示] 🌀 目标已锁定！你现在的状态停留在节点 `{target_checkpoint_id}`。")
                        print("接下来发送的消息，将从该节点分叉 (Branching) 继续对话。")
                    else:
                        print("\n[系统提示] ❌ 指令错误。用法: /revert <checkpoint_id>")
                    continue
                
                # 构建正常的输入
                stream_input = {"messages": [HumanMessage(content=user_input)]}
            
            # 3. 准备执行图的配置
            stream_config = config.copy()
            # 如果存在时间旅行目标，将其注入配置
            if target_checkpoint_id:
                stream_config["configurable"]["checkpoint_id"] = target_checkpoint_id
                # 注入完成后清除记录。LangGraph 会自动将新分支的末端设为下一次对话的起点
                target_checkpoint_id = None
            
            # 4. 流式执行并打印输出
            events = graph.stream(
                stream_input, 
                stream_config, 
                stream_mode="values"
            )
            
            printed_messages = set()

            for event in events:
                msg = event["messages"][-1]
                
                if msg.type == "ai" and msg.content and msg.id not in printed_messages:
                    print(f"🤖 Ai: {msg.content}")
                    printed_messages.add(msg.id)
                    
        except Exception as e:
            print(f"\n[系统提示] 发生异常: {e}")
            break

if __name__ == "__main__":
    run_interactive_loop()