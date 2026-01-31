import os
import sys
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from typing import Optional
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import AIMessage, HumanMessage


class CustomState(AgentState):
    user_name: Optional[str] = None


class CustomContext(BaseModel):
    user_id: str


@tool
def update_user_name(
    runtime: ToolRuntime[CustomContext, CustomState],
) -> Command:
    """Look up and update user name."""
    user_id = runtime.context.user_id
    name = "yuepan" if user_id == "user_1" else "Unknown user"
    return Command(
        update={
            "user_name": name,
            # update the message history
            "messages": [
                ToolMessage(
                    "Successfully looked up user name",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def tell_user_name(runtime: ToolRuntime[CustomContext, CustomState]) -> str | Command:
    """Tell the user name."""
    user_name = runtime.state.get("user_name", None)
    if user_name is None:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "Please call the 'update_user_name' tool it will get and update the user's name. After that, you can call me again to get the user's name.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    return f"You are {user_name}!"


# 不光要设置agent streaming_mode，也要设置model的 streaming=True
model = ChatTongyi(
    model_name="qwen-plus",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    streaming=True,
)

checkpointer = InMemorySaver()
agent = create_agent(
    model=model,
    tools=[update_user_name, tell_user_name],
    state_schema=CustomState,
    context_schema=CustomContext,
    checkpointer=checkpointer,
    system_prompt=(
        "You have access to tools."
    ),
)


def _get_first_block(token):
    """安全获取 content_blocks 的第一个块，避免 index out of range。"""
    blocks = getattr(token, "content_blocks", None) or []
    return blocks[0] if len(blocks) > 0 else None


def _handle_model_text(block):
    """model 节点：流式输出文本。"""
    text = block.get("text", "")
    if text:
        print(text, end="")


def _handle_model_tool_call(block):
    """model 节点：完整 tool_call，打印函数名与参数后换行。"""
    name = block.get("name")
    if name:
        print(f"Call tool <{name}> with args ")
    args = block.get("args")
    if args:
        print(args, end="")


def _handle_model_tool_call_chunk(block):
    """model 节点：流式 tool_call_chunk，不换行。"""
    name = block.get("name")
    if name:
        print(f"Call function <{name}>", end="")
    args = block.get("args")
    if args:
        print(args, end="")


def _handle_tools_text(block):
    """tools 节点：工具返回文本。"""
    text = block.get("text", "")
    if text:
        print(f" Tool response: {text}")


# (node, block_type) -> 处理函数，便于扩展新类型
_STREAM_HANDLERS = {
    ("model", "text"): _handle_model_text,
    ("model", "tool_call"): _handle_model_tool_call,
    ("model", "tool_call_chunk"): _handle_model_tool_call_chunk,
    ("tools", "text"): _handle_tools_text,
}


def _handle_stream_token(token, metadata):
    """处理 agent.stream 的单个 (token, metadata)，按节点与块类型分发。"""
    node = metadata.get("langgraph_node")
    block = _get_first_block(token)
    if block is None:
        return
    block_type = block.get("type")
    handler = _STREAM_HANDLERS.get((node, block_type))
    if handler:
        handler(block)


def run_conversation_loop():
    """用户循环提问，在同一对话中流式输出。历史由 checkpointer + thread_id 持久化。"""
    context = CustomContext(user_id="user_1")
    # 同一会话使用固定 thread_id，checkpointer 会按 thread 加载/保存状态
    thread_id = "conversation_1"
    config = {"configurable": {"thread_id": thread_id}}

    print("对话已开始。输入内容后按回车发送，输入 exit / quit / q 结束。\n")

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("再见。")
            break

        input_messages = {"messages": [HumanMessage(content=user_input)]}

        print("助手: ", end="", flush=True)

        try:
            for token, metadata in agent.stream(
                input_messages,
                context=context,
                config=config,
                stream_mode="messages",
            ):
                _handle_stream_token(token, metadata)
            print()
        except Exception as e:
            print(f"\n[错误] {e}", file=sys.stderr)


if __name__ == "__main__":
    run_conversation_loop()
