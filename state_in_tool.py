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


model = ChatTongyi(
    model_name="qwen-plus", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
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
            for token, _ in agent.stream(
                input_messages,
                context=context,
                config=config,
                stream_mode="messages",
            ):
                print(token.content_blocks)
            print()
        except Exception as e:
            print(f"\n[错误] {e}", file=sys.stderr)


if __name__ == "__main__":
    run_conversation_loop()
