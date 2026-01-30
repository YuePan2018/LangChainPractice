import os
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel
from typing import Optional
from langchain_community.chat_models.tongyi import ChatTongyi


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
agent = create_agent(
    model=model,
    tools=[update_user_name, tell_user_name],
    state_schema=CustomState,
    context_schema=CustomContext,
    system_prompt=(
        # 如果不指定每一步要解释，那么,中间过程的AIMessage.content会是空。
        "You have access to tools.在调用任何工具前，你必须先解释并输出你要做什么"
    ),
)

output = agent.invoke(
    {"messages": [{"role": "user", "content": "please call my name and greet me"}]},
    context=CustomContext(user_id="user_1"),
)

output_str = (
    str(output)
    .replace("AIMessage", "\nAIMessage")
    .replace("ToolMessage", "\nToolMessage")
)
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(output_str)
# validate that tool is called and final result contains user name
if "ToolMessage" in output_str:
    print("tool called")
if "yuepan" in str(output["messages"][-1].content).lower():
    print("user name is correct")
