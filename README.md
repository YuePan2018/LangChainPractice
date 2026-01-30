# state_in_tool.py

简要说明：

- 这是一个演示如何在 LangChain 代理中嵌套调用工具（tools）并维护自定义状态和上下文的示例脚本。

主要文件：

- `state_in_tool.py`：示例脚本，创建模型、注册工具并调用代理。运行后会把代理返回结构写入 `output.txt`。

运行要求：

- 需要设置环境变量 `DASHSCOPE_API_KEY`（脚本中用于构造 `ChatTongyi`）。

学习经验：

- 这个项目起源于langchain官网的教程。但是我复制官网代码运行时，发现agent并没有如同预想的调用tool，而是直接打招呼。更换model也无法解决这个问题。猜想是因为原始代码的greet要求，被agent认定为基础功能，不需要调用tool。经过调试，我发现需要调整tool name， tool description以及prompt，才能让agent意识到，必须用tool，此时agent也能用预想的顺序调用tool
- langchain agent的AImessage被要求结构化输出，所以不会显示思考过程。如果不在system prompt中指定“每一步要解释”，那么,中间过程的AIMessage.content会是空。