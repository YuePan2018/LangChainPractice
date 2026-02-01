# state_in_tool.py

简要说明：

- 演示在 LangChain 代理中：嵌套调用工具（tools），自定义状态（CustomState），自定义上下文（CustomContext）、以及流式输出。
- `InMemorySaver` 做对话持久化；通过 `stream_mode="messages"` 流式输出

运行要求：

- 环境变量 `DASHSCOPE_API_KEY`（用于 `ChatTongyi`）。

学习经验：

- 这个项目起源于langchain官网的教程。但是我复制官网代码运行时，发现agent并没有如同预想的调用tool，而是直接打招呼。更换model也无法解决这个问题。猜想是因为原始代码的greet要求，被agent认定为基础功能，不需要调用tool。经过调试，我发现需要调整tool name， tool description以及prompt，才能让agent意识到，必须用tool，此时agent也能用预想的顺序调用tool
- langchain agent的AImessage被要求结构化输出，所以不会显示思考过程。如果不在system prompt中指定“每一步要解释”，那么,中间过程的AIMessage.content会是空。
- enable streaming时，不光要设置agent，也要设置model
- stream_mode="messages"模式，是按照token增量streaming。需要解析token的内容，给用户区分显示中间过程的tool以及final AI message。