import asyncio, json, os
from openai import AsyncOpenAI
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession
from agents.mcp.util import MCPUtil  # only import you need

OPENAI_KEY = os.environ["OPENAI_API_KEY"]
ALPHA_KEY = os.environ["ALPHAVANTAGE_API_KEY"]
ALPHA_PATH = "/abs/path/to/mcp-alphavantage/build/index.js"


async def main():
    # 1️⃣  start Alpha‑Vantage MCP server over stdio
    params = StdioServerParameters(
        command="node",
        args=[ALPHA_PATH],
        env={"ALPHAVANTAGE_API_KEY": ALPHA_KEY},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as sess:
            await sess.initialize()

            # 2️⃣  convert all server tools → FunctionTool list
            tools = await MCPUtil.get_function_tools(
                server=sess,
                convert_schemas_to_strict=True,
                run_context=None,
                agent=None,
            )
            openai_tools = [t.to_openai_dict() for t in tools]

            # 3️⃣  ask the model
            client = AsyncOpenAI(api_key=OPENAI_KEY)
            msgs = [
                {"role": "user", "content": "What was TSLA’s closing price yesterday?"}
            ]
            first = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=msgs,
                tools=openai_tools,
                tool_choice="auto",
            )
            call = first.choices[0].message.tool_calls[0]

            # 4️⃣  dispatch the tool → MCP automatically
            t_obj = next(t for t in tools if t.name == call.function.name)
            result = await t_obj.run(None, call.function.arguments)  # <- one‑liner!

            # 5️⃣  feed result back
            msgs += [
                {"role": "assistant", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call.id, "content": result},
            ]
            final = await client.chat.completions.create(
                model="gpt-4o-mini", messages=msgs
            )
            print(final.choices[0].message.content)


asyncio.run(main())
