from agents.mcp.util import MCPUtil
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession
import json, asyncio


async def convert_all(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as sess:
            await sess.initialize()
            # Handy one‑liner does list_tools + conversion
            func_tools = await MCPUtil.get_function_tools(
                server=sess,
                convert_schemas_to_strict=True,
                run_context=None,  # ok outside full Runner
                agent=None,
            )
            # Turn each FunctionTool into the raw dict ChatCompletion expects
            openai_tools = [ft.to_openai_dict() for ft in func_tools]
            print(json.dumps(openai_tools, indent=2))


asyncio.run(
    convert_all(
        StdioServerParameters(
            command="node",
            args=["/path/build/index.js"],
            env={"ALPHAVANTAGE_API_KEY": "..."},
        )
    )
)
