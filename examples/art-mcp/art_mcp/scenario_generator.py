"""Generate scenarios for MCP agent evaluation using OpenAI's o3 model."""

import json
import os
from typing import List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import openai
from dotenv import load_dotenv

load_dotenv()


async def generate_scenarios(
    server_params: StdioServerParameters, num_scenarios: int = 24
) -> List[str]:
    """Generate scenarios for MCP agent evaluation using OpenAI's o3 model.

    Args:
        server_params: Parameters for the MCP server to analyze
        num_scenarios: Number of scenarios to generate (default: 24)

    Returns:
        List of scenario descriptions as strings
    """
    # Connect to MCP server to get available tools and resources
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get available tools
            tools_result = await session.list_tools()
            tools_info = []
            for tool in tools_result.tools:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                }
                tools_info.append(tool_info)

            # Get available resources
            try:
                resources_result = await session.list_resources()
                resources_info = []
                for resource in resources_result.resources:
                    resource_info = {
                        "uri": str(resource.uri),
                        "name": resource.name,
                        "description": resource.description,
                        "mimeType": resource.mimeType,
                    }
                    resources_info.append(resource_info)
            except Exception:
                # Some servers might not have resources
                resources_info = []

    # Prepare the prompt for o3
    tools_description = json.dumps(tools_info, indent=2)
    resources_description = (
        json.dumps(resources_info, indent=2)
        if resources_info
        else "No resources available"
    )

    prompt = f"""You are an expert at creating realistic scenarios for testing AI agents that interact with MCP (Model Context Protocol) servers.

Given the following available tools and resources from an MCP server, generate {num_scenarios} diverse, realistic scenarios that a user might want to accomplish using these tools.

AVAILABLE TOOLS:
{tools_description}

AVAILABLE RESOURCES:
{resources_description}

Requirements for scenarios:
1. Each scenario should be a clear, specific task that can be accomplished using the available tools
2. Scenarios should vary in complexity - some simple (1-2 tool calls), some complex (multiple tool calls)
3. Scenarios should cover different use cases and tool combinations
4. Each scenario should be realistic - something a real user might actually want to do
5. Be specific with company symbols, timeframes, indicators, etc. when relevant
6. Include scenarios for different user types (day traders, investors, analysts, researchers)

You must respond with a JSON array of exactly {num_scenarios} strings. Each string is a scenario description."""

    # Call OpenAI's model with structured JSON output
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Define the JSON schema for the response
    response_schema = {
        "type": "object",
        "properties": {
            "scenarios": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": num_scenarios,
                "maxItems": num_scenarios,
            }
        },
        "required": ["scenarios"],
        "additionalProperties": False,
    }

    # Try o3-mini first
    response = client.chat.completions.create(
        model="o3-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_completion_tokens=4000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "scenario_list", "schema": response_schema},
        },
    )

    # Parse the JSON response
    content = response.choices[0].message.content
    result = json.loads(content)

    # Extract scenarios from the response
    if "scenarios" in result:
        scenarios = result["scenarios"]
    else:
        # If the response is just an array
        scenarios = result if isinstance(result, list) else list(result.values())[0]

    # Validate we got exactly the right number
    if len(scenarios) != num_scenarios:
        raise ValueError(f"Expected {num_scenarios} scenarios, got {len(scenarios)}")

    return scenarios


async def test_scenario_generation():
    """Test the scenario generation function."""
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from servers.python.mcp_alphavantage.server_params import server_params

    print("Generating scenarios for MCP AlphaVantage server...")
    scenarios = await generate_scenarios(
        server_params, num_scenarios=3
    )  # Test with 3 scenarios for quick validation

    print(f"\nGenerated {len(scenarios)} scenarios:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_scenario_generation())
