"""MCP agent rollout implementation.

This module provides a rollout function for running MCP agents with scenarios.
Based on the art-e rollout.py structure.
"""

import art
import json
import logging
from dataclasses import dataclass
import os
import asyncio
from dotenv import load_dotenv
import weave

from mcp_server import McpServer, AlphaMcpServer

load_dotenv()

logging.getLogger("weave.trace.op").setLevel(logging.WARNING)


@dataclass
class McpScenario:
    """A scenario for MCP agent evaluation."""

    task_description: str
    mcp_server: McpServer
    max_turns: int = 10


@weave.op()
async def rollout(
    model: art.Model,
    scenario: McpScenario,
    debug: bool = False,
) -> art.Trajectory:
    """Run an MCP agent rollout with a pre-started MCP server.

    Args:
        model: The ART model to use for the agent
        scenario: The MCP scenario to run (must include mcp_server)

    Returns:
        Trajectory containing the results of the rollout
    """
    traj = art.Trajectory(
        messages_and_choices=[],
        reward=0,
        metadata={"task": scenario.task_description},
        metrics={
            "task_completed": False,
        },
        scenario=scenario,
    )

    # Initialize system prompt
    system_prompt = """You are an MCP (Model Context Protocol) agent.\n\nYou have access to MCP tools through the server. Use them to complete your task.\n\nWhen you believe you have completed the task, call the 'complete_task' function with a summary of what you accomplished."""

    try:
        try:
            # Get available tools from the server
            tool_schemas = await scenario.mcp_server.get_tools()
        except Exception as e:
            print(f"Error getting tools from MCP server: {e}")
            raise e

        if debug:
            available_tools = [tool["function"]["name"] for tool in tool_schemas]
            print(f"Available MCP tools: {available_tools}")

        # Add completion tool schema
        tool_schemas.append(
            {
                "type": "function",
                "function": {
                    "name": "complete_task",
                    "description": "Complete the task with a summary",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string",
                                "description": "Summary of accomplishments",
                            }
                        },
                        "required": ["summary"],
                    },
                },
            }
        )

        traj.tools = tool_schemas

        # Initialize conversation
        traj.messages_and_choices = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Please complete this task: {scenario.task_description}",
            },
        ]

        if debug:
            print(traj.messages())

        num_turns = 0

        # Main interaction loop
        while num_turns < scenario.max_turns:
            num_turns += 1

            try:
                # Get LLM response
                async with traj.track_duration("llm_completion"):
                    client = model.openai_client()

                    response = await client.chat.completions.create(
                        model=model.inference_model_name
                        if model.inference_model_name
                        else model.name,
                        messages=traj.messages(),
                        tools=tool_schemas,
                        max_tokens=getattr(model.config, "max_tokens", 1000),
                    )

                choice = response.choices[0]

                if debug:
                    print(f"Choice: {choice.message}")

                traj.messages_and_choices.append(choice)

                # Handle tool calls
                if choice.message.tool_calls:
                    for tool_call in choice.message.tool_calls:
                        try:
                            tool_args = json.loads(tool_call.function.arguments)

                            if tool_call.function.name == "complete_task":
                                traj.metrics["task_completed"] = True
                                traj.log(
                                    f"Task completion attempted with summary: {tool_args['summary']}"
                                )
                                break
                            else:
                                # Call MCP tool through server
                                result = await scenario.mcp_server.apply_tool(
                                    tool_call.function.name, tool_args
                                )

                            # Add tool response
                            traj.messages_and_choices.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": str(result),
                                }
                            )

                            if debug:
                                print(f"Tool call result: {result}")

                        except Exception as e:
                            traj.log(f"Tool call error: {e}")

                            # Add error response
                            traj.messages_and_choices.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": f"Error: {str(e)}",
                                }
                            )
                else:
                    # No tool calls, just continue conversation
                    break

            except Exception as e:
                traj.log(f"Error in turn {num_turns}: {e}")
                break

    except Exception as e:
        traj.log(f"MCP server error: {e}")

    return traj.finish()


async def test_rollout():
    mcp_server = AlphaMcpServer(api_key=os.getenv("ALPHAVANTAGE_API_KEY"))

    # Start the server explicitly
    await mcp_server.start()

    try:
        model = art.Model(
            name="gpt-4o-mini",
            project="mcp-agent-training",
            inference_model_name="gpt-4o-mini",
            inference_api_key=os.getenv("OPENAI_API_KEY"),
            inference_base_url="https://api.openai.com/v1",
        )
        scenario = McpScenario(
            task_description="Get the current stock price for AAPL",
            mcp_server=mcp_server,
        )
        traj = await rollout(model, scenario, debug=True)
        print(traj)
    finally:
        # Clean up the server session
        await mcp_server.stop()


async def main():
    """Run test scenario."""
    print("=== Testing AlphaMcpServer ===")
    await test_rollout()


if __name__ == "__main__":
    asyncio.run(main())
