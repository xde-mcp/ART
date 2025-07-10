import asyncio
import openai
import os

from instances import get_filtered_swe_smith_instances_df, as_instances_iter
from sandbox import new_sandbox, Sandbox


async def claude_rollout(
    problem_statement: str, sandbox: Sandbox, max_steps: int = 10
) -> str:
    """
    Roll out a Claude agent on a given problem statement in an isolated sandbox.

    Args:
        problem_statement: The problem statement to solve.
        sandbox: The sandbox to use for the rollout.

    Returns:
        The contents of the assistant's final response.
    """
    client = openai.AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1"
    )

    # Example prompt:
    response = await client.chat.completions.create(
        model="anthropic/claude-sonnet-4",
        messages=[
            {"role": "user", "content": "Hello, how are you?"},
        ],
    )
    assert response.choices[0].message.content
    return response.choices[0].message.content

    # TODO: Implement the Claude rollout function so that the agent can solve the problem statement in the sandbox.
    # 1. Provide a good system/user prompt with the problem statement.
    # 2. Create 3 OpenAI API-compatible tool definitions for the agent to use:
    #    - `bash`: Run a bash command in the sandbox.
    #    - `str_replace_editor`: Multi-function text editor (see `tools/edit_anthropic`).
    # 3. Create the agent loop that does the following until a stop condition is met:
    #    - Create a new chat completion with the chat history and tool definitions.
    #    - If the response has no tool call, early return the assistant's response.
    #    - Transform the chat completion choice object into a valid OpenAI API-compatible assistant message object and append it to the chat history.
    #       - Be sure to include the tool call in the assistant message object.
    #    - Parse and execute the response's tool call.
    #    - Append the sandbox's tool call response to the chat history.
    #    - If the sandbox raises an exception, append the exception to the chat history.
    #    - Raise an exception if uncompleted in `max_steps` steps.


async def test_claude_rollout(instance_idx: int) -> None:
    instance = next(
        get_filtered_swe_smith_instances_df()
        .pipe(lambda df: df.tail(-instance_idx) if instance_idx > 0 else df)
        .pipe(as_instances_iter)
    )

    # Calculate dynamic timeout based on number of tests
    # Formula: base_timeout + num_tests * per_test_time
    base_timeout = 120  # Base time for dependency installation
    per_test_time = 0.05  # Per-test time (reduced since most tests are fast)

    fail_to_pass_timeout = int(
        base_timeout + len(instance["FAIL_TO_PASS"]) * per_test_time
    )
    pass_to_pass_timeout = int(
        base_timeout + len(instance["PASS_TO_PASS"]) * per_test_time
    )

    async with new_sandbox(image=instance["image_name"], provider="daytona") as sandbox:
        failed, passed = await sandbox.run_tests(
            instance["FAIL_TO_PASS"], fail_to_pass_timeout
        )
        assert failed == 0
        assert passed == len(instance["FAIL_TO_PASS"])
        await sandbox.apply_patch(instance["patch"], 10)
        failed, passed = await sandbox.run_tests(
            instance["FAIL_TO_PASS"], fail_to_pass_timeout
        )
        assert failed == len(instance["FAIL_TO_PASS"])
        assert passed == 0
        failed, passed = await sandbox.run_tests(
            instance["PASS_TO_PASS"], pass_to_pass_timeout
        )
        assert failed == 0
        assert passed == len(instance["PASS_TO_PASS"])
        await claude_rollout(instance["problem_statement"], sandbox)
        f2p_failed, f2p_passed = await sandbox.run_tests(
            instance["FAIL_TO_PASS"], fail_to_pass_timeout
        )
        p2p_failed, p2p_passed = await sandbox.run_tests(
            instance["PASS_TO_PASS"], pass_to_pass_timeout
        )
        print(
            f"f2p_failed: {f2p_failed}/{len(instance['FAIL_TO_PASS'])}, f2p_passed: {f2p_passed}/{len(instance['FAIL_TO_PASS'])}"
        )
        print(
            f"p2p_failed: {p2p_failed}/{len(instance['PASS_TO_PASS'])}, p2p_passed: {p2p_passed}/{len(instance['PASS_TO_PASS'])}"
        )


if __name__ == "__main__":
    asyncio.run(test_claude_rollout(instance_idx=0))
