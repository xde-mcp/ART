import art
from typing import List, Any
from art_e.data.types_enron import SyntheticQuery
from art import Trajectory
from litellm import acompletion
import litellm
from art_e.email_search_tools import search_emails, read_email
from langchain_core.utils.function_calling import convert_to_openai_tool
from litellm.caching.caching import LiteLLMCacheType, Cache
import json
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from litellm.types.utils import Choices, ModelResponse, Message
from dataclasses import asdict
from art.utils.litellm import convert_litellm_choice_to_openai
from dataclasses import dataclass
from langfuse.decorators import langfuse_context, observe  # type: ignore
from art_e.project_types import PolicyConfig
import textwrap
from tenacity import retry, stop_after_attempt
from panza import limit_concurrency

litellm.cache = Cache(type=LiteLLMCacheType.DISK)

# We remove the inbox parameter before describing the tool for OpenAI because we'll set that value explicitly based on the user we're running on behalf of.
search_tool = convert_to_openai_tool(search_emails)
del search_tool["function"]["parameters"]["properties"]["inbox"]
search_tool["function"]["parameters"]["required"].remove("inbox")


@dataclass
class FinalRubric:
    answer_correct: bool = False
    sources_correct: bool = False
    num_turns: int = 0
    attempted_answer: bool = False
    ever_found_right_email: bool = False
    ever_read_right_email: bool = False
    cant_parse_tool_call: bool = False
    bad_tool_call_name: bool = False
    bad_tool_call_args: bool = False
    ran_out_of_turns: bool = False
    returned_i_dont_know: bool = False
    num_sources: int = 0
    ever_tried_to_read_invalid_email: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0


def calculate_reward(
    policy_config: PolicyConfig, rubric: FinalRubric, traj: Trajectory
) -> float:
    # As an ablation, let's try the simplest possible reward function: just give
    # 1 point for a correct answer, and 0 for anything else. Otherwise, we'll do something
    # more complex.
    if policy_config.stupid_simple_reward_fn:
        return float(rubric.answer_correct)

    # Note: make sure all possible partial rewards always sum to less than 0.5.
    partial_rewards = 0
    partial_rewards += 0.1 if rubric.ever_found_right_email else 0
    partial_rewards += 0.1 if rubric.ever_read_right_email else 0
    partial_rewards += 0.1 if not rubric.ever_tried_to_read_invalid_email else 0
    partial_rewards += 0.1 if rubric.sources_correct else 0

    # Formatting error: reward will be -2 to -1
    if rubric.cant_parse_tool_call:
        return -2 + partial_rewards

    if rubric.bad_tool_call_name:
        return -1.9 + partial_rewards

    if rubric.bad_tool_call_args:
        return -1.8 + partial_rewards

    # No formatting error, but wrong answer: reward will be -1 to 0
    if rubric.attempted_answer and not rubric.answer_correct:
        return -1 + partial_rewards

    # Returned no answer at all: reward will be 0 to 1
    if rubric.returned_i_dont_know or rubric.ran_out_of_turns:
        return 0 + partial_rewards

    # Answer is correct: reward will be 1 to 2
    if rubric.answer_correct:
        # Partial credit calculation is different for correct answers.

        reward = 1
        reward += 0.3 if rubric.sources_correct else 0

        # Extra credit for not including extra sources.
        reward += 0.1 / rubric.num_sources if rubric.num_sources > 0 else 0

        # Extra credit for being faster (taking fewer turns).
        reward += 0.1 * (1 - rubric.num_turns / policy_config.max_turns)
        return reward

    traj.logs.append(f"Rubric: {rubric}")
    traj.logs.append("Rubric not handled properly")
    raise ValueError("Rubric is not handled properly")


def return_final_answer(answer: str, sources: List[str] | None) -> str:
    """
    This function is used to return the final answer to the user's query.
    It should be called with the answer and the sources. If you cannot find the answer, you should return "I don't know" with an empty list of sources.

    Args:
        answer: (str) the answer to the user's query. If you cannot find the answer, you should return "I don't know" with an empty list of sources.
        sources: (list[str]) a list of message ids that are relevant to the query. Usually there will be only one.

    Returns:
        (str) the final answer to the user's query
    """
    ...


def tool_response(response: Any, message: Message) -> ChatCompletionMessageParam:
    """Generate a response for a tool call.

    Args:
        response: The response from the tool
        message: The message that's being responded to

    Returns:
        A message that can be added to the conversation
    """
    if message.tool_calls:
        return {
            "role": "tool",
            "tool_call_id": message.tool_calls[0].id,
            "content": json.dumps(response),
        }
    else:
        return {
            "role": "user",
            "content": json.dumps(response),
        }


tools: list[ChatCompletionToolParam] = [
    search_tool,
    convert_to_openai_tool(read_email),
    convert_to_openai_tool(return_final_answer),
]  # type: ignore


@retry(stop=stop_after_attempt(3))
async def determine_if_answer_is_correct(answer: str, query: SyntheticQuery) -> bool:
    system_prompt = "You will be given an question and two different answers to the question, the correct answer and the answer given by an AI. Your job is to determine if the answer given by the AI is correct. Return True if the answer is semantically similar to the correct answer, and False otherwise. Return only the word True or False, no other text."

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Question: {query.question}\nCorrect answer: {query.answer}\nAI answer: {answer}",
        },
    ]

    response = await acompletion(
        model="gemini/gemini-2.0-flash",
        messages=messages,
        temperature=0,
        caching=True,
        max_tokens=2,
        metadata={
            "generation_name": "determine_if_answer_is_correct",
            "existing_trace_id": langfuse_context.get_current_trace_id(),
            "parent_observation_id": langfuse_context.get_current_observation_id(),
        },
    )

    return response.choices[0].message.content.strip().lower().startswith("t")  # type: ignore


@observe(name="art_e_rollout")
@limit_concurrency(100)
@retry(stop=stop_after_attempt(3))
async def rollout(
    model: art.Model,
    scenario: SyntheticQuery,
) -> Trajectory:
    rubric = FinalRubric()
    traj = Trajectory(
        messages_and_choices=[],
        reward=0,
        metadata={"email_inbox": scenario.inbox_address, "scenario_id": scenario.id},
    )
    assert isinstance(model.config, PolicyConfig)

    if model.config.log_to_langfuse:
        litellm.success_callback = ["langfuse"]
        litellm.failure_callback = ["langfuse"]
        langfuse_context.update_current_trace(
            metadata={"email_inbox": scenario.inbox_address, "scenario_id": scenario.id}
        )

    system_prompt = textwrap.dedent(f"""\
        You are an email search agent. You are given a user query and a list of tools you can use to search the user's email. Use the tools to search the user's emails and find the answer to the user's query. You may take up to {model.config.max_turns} turns to find the answer, so if your first seach doesn't find the answer, you can try with different keywords.

        User's email address is {scenario.inbox_address}
        Today's date is {scenario.query_date}
    """)

    if model.config.use_tools:
        traj.tools = tools
    else:
        system_prompt += textwrap.dedent(f"""\
            
            Here are the tools you can use:
            {tools}
            
            Respond with a valid JSON object with the following fields:
            - tool_name: (str) the name of the tool to use
            - tool_args: (JSON) the arguments to pass to the tool

            For example, to read a specific email, you should respond with:
            {{
                "tool_name": "read_email",
                "tool_args": {{
                    "message_id": "<12635597.1075855702772.JavaMail.evans@thyme>"
                }}
            }}
        """)

    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario.question},
    ]
    llm_response: ModelResponse | None = None
    final_answer = None

    while True:
        rubric.num_turns += 1

        if rubric.num_turns > model.config.max_turns:
            rubric.ran_out_of_turns = True
            break

        model_name = model.get_inference_name()
        if isinstance(model, art.TrainableModel):
            model_name = f"hosted_vllm/{model_name}"

        async with traj.track_duration("llm_completion"):
            llm_response = await acompletion(
                model=model_name,
                base_url=model.inference_base_url,
                messages=traj.messages(),
                caching=not model.trainable,
                api_key=model.inference_api_key,
                max_completion_tokens=model.config.max_tokens,
                tools=tools if model.config.use_tools else None,
                tool_choice="required"
                if model.config.use_tools and not model.trainable
                else None,
                metadata={
                    "generation_name": "agent_turn",
                    "existing_trace_id": langfuse_context.get_current_trace_id(),
                    "parent_observation_id": langfuse_context.get_current_observation_id(),
                },
            )  # type: ignore

        assert isinstance(llm_response, ModelResponse)
        rubric.prompt_tokens += llm_response.usage.prompt_tokens  # type: ignore
        rubric.completion_tokens += llm_response.usage.completion_tokens  # type: ignore
        choice = llm_response.choices[0]  # type: ignore
        assert isinstance(choice, Choices)

        # Our rollout is only set up to handle one tool call at a time, so just ignore any parallel tool calls.
        if choice.message.tool_calls is not None and len(choice.message.tool_calls) > 1:
            choice.message.tool_calls = choice.message.tool_calls[:1]
        traj.messages_and_choices.append(convert_litellm_choice_to_openai(choice))

        if model.config.use_tools:
            tool_call = (
                choice.message.tool_calls[0].get("function")
                if choice.message.tool_calls
                else None
            )
            if tool_call is None:
                rubric.bad_tool_call_args = True
                break
            tool_name = tool_call["name"]
            try:
                tool_args = json.loads(tool_call["arguments"])
                assert isinstance(tool_args, dict)
            except Exception as e:
                rubric.bad_tool_call_args = True
                break
        else:
            raw_content = choice.message.content
            if raw_content is None:
                rubric.cant_parse_tool_call = True
                break
            start_index = raw_content.find("{")
            end_index = raw_content.rfind("}")
            if not (start_index != -1 and end_index != -1 and start_index < end_index):
                rubric.cant_parse_tool_call = True
                break
            json_str = raw_content[start_index : end_index + 1]

            try:
                tool_call = json.loads(json_str)
            except Exception as e:
                traj.logs.append(f"Error parsing tool call: {e}")
                rubric.cant_parse_tool_call = True
                break

            if "tool_args" not in tool_call:
                rubric.bad_tool_call_args = True
                traj.logs.append(f"Tool call missing tool_args: {tool_call}")
                break
            tool_name = tool_call.get("tool_name")
            tool_args = tool_call.get("tool_args")

        match tool_name:
            case "search_emails":
                try:
                    search_results = search_emails(
                        **tool_args,
                        inbox=scenario.inbox_address,
                    )
                    traj.messages_and_choices.append(
                        tool_response(
                            [asdict(r) for r in search_results],
                            choice.message,
                        )
                    )
                    for r in search_results:
                        if r.message_id == scenario.message_ids[0]:
                            rubric.ever_found_right_email = True
                except Exception as e:
                    rubric.bad_tool_call_args = True
                    traj.logs.append(f"Error searching emails: {e}")
                    break
            case "read_email":
                message_id_to_read = tool_args.get("message_id")
                if not isinstance(message_id_to_read, str):
                    rubric.bad_tool_call_args = True
                    break
                if message_id_to_read == scenario.message_ids[0]:
                    rubric.ever_read_right_email = True
                email_content = read_email(message_id_to_read)
                if email_content is None:
                    traj.messages_and_choices.append(
                        tool_response({"error": "Email not found"}, choice.message)
                    )
                    rubric.ever_tried_to_read_invalid_email = True
                else:
                    traj.messages_and_choices.append(
                        tool_response(email_content.model_dump(), choice.message)
                    )
            case "return_final_answer":
                final_answer = tool_args.get("answer")
                final_sources = tool_args.get("sources")

                if (
                    final_answer is None
                    or final_sources is None
                    or not isinstance(final_sources, list)
                ):
                    rubric.bad_tool_call_args = True
                    break

                if final_answer == "I don't know":
                    rubric.returned_i_dont_know = True
                else:
                    rubric.attempted_answer = True
                    async with traj.track_duration("determine_if_answer_is_correct"):
                        rubric.answer_correct = await determine_if_answer_is_correct(
                            final_answer, scenario
                        )
                    rubric.sources_correct = scenario.message_ids[0] in final_sources
                break
            case _:
                rubric.bad_tool_call_name = True
                break

    traj.finish()
    traj.reward = calculate_reward(model.config, rubric, traj)
    traj.metrics = asdict(rubric)

    if model.config.log_to_langfuse:
        langfuse_context.update_current_trace(
            metadata={**traj.metadata, "model": model.name}
        )
        langfuse_context.score_current_trace(name="reward", value=traj.reward)
        for k, v in traj.metrics.items():
            langfuse_context.score_current_trace(name=k, value=v)

    return traj.finish()


if __name__ == "__main__":
    from art_e.data.query_iterators import load_synthetic_queries
    from dotenv import load_dotenv
    import asyncio
    import yaml

    load_dotenv()

    traj = asyncio.run(
        rollout(
            art.Model(
                name="gpt-4o",
                project="email_agent",
                inference_model_name="openai/gpt-4.1",
                config=PolicyConfig(
                    log_to_langfuse=True,
                    use_tools=True,
                ),
            ),
            load_synthetic_queries(split="test", limit=1)[0],
        )
    )
    print(yaml.dump(traj.for_logging()))
