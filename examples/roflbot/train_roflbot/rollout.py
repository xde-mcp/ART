import art
from typing import TypedDict, NotRequired
from art import Trajectory
from litellm import acompletion
import litellm
from litellm.caching.caching import LiteLLMCacheType, Cache
from litellm.types.utils import Choices, ModelResponse
from art.utils.litellm import convert_litellm_choice_to_openai
import textwrap
from tenacity import retry, stop_after_attempt
from project_types import PolicyConfig
from dataset import RedditJoke
import re
from score_joke import score_joke
import math
from langfuse.decorators import langfuse_context, observe

litellm.cache = Cache(type=LiteLLMCacheType.DISK)


class ProjectMetrics(TypedDict):
    rm_score: NotRequired[float]
    thinking_length: NotRequired[int]
    joke_length: NotRequired[int]
    matches_prefix: NotRequired[bool]
    no_content: NotRequired[bool]
    finish_reason_length: NotRequired[bool]
    successfully_parsed_thinking: NotRequired[bool]
    successfully_parsed_joke: NotRequired[bool]
    prompt_tokens: NotRequired[int]
    completion_tokens: NotRequired[int]


class ProjectTrajectory(Trajectory):
    metrics: ProjectMetrics = {}

    async def finish(
        self,
        model_name: str,
        policy_config: PolicyConfig,
    ) -> Trajectory:
        super().finish()
        self.reward = calculate_reward(policy_config, self.metrics)
        if policy_config.log_to_langfuse:
            langfuse_context.update_current_trace(
                metadata={
                    **self.metadata,
                    "model": model_name,
                }
            )
            langfuse_context.score_current_trace(
                name="reward",
                value=self.reward,
            )
            for k, v in self.metrics.items():
                langfuse_context.score_current_trace(
                    name=k,
                    value=v,  # type: ignore
                )
        return self


def calculate_reward(policy_config: PolicyConfig, metrics: ProjectMetrics) -> float:
    # Early exits for failure conditions
    if metrics.get("no_content"):
        return 0.0
    if not metrics.get("successfully_parsed_joke"):
        return 0.0

    # Handle thinking length penalty
    if metrics.get("thinking_length", 0) > policy_config.thinking_char_budget:
        # If budget is 0 or negative, any thinking results in 0 reward
        if policy_config.thinking_char_budget <= 0:
            return 0.0
        else:
            # Calculate ratio, ensure it's > 1
            # (since rubric.thinking_length > policy_config.thinking_char_budget)
            ratio = (
                metrics.get("thinking_length", 0) / policy_config.thinking_char_budget
            )
            # Apply log penalty: 1 / (1 + log(ratio))
            # This approaches 1 as ratio -> 1+, and decreases towards 0 as ratio increases.
            reward = 1.0 / (1.0 + math.log(ratio))
            return reward  # Return the penalized score directly

    # If thinking length is within budget:
    # Rule 3: Joke generated, thinking OK, but doesn't match prefix
    if not metrics.get("matches_prefix"):
        return 1.0
    # Rule 4: Joke generated, thinking OK, matches prefix
    else:
        # Add rm_score if available, otherwise base reward is 1.0
        base_reward = 1.0
        bonus = metrics.get("rm_score", 0.0)
        return base_reward + bonus


@retry(stop=stop_after_attempt(3))
async def determine_if_completion_matches_prefix(joke: str, prefix: str) -> bool:
    system_prompt = textwrap.dedent("""\
        You will be given a joke prefix and then the rest of the joke. Your job is to determine whether the joke body is a natural continuation of the joke prefix. It's ok if it takes things in an unexpected direction, but there should be a clear connection between the prefix and the joke body.
                                    
        Simply return True if the joke body is a natural continuation of the joke prefix, and False otherwise. Do not return anything except True or False. /no_think
        """)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Joke prefix: {prefix}\nJoke body: {joke}",
        },
    ]

    response = await acompletion(
        model="openrouter/qwen/qwen3-235b-a22b",
        messages=messages,
        temperature=0,
        caching=True,
        metadata={
            "existing_trace_id": langfuse_context.get_current_trace_id(),
            "parent_observation_id": langfuse_context.get_current_observation_id(),
        },
    )

    return response.choices[0].message.content.strip().lower().startswith("t")  # type: ignore


async def rollout(
    model: art.Model,
    scenario: RedditJoke,
) -> Trajectory:
    assert isinstance(model.config, PolicyConfig)
    if model.config.log_to_langfuse:
        return await rollout_with_langfuse(model, scenario)
    else:
        return await rollout_implementation(model, scenario)


@observe()
async def rollout_with_langfuse(
    model: art.Model,
    scenario: RedditJoke,
) -> Trajectory:
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]
    return await rollout_implementation(model, scenario)


async def rollout_implementation(
    model: art.Model,
    scenario: RedditJoke,
) -> Trajectory:
    traj = ProjectTrajectory(
        messages_and_choices=[],
        reward=0,
        metadata={"joke_id": scenario.id},
    )
    assert isinstance(model.config, PolicyConfig)

    system_prompt = textwrap.dedent(f"""
        You are a joke generator, and your goal is to generate hilarious jokes. As a starting point, you will be given a joke prefix. Your job is to generate a continuation of the joke, maximizing humor.
                                    
        You should take some time to think about the best way to complete the joke. You can think for up to {model.config.thinking_char_budget} characters. If you take longer than that to think, your response will be penalized. Do not repeat the joke prefix in your response, only return the continuation.

        You should respond in the following format:
        <think>some thoughts</think>
        <joke_continuation>the joke continuation</joke_continuation>
                                    
        Here are some examples of jokes:
        <joke_prefix>What is a pirate's favorite letter?</joke_prefix>
        <think>I should respond with a letter that pirates are known for.</think>
        <joke_continuation>ARRR!</joke_continuation>

        <joke_prefix>What do you call a belt made of watches?</joke_prefix>
        <think>What is a phrase that would make sense here? Maybe something about time? Should be related to belts too, hmm. Oh, I know! A waist of time.</think>
        <joke_continuation>A waist of time.</joke_continuation>
    """)

    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"<joke_prefix>{scenario.title}</joke_prefix>"},
    ]

    model_name = model.get_inference_name()
    if isinstance(model, art.TrainableModel):
        model_name = f"hosted_vllm/{model_name}"

    async with traj.track_duration("policy_completion"):
        llm_response = await acompletion(
            model=model_name,
            base_url=model.inference_base_url,
            messages=traj.messages(),
            caching=not isinstance(model, art.TrainableModel),
            api_key=model.inference_api_key,
            max_completion_tokens=model.config.max_tokens,
            metadata={
                "existing_trace_id": langfuse_context.get_current_trace_id(),
                "parent_observation_id": langfuse_context.get_current_observation_id(),
            },
            timeout=60 * 60 * 2,
        )
    assert isinstance(llm_response, ModelResponse)
    traj.metrics["prompt_tokens"] = llm_response.usage.prompt_tokens  # type: ignore
    traj.metrics["completion_tokens"] = llm_response.usage.completion_tokens  # type: ignore
    choice = llm_response.choices[0]  # type: ignore
    assert isinstance(choice, Choices)
    if (
        hasattr(choice.message, "reasoning_content")
        and choice.message.reasoning_content
    ):
        # LiteLLM "helpfully" extracts the reasoning content into a separate field, but our code expects it to be in the content field.
        choice.message.content = (
            f"<think>{choice.message.reasoning_content}</think>{choice.message.content}"
        )

    traj.messages_and_choices.append(convert_litellm_choice_to_openai(choice))

    content = choice.message.content

    if content is None:
        traj.metrics["no_content"] = True
        return await traj.finish(model.name, model.config)
    else:
        traj.metrics["no_content"] = False

    thinking = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    joke_continuation = re.search(
        r"<joke_continuation>(.*?)</joke_continuation>", content, re.DOTALL
    )

    if thinking is None:
        traj.metrics["successfully_parsed_thinking"] = False
    else:
        traj.metrics["successfully_parsed_thinking"] = True
        traj.metrics["thinking_length"] = len(thinking.group(1).strip())

    if joke_continuation is None:
        traj.metrics["successfully_parsed_joke"] = False
        return await traj.finish(model.name, model.config)
    else:
        traj.metrics["successfully_parsed_joke"] = True
        traj.metrics["joke_length"] = len(joke_continuation.group(1).strip())

    joke_content = joke_continuation.group(1).strip()

    async with traj.track_duration("determine_if_completion_matches_prefix"):
        traj.metrics["matches_prefix"] = await determine_if_completion_matches_prefix(
            joke_content, scenario.title
        )

    async with traj.track_duration("score_joke"):
        traj.metrics["rm_score"] = await score_joke(scenario.title, joke_content)

    return await traj.finish(model.name, model.config)


if __name__ == "__main__":
    from dataset import get_jokes
    from dotenv import load_dotenv
    import asyncio
    import yaml

    load_dotenv()

    test_joke = get_jokes(split="test", limit=1)[0]
    traj = asyncio.run(
        rollout(
            art.Model(
                name="qwen3-32b",
                project="roflbot",
                inference_model_name="openrouter/qwen/qwen3-32b",
                config=PolicyConfig(log_to_langfuse=False, max_tokens=20000),
            ),
            test_joke,
        )
    )
    # print(yaml.dump(traj.for_logging()))
