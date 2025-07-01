# Copyright Sierra

import json
from litellm import Choices, acompletion
from litellm.types.utils import ModelResponse
from typing import List, Optional, Dict, Any

from art.utils.litellm import convert_litellm_choice_to_openai
from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME

class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        *args,
        **kwargs,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature

    async def llm_completion(self, messages: List[Dict[str, Any]]) -> ModelResponse:
        completion_obj = await acompletion(
            messages=messages,
            model=self.model,
            custom_llm_provider=self.provider,
            tools=self.tools_info,
            temperature=self.temperature,
        )
        assert isinstance(completion_obj, ModelResponse)
        return completion_obj
    
    async def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = await env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        final_prompt_tokens = 0
        avg_completion_tokens = 0
        max_completion_tokens = 0
        for curr_step_number in range(max_num_steps):
            res = await self.llm_completion(messages)
            final_prompt_tokens = res.usage.prompt_tokens # type: ignore
            avg_completion_tokens += res.usage.completion_tokens # type: ignore
            max_completion_tokens = max(max_completion_tokens, res.usage.completion_tokens) # type: ignore
            next_message = res.choices[0].message.model_dump() # type: ignore
            total_cost += (res._hidden_params.get("response_cost") or 0.0)
            action = message_to_action(next_message)
            env_response = await env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            if env_response.done:
                break
            if final_prompt_tokens > 20000:
                reward = -1
                break
            if res.choices[0].finish_reason == "length":
                reward = -1
                break
        info["total_steps"] = curr_step_number + 1
        info["avg_completion_tokens"] = avg_completion_tokens / info["total_steps"]
        info["max_completion_tokens"] = max_completion_tokens
        info["final_prompt_tokens"] = final_prompt_tokens
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )

class ToolCallingRLAgent(ToolCallingAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url")
        self.choices = []
    
    async def llm_completion(self, messages: List[Dict[str, Any]]):
        response = await acompletion(
            messages=messages,
            model=self.model,
            custom_llm_provider=self.provider,
            api_key=self.api_key,
            base_url=self.base_url,
            tools=self.tools_info,
            temperature=self.temperature,
            max_completion_tokens=1024,
            logprobs=True,
        )
        choice = response.choices[0] # type: ignore
        assert isinstance(choice, Choices)
        self.choices.append(convert_litellm_choice_to_openai(choice))
        return response
    
    def create_messages_and_choices(self, messages: List[Dict[str, Any]]):
        messages_and_choices = []
        choice_idx = 0
        for message in messages:
            if message["role"] == "assistant":
                messages_and_choices.append(self.choices[choice_idx])
                choice_idx += 1
            else:
                messages_and_choices.append(message)
        return messages_and_choices

def message_to_action(
    message: Dict[str, Any],
) -> Action:
    if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})
