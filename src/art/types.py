from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
import pydantic
from typing import Literal

Message = ChatCompletionMessageParam
MessageOrChoice = Message | Choice
Messages = list[Message]
MessagesAndChoices = list[MessageOrChoice]
Tools = list[ChatCompletionToolParam]


class TrainConfig(pydantic.BaseModel):
    learning_rate: float = 5e-6
    beta: float = 0.0
    clip_epsilon_low: float = 0.2
    clip_epsilon_high: float = 0.28


Verbosity = Literal[0, 1, 2]
