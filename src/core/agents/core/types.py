from typing import Literal, Optional, TypedDict

from pydantic import BaseModel


class AgentMessage(TypedDict):
    role: Literal["user", "assistant", "system", "tool"]
    content: str


class AgentOutput(TypedDict):
    messages: list
    structured_response: Optional[BaseModel]


class MessagesDict(TypedDict):
    messages: list[AgentMessage]
