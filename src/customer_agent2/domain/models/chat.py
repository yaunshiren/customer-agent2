"""Provider-neutral chat model request and response contracts."""

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class ChatRole(StrEnum):
    """Message roles supported by the P0 chat pipeline."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One validated text message sent to a chat model."""

    role: ChatRole
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("ChatMessage.content 不能为空")


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """Provider-neutral chat generation request."""

    messages: tuple[ChatMessage, ...]
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning_enabled: bool | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("ChatRequest.messages 不能为空")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ValueError("ChatRequest.temperature 必须在 0 到 2 之间")
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise ValueError("ChatRequest.max_output_tokens 必须大于 0")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Input and output token counts reported by a model provider."""

    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("TokenUsage 不能包含负数")

    @property
    def total_tokens(self) -> int:
        """Return the combined token count."""
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Complete chat generation result."""

    model_id: str
    content: str
    finish_reason: str
    reasoning_content: str | None = None
    usage: TokenUsage | None = None

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("ChatResult.model_id 不能为空")
        if not self.finish_reason.strip():
            raise ValueError("ChatResult.finish_reason 不能为空")


@dataclass(frozen=True, slots=True)
class ChatStreamChunk:
    """One provider-neutral incremental chat event."""

    delta: str = ""
    reasoning_delta: str = ""
    finish_reason: str | None = None
    usage: TokenUsage | None = None

    def __post_init__(self) -> None:
        if not (self.delta or self.reasoning_delta or self.finish_reason or self.usage):
            raise ValueError("ChatStreamChunk 必须包含增量内容或完成信息")


class ChatModel(Protocol):
    """Chat capability required by application use cases."""

    @property
    def model_id(self) -> str: ...

    async def complete(self, request: ChatRequest) -> ChatResult: ...

    def stream(self, request: ChatRequest) -> AsyncGenerator[ChatStreamChunk, None]: ...
