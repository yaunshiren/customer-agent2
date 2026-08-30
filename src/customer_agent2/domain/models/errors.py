"""Provider-neutral model error taxonomy."""

from enum import StrEnum


class ModelErrorCode(StrEnum):
    """Stable categories that application code can handle without parsing messages."""

    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    ACCESS_DENIED = "access_denied"
    QUOTA_EXHAUSTED = "quota_exhausted"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    FIRST_PACKET_TIMEOUT = "first_packet_timeout"
    PROTOCOL = "protocol"
    UNAVAILABLE = "unavailable"


class ModelError(RuntimeError):
    """A sanitized model failure with a stable code and retry hint."""

    def __init__(
        self,
        code: ModelErrorCode,
        public_message: str,
        *,
        retryable: bool,
    ) -> None:
        normalized_message = public_message.strip()
        if not normalized_message:
            raise ValueError("public_message 不能为空")
        super().__init__(normalized_message)
        self.code = code
        self.public_message = normalized_message
        self.retryable = retryable
