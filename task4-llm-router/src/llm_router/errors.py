"""Gateway error taxonomy.

Every error that leaves the gateway has the same shape::

    {"error": {"type": "...", "code": "...", "message": "...", "request_id": "req_..."}}

* ``type`` is a coarse category clients can branch on, ``code`` is the specific reason.
* ``message`` is written by the gateway, never copied from an upstream body, an exception or user
  input. It cannot leak stack traces, hostnames, provider names or credentials.
* ``request_id`` correlates the response with the server logs, which hold the full details
  (``log_detail``, upstream status and body snippets, tracebacks).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar


class GatewayError(Exception):
    status: int = 500
    type: ClassVar[str] = "internal_error"
    default_code: ClassVar[str] = "internal_error"
    default_message: ClassVar[str] = "The gateway encountered an unexpected error. Please retry later."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        status: int | None = None,
        headers: Mapping[str, str] | None = None,
        log_detail: str | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.code = code or self.default_code
        if status is not None:
            self.status = status
        self.headers: dict[str, str] = dict(headers or {})
        #: Server-side context for the logs. Never sent to the client.
        self.log_detail = log_detail
        super().__init__(self.message)

    def payload(self, request_id: str) -> dict[str, Any]:
        return {
            "error": {
                "type": self.type,
                "code": self.code,
                "message": self.message,
                "request_id": request_id,
            }
        }


class InternalError(GatewayError):
    pass


class InvalidRequestError(GatewayError):
    status = 400
    type = "invalid_request_error"
    default_code = "invalid_request"
    default_message = "The request is invalid."


class AuthenticationError(GatewayError):
    status = 401
    type = "authentication_error"
    default_code = "invalid_api_key"
    default_message = "A valid API key is required: send 'Authorization: Bearer <key>'."

    def __init__(self, message: str | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.headers.setdefault("WWW-Authenticate", "Bearer")


class NotFoundError(GatewayError):
    status = 404
    type = "not_found_error"
    default_code = "not_found"
    default_message = "The requested endpoint does not exist."


class MethodNotAllowedError(GatewayError):
    status = 405
    type = "invalid_request_error"
    default_code = "method_not_allowed"
    default_message = "This HTTP method is not allowed for the endpoint."


class RequestTooLargeError(GatewayError):
    """The request can never be served as sent (body too big, or cost above the whole window)."""

    status = 413
    type = "invalid_request_error"
    default_code = "request_too_large"
    default_message = "The request is too large."


class RateLimitError(GatewayError):
    status = 429
    type = "rate_limit_error"
    default_code = "tenant_token_limit_exceeded"

    def __init__(self, retry_after_seconds: int, message: str | None = None, **kwargs: Any) -> None:
        message = message or (
            "Token rate limit exceeded for this API key. "
            f"Retry after {retry_after_seconds} second{'s' if retry_after_seconds != 1 else ''}."
        )
        super().__init__(message, **kwargs)
        self.retry_after_seconds = retry_after_seconds
        self.headers["Retry-After"] = str(retry_after_seconds)


class ServiceUnavailableError(GatewayError):
    status = 503
    type = "service_unavailable_error"
    default_code = "service_unavailable"
    default_message = "The gateway is temporarily unable to serve requests. Please retry later."


class UpstreamError(GatewayError):
    """No model provider produced a usable completion.

    ``failures`` holds the router's per-attempt records (upstream label, failure kind, status,
    body snippet) for the logs only.
    """

    status = 502
    type = "upstream_error"
    default_code = "upstream_bad_response"
    default_message = "The model provider returned an invalid response."

    def __init__(self, message: str | None = None, *, failures: Sequence[Any] = (), **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.failures = tuple(failures)


class UpstreamRejectedError(UpstreamError):
    """The provider refused the request itself (400/413/422): retrying elsewhere would not help."""

    status = 400
    type = "invalid_request_error"
    default_code = "upstream_rejected_request"
    default_message = "The model provider rejected the request. Check the messages and parameters."


class UpstreamUnavailableError(UpstreamError):
    status = 503
    default_code = "upstream_unavailable"
    default_message = "No model provider is available to serve this request right now. Please retry later."


class UpstreamTimeoutError(UpstreamError):
    status = 504
    type = "timeout_error"
    default_code = "upstream_timeout"
    default_message = "The model provider did not respond in time. Please retry."
