"""HTTP error handling: request ids, typed error responses and the catch-all.

Layering inside a Starlette app (outermost first)::

    ServerErrorMiddleware (Starlette)  - last resort, normally never reached
    RequestContextMiddleware           - request id, X-Request-Id header, catch-all -> 500
    ExceptionMiddleware (Starlette)    - GatewayError / HTTPException -> standardized payload
    routes
"""

from __future__ import annotations

import logging
import uuid
from http import HTTPStatus

from starlette.datastructures import MutableHeaders
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from llm_router.errors import (
    GatewayError,
    InternalError,
    InvalidRequestError,
    MethodNotAllowedError,
    NotFoundError,
    RequestTooLargeError,
    UpstreamError,
)
from llm_router.logging_config import request_id_var

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-Id"


def new_request_id() -> str:
    # Always generated here, never taken from the client: a client-chosen id could collide with
    # another request's or smuggle text into the logs.
    return f"req_{uuid.uuid4().hex}"


def error_response(error: GatewayError, request_id: str) -> JSONResponse:
    headers = {**error.headers, "Cache-Control": "no-store"}
    return JSONResponse(error.payload(request_id), status_code=error.status, headers=headers)


class RequestContextMiddleware:
    """Assigns the request id and turns any unhandled exception into a generic 500 payload.

    A client that disconnects mid-body (Starlette's ``ClientDisconnect``) is logged at INFO and
    gets no response at all, rather than an ERROR line and a 500 written to a closed socket.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = new_request_id()
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)
        response_started = False

        async def send_with_request_id(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except ClientDisconnect:
            # The client hung up before sending the whole body. That is not a server error, and
            # there is nobody left to send a response to.
            logger.info(
                "client disconnected before sending the whole request body: %s %s",
                scope.get("method"),
                scope.get("path"),
            )
        except Exception:
            # The traceback goes to the logs only; the client gets a fixed message and the id.
            logger.exception("unhandled error serving %s %s", scope.get("method"), scope.get("path"))
            if response_started:
                raise  # too late for a payload: let the server abort the connection
            await error_response(InternalError(), request_id)(scope, receive, send_with_request_id)
        finally:
            request_id_var.reset(token)


def current_request_id(request: Request) -> str:
    return request.scope.get("state", {}).get("request_id") or request_id_var.get()


def _describe_failures(error: UpstreamError) -> str:
    return ", ".join(
        f"{f.upstream}:{f.kind}" + (f"({f.status})" if f.status is not None else "") for f in error.failures
    )


async def gateway_error_handler(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, GatewayError)
    level = logging.WARNING if exc.status >= 500 else logging.INFO
    context = f" attempts=[{_describe_failures(exc)}]" if isinstance(exc, UpstreamError) else ""
    if exc.log_detail:
        context += f" detail={exc.log_detail}"
    logger.log(
        level,
        "%s %s -> %d %s%s",
        request.method,
        request.url.path,
        exc.status,
        exc.code,
        context,
        exc_info=exc if isinstance(exc, InternalError) and exc.__cause__ else None,
    )
    return error_response(exc, current_request_id(request))


def gateway_error_from_http_exception(exc: HTTPException) -> GatewayError:
    """Map Starlette's routing errors (404, 405, ...) onto the gateway taxonomy."""
    if exc.status_code == 404:
        return NotFoundError()
    if exc.status_code == 405:
        allow = (exc.headers or {}).get("Allow")
        return MethodNotAllowedError(headers={"Allow": allow} if allow else None)
    if exc.status_code == 413:
        return RequestTooLargeError()
    if exc.status_code >= 500:
        return InternalError()
    # Any other client error: a generic payload with the standard reason phrase. exc.detail is not
    # used because it is free text.
    phrase = HTTPStatus(exc.status_code).phrase
    return InvalidRequestError(f"{phrase}.", code=f"http_{exc.status_code}", status=exc.status_code)


async def http_exception_handler(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, HTTPException)
    return await gateway_error_handler(request, gateway_error_from_http_exception(exc))


EXCEPTION_HANDLERS = {GatewayError: gateway_error_handler, HTTPException: http_exception_handler}
MIDDLEWARE = [Middleware(RequestContextMiddleware)]
