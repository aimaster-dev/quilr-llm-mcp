"""Incremental Server-Sent Events parser and encoder.

Bytes from the network can be split anywhere: inside a multi-byte UTF-8 character, between the
``\\r`` and ``\\n`` of a line ending, or in the middle of a field name. The parser keeps only the
current partial line and the fields of the event being built, and yields each event as soon as its
terminating blank line arrives.

It follows the WHATWG event-stream rules (``\\r\\n``, ``\\n`` and ``\\r`` line endings, one optional
space after the colon, multi-line ``data``, ``:`` comments) with two proxy-friendly changes:
comments are returned as events so keep-alives can be forwarded, and an event left unterminated at
EOF is dispatched instead of dropped.
"""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass, field

_EOL = re.compile(r"\r\n|\r|\n")


class SSEError(ValueError):
    """The upstream event stream is malformed or exceeds the configured limits."""


@dataclass(slots=True)
class SSEEvent:
    data: str | None = None  # None when the event had no data field
    event: str | None = None
    id: str | None = None
    retry: str | None = None
    comments: list[str] = field(default_factory=list)

    def encode(self) -> bytes:
        lines = [f": {c}" if c else ":" for c in self.comments]
        if self.event is not None:
            lines.append(f"event: {self.event}")
        if self.id is not None:
            lines.append(f"id: {self.id}")
        if self.retry is not None:
            lines.append(f"retry: {self.retry}")
        if self.data is not None:
            lines.extend(f"data: {line}" for line in self.data.split("\n"))
        return ("\n".join(lines) + "\n\n").encode()


class SSEParser:
    def __init__(self, max_event_size: int = 1 << 20) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._max = max_event_size  # characters, after UTF-8 decoding
        self._line = ""  # partial line carried over from the previous chunk
        self._skip_lf = False  # previous chunk ended with "\r", so a leading "\n" is part of it
        self._size = 0
        self._reset()

    def _reset(self) -> None:
        self._data: list[str] | None = None
        self._event: str | None = None
        self._id: str | None = None
        self._retry: str | None = None
        self._comments: list[str] = []
        self._size = 0

    def feed(self, chunk: bytes) -> list[SSEEvent]:
        return self._process(self._decoder.decode(chunk))

    def close(self) -> list[SSEEvent]:
        """Signal EOF: decode any trailing bytes and dispatch an unterminated event."""
        events = self._process(self._decoder.decode(b"", final=True))
        if self._line:
            self._handle_line(self._line, events)
            self._line = ""
        self._dispatch(events)
        return events

    def _process(self, text: str) -> list[SSEEvent]:
        events: list[SSEEvent] = []
        if not text:
            return events
        if self._skip_lf:
            self._skip_lf = False
            if text[0] == "\n":
                text = text[1:]
        buf = self._line + text if self._line else text
        start = 0
        for m in _EOL.finditer(buf):
            self._handle_line(buf[start : m.start()], events)
            start = m.end()
        # A lone "\r" at the very end may be the first half of a "\r\n" split across chunks.
        self._skip_lf = start == len(buf) and buf.endswith("\r")
        self._line = buf[start:]
        if len(self._line) + self._size > self._max:
            raise SSEError("event exceeds the maximum size")
        return events

    def _handle_line(self, line: str, events: list[SSEEvent]) -> None:
        if not line:
            self._dispatch(events)
            return
        self._size += len(line)
        if self._size > self._max:
            raise SSEError("event exceeds the maximum size")
        name, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if not name:
            self._comments.append(value)
        elif name == "data":
            if self._data is None:
                self._data = [value]
            else:
                self._data.append(value)
        elif name == "event":
            self._event = value
        elif name == "id" and "\0" not in value:
            self._id = value
        elif name == "retry" and value.isascii() and value.isdigit():
            self._retry = value
        # Anything else (unknown fields, an id with NUL, a non-numeric retry) is ignored per spec.

    def _dispatch(self, events: list[SSEEvent]) -> None:
        if self._data is None and self._event is None and self._id is None and self._retry is None:
            if self._comments:
                events.append(SSEEvent(comments=self._comments))
            self._reset()
            return
        events.append(
            SSEEvent(
                data=None if self._data is None else "\n".join(self._data),
                event=self._event,
                id=self._id,
                retry=self._retry,
                comments=self._comments,
            )
        )
        self._reset()
