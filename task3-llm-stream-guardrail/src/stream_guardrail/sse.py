"""Incremental Server-Sent Events parser and encoder.

Bytes from the network can be split anywhere: inside a multi-byte UTF-8 character, between the
``\\r`` and ``\\n`` of a line ending, or in the middle of a field name. The parser keeps only the
current partial line and the fields of the event being built, and yields each event as soon as its
terminating blank line arrives.

Each read is scanned once: only the new text is searched for line endings, and a line that spans
reads is accumulated and joined once, when its end arrives. So a long line delivered in small
reads costs linear time, not quadratic.

It follows the WHATWG event-stream rules (``\\r\\n``, ``\\n`` and ``\\r`` line endings, one leading
byte order mark stripped, one optional space after the colon, multi-line ``data``, ``:`` comments)
with two proxy-friendly changes: comments are returned as events so keep-alives can be forwarded,
and an event left unterminated at EOF is dispatched instead of dropped.
"""

from __future__ import annotations

import codecs
import io
import re
from dataclasses import dataclass, field

_EOL = re.compile(r"\r\n|\r|\n")
_BOM = chr(0xFEFF)  # U+FEFF, the byte order mark


class SSEError(ValueError):
    """The upstream event stream is malformed or exceeds the configured limits.

    ``events`` holds the complete events parsed from the same input before the problem, so the
    caller can still deliver them.
    """

    def __init__(self, message: str, events: list[SSEEvent] | None = None) -> None:
        super().__init__(message)
        self.events: list[SSEEvent] = events or []


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
        self._partial: io.StringIO | None = None  # the current line, when it spans reads
        self._partial_size = 0
        self._skip_lf = False  # previous read ended with "\r", so a leading "\n" is part of it
        self._started = False  # the first character has been seen (for the BOM)
        self._failed = False
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
        """Parse one read. Raises ``SSEError`` (with the events completed before it) on bad input."""
        return self._run(self._decoder.decode(chunk), final=False)

    def close(self) -> list[SSEEvent]:
        """Signal EOF: decode any trailing bytes and dispatch an unterminated event."""
        return self._run(self._decoder.decode(b"", final=True), final=True)

    def _run(self, text: str, final: bool) -> list[SSEEvent]:
        if self._failed:
            raise SSEError("the event stream already failed")
        events: list[SSEEvent] = []
        try:
            self._process(text, events)
            if final:
                if self._partial is not None:
                    self._handle_line(self._take_line(""), events)
                self._dispatch(events)
        except SSEError as exc:
            self._failed = True
            exc.events = events
            raise
        return events

    def _process(self, text: str, events: list[SSEEvent]) -> None:
        if not text:
            return
        if not self._started:
            self._started = True
            if text[0] == _BOM:  # one leading byte order mark is not part of the stream
                text = text[1:]
        if self._skip_lf:
            self._skip_lf = False
            if text.startswith("\n"):
                text = text[1:]
        start = 0
        for m in _EOL.finditer(text):  # only the new text is scanned
            self._handle_line(self._take_line(text[start : m.start()]), events)
            start = m.end()
        # A "\r" at the very end may be the first half of a "\r\n" split across reads.
        self._skip_lf = start == len(text) and text.endswith("\r")
        if start < len(text):
            if self._partial is None:
                self._partial = io.StringIO()
            self._partial_size += self._partial.write(text[start:])
            if self._partial_size + self._size > self._max:
                raise SSEError("event exceeds the maximum size")

    def _take_line(self, tail: str) -> str:
        """The line ending at ``tail``, joined once with what earlier reads left of it."""
        if self._partial is None:
            return tail
        self._partial.write(tail)
        line = self._partial.getvalue()
        self._partial = None
        self._partial_size = 0
        return line

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
