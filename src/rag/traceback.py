"""Python tracebacks (as stored by `queue.job.exc_info` or pasted by a user) -> frames, exception, error codes.

Tolerant on purpose: a traceback arrives truncated (no header, no exception line), with wrapped lines, CRLF line
endings, chained or glued together. A frame's path is mapped to an indexed module by directory name, never by a
prefix: the prefix on a customer's server is unknown.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection
from dataclasses import dataclass, field

HEADER = re.compile(r"^Traceback \(most recent call last\):[ \t]*$", re.M)
CHAIN = re.compile(
    r"^(?:During handling of the above exception, another exception occurred:"
    r"|The above exception was the direct cause of the following exception:)[ \t]*$",
    re.M,
)
# Path and header may be wrapped by a terminal or mail client: whitespace is allowed between the parts.
FRAME = re.compile(r'File\s+"([^"]*)",\s*line\s+(\d+)(?:,\s*in\s+(\S+))?')
CARETS = re.compile(r"[\s^~\-]+")  # the ^^^^ / ~~~^^^ underline Python 3.11+ prints under a code line
# `pkg.module.ClassName: message`: the last dotted segment is a class name.
EXC_LINE = re.compile(r"^((?:[A-Za-z_]\w*\.)*[A-Z]\w*)(?::[ \t]*(.*))?$")
EXC_SUFFIXES = ("Error", "Exception", "Warning", "Exit", "Interrupt", "Violation")
ERROR_CODE = re.compile(r"(?<![A-Za-z0-9])E\d{3}(?!\d)")


@dataclass
class Frame:
    path: str
    line: int
    func: str
    code: str = ""
    module: str = ""  # indexed module the path belongs to; "" for anything else (queue job, Odoo core, stdlib)
    rel: str = ""  # path inside the module
    alternatives: list[tuple[str, str]] = field(default_factory=list)  # other (module, rel) readings of the path


@dataclass
class ExceptionInfo:
    name: str
    message: str

    @property
    def short_name(self) -> str:
        return self.name.rsplit(".", 1)[-1]


@dataclass
class ParsedTraceback:
    frames: list[Frame] = field(default_factory=list)  # outermost first, as printed
    exception: ExceptionInfo | None = None
    error_codes: list[str] = field(default_factory=list)
    chained: list[ParsedTraceback] = field(default_factory=list)  # earlier tracebacks; this one is the last

    @property
    def own_frames(self) -> list[Frame]:
        """Frames inside indexed modules, innermost first: where the failure happened comes before who called it."""
        return [f for f in reversed(self.frames) if f.module]


def module_candidates(path: str, modules: Collection[str]) -> list[tuple[str, str]]:
    """(module, path inside it) for every directory of `path` that is a known module name, the one closest to the
    file first. More than one when a folder above the addon carries a module's name too (a repository called
    `integration` holding `integration_shopify`), or the addon holds a folder of its own name."""
    parts = path.replace("\\", "/").split("/")
    return [(parts[i], "/".join(parts[i + 1 :])) for i in range(len(parts) - 2, -1, -1) if parts[i] in modules]


def resolve_module(path: str, modules: Collection[str]) -> tuple[str, str]:
    """(module, path inside it) by the directory closest to the file that is a known module name; ("", "") if none.

    `/srv/integration/ext/integration_shopify/models/a.py` is `integration_shopify`, not `integration`. A path that
    is ambiguous (`.../demo_core/demo_core/a.py`) keeps its other readings in `Frame.alternatives`: the lookup
    in the index (`resolve_frames`) takes the first one that exists."""
    candidates = module_candidates(path, modules)
    return candidates[0] if candidates else ("", "")


def split_blocks(text: str) -> list[str]:
    """One block per traceback: split at chaining sentences and at repeated `Traceback` headers."""
    blocks = []
    for part in CHAIN.split(text):
        cuts = [0] + [m.start() for m in HEADER.finditer(part)][1:]
        blocks += [p for a, b in zip(cuts, cuts[1:] + [len(part)]) if (p := part[a:b]).strip()]
    return blocks


def exception_from(lines: list[str], strict: bool) -> ExceptionInfo | None:
    """The exception in `lines` (the first column-0 line and what follows). `strict`: no frames came before it, so
    the name must look like an exception class, or any `Hello: text` line would pass."""
    for i, line in enumerate(lines):
        if not line.strip() or line[0] in " \t":
            continue
        m = EXC_LINE.match(line.rstrip())
        if not m or (strict and "." not in m[1] and not m[1].endswith(EXC_SUFFIXES)):
            if strict:
                continue
            return None
        message = "\n".join([m[2] or "", *lines[i + 1 :]]).strip()
        return ExceptionInfo(m[1], message)
    return None


def parse_block(block: str, modules: Collection[str]) -> ParsedTraceback:
    matches = list(FRAME.finditer(block))
    tb = ParsedTraceback()
    last_segment: list[str] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        segment = block[m.end() : end].split("\n")[1:]  # the rest of the header line is dropped
        code = ""
        for n, line in enumerate(segment):
            if not line.strip():
                continue
            if line[0] not in " \t":
                segment = segment[n:]  # the exception starts here
                break
            if not code and not CARETS.fullmatch(line):
                code = line.strip()
        else:
            segment = []
        path = re.sub(r"\s+", "", m[1])
        (module, rel), *alternatives = module_candidates(path, modules) or [("", "")]
        tb.frames.append(Frame(path, int(m[2]), m[3] or "", code, module, rel, alternatives))
        last_segment = segment
    tb.exception = exception_from(last_segment if matches else block.split("\n"), strict=not matches)
    codes = ERROR_CODE.findall(tb.exception.message) if tb.exception else []
    for f in tb.frames:
        codes += ERROR_CODE.findall(f.code)
    tb.error_codes = list(dict.fromkeys(codes))
    return tb


def parse_traceback(text: str, modules: Collection[str]) -> ParsedTraceback:
    """Parse `text`; with several tracebacks the last is the result and the earlier ones are in `chained`."""
    blocks = [parse_block(b, modules) for b in split_blocks(text.replace("\r\n", "\n").replace("\r", "\n"))]
    blocks = [b for b in blocks if b.frames or b.exception]
    if not blocks:
        return ParsedTraceback()
    main = blocks[-1]
    main.chained = blocks[:-1]
    return main


def looks_like_traceback(text: str) -> bool:
    """Cheap check for the UI: does the pasted text look like a traceback rather than a question?"""
    tb = parse_traceback(text, ())
    return bool(tb.frames or tb.exception)


def readable_message(message: str) -> str:
    """A platform's JSON error body reduced to its code and message; any other text is only whitespace-normalised."""
    try:
        body = json.loads(message)
    except ValueError:
        body = None
    if isinstance(body, dict) and isinstance(body.get("message"), str):
        return " ".join(str(x) for x in (body.get("code"), body["message"]) if x).strip()
    return " ".join(message.split())


def semantic_query(tb: ParsedTraceback, max_chars: int = 400) -> str:
    """The text for the semantic search: the error message, the exception class and the names of our own functions."""
    parts = []
    if tb.exception:
        parts += [readable_message(tb.exception.message)[:300], tb.exception.short_name]
    parts += list(dict.fromkeys(f.func for f in tb.own_frames))[:6]
    return " ".join(p for p in parts if p)[:max_chars].strip()


def version_note(client_version: str | None, indexed_version: str) -> str | None:
    if not client_version or client_version == indexed_version:
        return None
    return f"The client runs {client_version}, the indexed code is {indexed_version}: line numbers may not match."
