"""Knowledge units from an Odoo addon: manifest, model cards, methods, views, data records, docs.

One parser for any addon (core, connector, customer customisation): it knows only Odoo conventions, nothing about a
particular module. Files come from a directory or from a git ref (``git show``), so an indexed version is a tag and
never the working tree. Tests and migrations are skipped; JS/OWL and ``.po`` are not parsed yet.

Static analysis only: a dynamic ``_inherit`` or a method called by its string name is invisible here.
"""

from __future__ import annotations

import ast
import logging
import re
import subprocess
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Protocol
from xml.parsers import expat

log = logging.getLogger(__name__)

# Directories that never reach the index: tests, migration scripts, translations, JS/OWL sources.
SKIP_DIRS = ("tests/", "migrations/", "i18n/", "static/src/")
# A class deriving from one of these is a platform adapter; its methods are marked ``adapter``.
ADAPTER_BASES = ("AbsApiClient",)
DOC_FILES = ("README.rst", "README.md", "static/description/index.html")
# View attributes that carry text a user (or a support engineer) reads.
TEXT_ATTRS = ("string", "help", "placeholder", "title", "confirm", "alt", "label")


def unit_key(module: str, path: str, line_start: int, kind: str) -> str:
    """The id of a unit: where it starts. The one place that builds it (the graph and the evidence look units up by it)."""
    return f"{module}:{path}:{line_start}:{kind}"


def split_lines(source: str) -> list[str]:
    """Lines as Python's tokenizer and expat count them: only \\n, \\r\\n and \\r end a line. `str.splitlines()` also
    splits at form feed, U+2028 and the like, which shifts every line number after such a character."""
    lines = re.split(r"\r\n|\r|\n", source)
    return lines[:-1] if lines and lines[-1] == "" else lines


@dataclass
class CodeUnit:
    kind: str  # manifest | model | class | method | view | data | doc
    module: str
    module_version: str
    origin: str  # core | connector | customer
    model: str
    name: str
    path: str  # inside the module
    line_start: int
    line_end: int
    text: str
    embed_text: str
    extra: dict = field(default_factory=dict)

    @property
    def unit_id(self) -> str:
        """Identifies the unit; the parts of a split unit share it."""
        return self.extra.get("unit_id") or unit_key(self.module, self.path, self.line_start, self.kind)

    @property
    def id(self) -> str:
        return self.unit_id + (f"#{self.extra['part']}" if "part" in self.extra else "")


class FileSource(Protocol):
    def files(self) -> list[str]: ...

    def read(self, rel: str) -> str: ...


class DirSource:
    """The working tree of a module directory."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def files(self) -> list[str]:
        return sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file())

    def read(self, rel: str) -> str:
        return (self.root / rel).read_text(encoding="utf-8", errors="replace")


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)
    return out.stdout.decode("utf-8", errors="replace")


class GitSource:
    """A module directory (``prefix`` inside the repository) at a git ref; the working tree is never read."""

    def __init__(self, repo: Path, ref: str, prefix: str):
        self.repo, self.ref, self.prefix = Path(repo), ref, prefix.strip("/")

    def files(self) -> list[str]:
        out = git(self.repo, "ls-tree", "-r", "--name-only", "-z", self.ref, "--", self.prefix + "/")
        return sorted(p[len(self.prefix) + 1 :] for p in out.split("\0") if p)

    def read(self, rel: str) -> str:
        return git(self.repo, "show", f"{self.ref}:{self.prefix}/{rel}")


def git_location(module_dir: Path) -> tuple[Path, str]:
    """(repository root, module path inside it) for a module directory."""
    module_dir = Path(module_dir).resolve()
    top = Path(git(module_dir, "rev-parse", "--show-toplevel").strip())
    return top, module_dir.relative_to(top).as_posix()


def latest_tag(repo: Path, prefix: str, series: str) -> str | None:
    """Highest ``<prefix>-<series>.<n>[.<n>...]`` tag, compared numerically; pre-release suffixes are ignored."""
    pattern = re.compile(rf"^{re.escape(prefix)}-{re.escape(series)}\.(\d+(?:\.\d+)*)$")
    found = []
    for tag in git(repo, "tag", "-l", f"{prefix}-{series}.*").split():
        if m := pattern.match(tag):
            found.append((tuple(int(n) for n in m.group(1).split(".")), tag))
    return max(found)[1] if found else None


def parse_module(source: FileSource, module: str, origin: str = "connector") -> list[CodeUnit]:
    """Every knowledge unit of a module. A file that cannot be parsed is logged and skipped."""
    files = [f for f in source.files() if not f.startswith(SKIP_DIRS)]
    version = ""
    units: list[CodeUnit] = []

    if "__manifest__.py" in files:
        try:
            manifest = ast.literal_eval(source.read("__manifest__.py"))
            version = str(manifest.get("version", ""))
            units.append(manifest_unit(module, origin, manifest))
        except (ValueError, SyntaxError, AttributeError) as e:
            log.warning("%s: unreadable __manifest__.py: %s", module, e)

    for rel in files:
        if rel == "__manifest__.py":
            continue
        parse = None
        if rel.endswith(".py"):
            parse = python_units
        elif rel.endswith(".xml"):
            parse = xml_units
        elif is_doc(rel):
            parse = doc_units
        if parse is None:
            continue
        try:
            units.extend(parse(module, version, origin, rel, source.read(rel)))
        except (SyntaxError, ValueError, expat.ExpatError) as e:
            log.warning("%s/%s: skipped, cannot parse: %s", module, rel, e)
    return unique_ids(units)


def unique_ids(units: list[CodeUnit]) -> list[CodeUnit]:
    """Units that start on the same line and have the same kind (two records on one line of a minified XML file) would
    share an id and overwrite each other in the index: the later ones get a counter."""
    seen: dict[str, int] = {}
    for u in units:
        base = u.unit_id
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            u.extra["unit_id"] = f"{base}~{seen[base]}"
    return units


def is_doc(rel: str) -> bool:
    return rel in DOC_FILES or (rel.startswith("doc/") and rel.endswith((".rst", ".md")))


def manifest_unit(module: str, origin: str, manifest: dict) -> CodeUnit:
    depends = list(manifest.get("depends", []))
    lines = [str(manifest.get("name", module)), str(manifest.get("summary", ""))]
    if isinstance(manifest.get("description"), str):
        lines.append(manifest["description"].strip())
    lines += [f"version: {manifest.get('version', '')}", "depends: " + ", ".join(depends)]
    text = "\n".join(line for line in lines if line)
    return CodeUnit(
        "manifest", module, str(manifest.get("version", "")), origin, "", module, "__manifest__.py", 1, 1,
        text, f"{module}\n{text}", {"depends": depends},
    )


# --- Python -----------------------------------------------------------------------------------------------------


def cut(lines: list[str], start: int, end: int) -> str:
    """Source lines start..end (1-based), with the first line's indentation removed from all of them."""
    chunk = lines[start - 1 : end]
    indent = len(chunk[0]) - len(chunk[0].lstrip())
    return "\n".join(line[indent:] if line[:indent].strip() == "" else line for line in chunk)


def const(node: ast.AST | None):
    try:
        return ast.literal_eval(node) if node is not None else None
    except (ValueError, TypeError, SyntaxError):
        return None


def name_of(node: ast.AST) -> str:
    return ast.unparse(node)


def class_attrs(cls: ast.ClassDef) -> dict:
    """Literal values of ``_name``, ``_inherit``, ``_inherits``, ``_description`` in the class body."""
    attrs = {}
    for node in cls.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("_name", "_inherit", "_inherits", "_description"):
                attrs[node.targets[0].id] = const(node.value)
    return attrs


def model_of(attrs: dict) -> tuple[str, bool]:
    """(model name, extends an existing model?) from the class attributes; ('', False) for a plain class."""
    inherit = attrs.get("_inherit")
    parents = [inherit] if isinstance(inherit, str) else list(inherit or [])
    own = attrs.get("_name")
    if own:
        return own, own in parents
    if parents:
        return parents[0], True
    return "", False


def field_line(target: str, call: ast.Call) -> str | None:
    """One card line for ``name = fields.X(...)``, or None when the call is not a field definition."""
    func = call.func
    if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "fields"):
        return None
    kwargs = {k.arg: const(k.value) for k in call.keywords if k.arg}
    first = const(call.args[0]) if call.args else None
    line = f"- {target} ({func.attr})"
    if kwargs.get("string"):
        line += f' "{kwargs["string"]}"'
    if func.attr == "Selection":
        selection = first if isinstance(first, list) else kwargs.get("selection")
        if isinstance(selection, list):
            line += " [" + ", ".join(f"{k}={v}" for k, v in selection if isinstance(k, str)) + "]"
    elif isinstance(first, str) and func.attr in ("Many2one", "One2many", "Many2many"):
        line += f" -> {first}"
    if kwargs.get("help"):
        line += ": " + " ".join(str(kwargs["help"]).split())
    return line


def model_card(cls: ast.ClassDef, attrs: dict, model: str, extends: bool, module: str) -> str:
    head = f"Model {model}" + (" (extends it in this module)" if extends else " (defined in this module)")
    lines = [head + f" in {module}"]
    if attrs.get("_description"):
        lines.append(f"Description: {attrs['_description']}")
    if doc := ast.get_docstring(cls):
        lines.append(" ".join(doc.split()))
    fields_ = [
        line
        for node in cls.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and (line := field_line(node.targets[0].id, node.value))
    ]
    if fields_:
        lines += ["Fields:", *fields_]
    return "\n".join(lines)


def is_abstract(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(name_of(d).split(".")[-1] == "abstractmethod" for d in func.decorator_list)


def calls_super(func: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "super" for n in ast.walk(func)
    )


def python_units(module: str, version: str, origin: str, rel: str, source: str) -> list[CodeUnit]:
    tree = ast.parse(source)
    lines = split_lines(source)
    units: list[CodeUnit] = []

    def functions(body: list[ast.stmt]):
        return [n for n in body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]

    def method_unit(func, model: str, cls: ast.ClassDef | None, bases: list[str]) -> CodeUnit:
        start = min([func.lineno, *(d.lineno for d in func.decorator_list)])
        text = cut(lines, start, func.end_lineno)
        owner = model or (cls.name if cls else PurePosixPath(rel).stem)
        extra = {
            "class_name": cls.name if cls else "",
            "bases": bases,
            "adapter": any(b.split(".")[-1] in ADAPTER_BASES for b in bases),
            "abstract": is_abstract(func),
            "calls_super": calls_super(func),
            "docstring": ast.get_docstring(func) or "",
        }
        return CodeUnit(
            "method", module, version, origin, model, func.name, rel, start, func.end_lineno, text,
            f"{module} > {owner} > {func.name}\n{text}", extra,
        )

    def class_header_unit(cls: ast.ClassDef, bases: list[str]) -> CodeUnit | None:
        """A plain class's docstring and attributes (a registry dict, constants) up to its first method."""
        methods = functions(cls.body)
        stop = min([m.lineno for m in methods] + [d.lineno for m in methods for d in m.decorator_list], default=cls.end_lineno + 1)
        end = stop - 1
        while end > cls.lineno and not lines[end - 1].strip():
            end -= 1
        if end == cls.lineno:
            return None  # `class X:` and nothing else before the methods
        text = cut(lines, cls.lineno, end)
        return CodeUnit(
            "class", module, version, origin, "", cls.name, rel, cls.lineno, end, text,
            f"{module} > {cls.name}\n{text}", {"class_name": cls.name, "bases": bases},
        )

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            attrs = class_attrs(node)
            model, extends = model_of(attrs)
            bases = [name_of(b) for b in node.bases]
            if model:
                card = model_card(node, attrs, model, extends, module)
                units.append(CodeUnit(
                    "model", module, version, origin, model, model, rel, node.lineno, node.end_lineno,
                    card, f"{module} > {model}\n{card}",
                    {"extends": extends, "class_name": node.name, "bases": bases},
                ))
            elif unit := class_header_unit(node, bases):
                units.append(unit)
            units += [method_unit(f, model, node, bases) for f in functions(node.body)]
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            units.append(method_unit(node, "", None, []))
    return units


# --- XML --------------------------------------------------------------------------------------------------------


@dataclass
class Node:
    tag: str
    attrs: dict
    line_start: int
    line_end: int = 0
    text: str = ""
    children: list[Node] = field(default_factory=list)


def parse_xml(source: str) -> Node:
    """A small element tree with line numbers (ElementTree drops them)."""
    parser = expat.ParserCreate()
    parser.buffer_text = True
    root = Node("", {}, 0)
    stack = [root]

    def start(tag, attrs):
        node = Node(tag, dict(attrs), parser.CurrentLineNumber)
        stack[-1].children.append(node)
        stack.append(node)

    def end(_tag):
        stack.pop().line_end = parser.CurrentLineNumber

    def chars(data):
        stack[-1].text += data

    parser.StartElementHandler, parser.EndElementHandler, parser.CharacterDataHandler = start, end, chars
    parser.Parse(source.encode("utf-8"), True)
    return root.children[0]


def visible_strings(node: Node, in_arch: bool = False) -> list[str]:
    """Texts a reader of a view or record sees: labels, help, button and field names, text nodes."""
    out = [v for k, v in node.attrs.items() if k in TEXT_ATTRS and v]
    if in_arch and node.tag in ("button", "field", "widget") and node.attrs.get("name"):
        out.append(node.attrs["name"])
    if node.text.strip():
        out.append(" ".join(node.text.split()))
    for child in node.children:
        out += visible_strings(child, in_arch)
    return out


def xml_units(module: str, version: str, origin: str, rel: str, source: str) -> list[CodeUnit]:
    lines = split_lines(source)
    units: list[CodeUnit] = []

    def walk(node: Node):
        if node.tag in ("record", "template", "menuitem"):
            unit = xml_unit(node)
            if unit:
                units.append(unit)
        else:
            for child in node.children:
                walk(child)

    def xml_unit(node: Node) -> CodeUnit | None:
        name = node.attrs.get("id", "")
        fields_ = {c.attrs.get("name"): c for c in node.children if c.tag == "field"}
        if node.tag == "template":
            kind, model, strings = "view", "", visible_strings(node, True)
        elif node.tag == "menuitem":
            kind, model = "data", "ir.ui.menu"
            strings = [node.attrs.get("name", "")] if node.attrs.get("name") else []
        elif node.attrs.get("model") == "ir.ui.view":
            kind = "view"
            model = fields_["model"].text.strip() if "model" in fields_ else ""
            strings = []
            for fname, fnode in fields_.items():
                if fname == "arch":  # the wrapper field itself is not a visible string
                    strings += [s for child in fnode.children for s in visible_strings(child, True)]
                else:
                    strings += visible_strings(fnode)
        else:
            kind, model = "data", node.attrs.get("model", "")
            strings = [f"{n}: {' '.join(visible_strings(c) or ['']) }".strip() for n, c in fields_.items()]
        text = cut(lines, node.line_start, node.line_end)
        header = f"{module} > {model} > {name}" if model else f"{module} > {name}"
        return CodeUnit(
            kind, module, version, origin, model, name, rel, node.line_start, node.line_end, text,
            header + "\n" + "\n".join(strings), {"tag": node.tag, "record_model": node.attrs.get("model", "")},
        )

    walk(parse_xml(source))
    return units


# --- docs -------------------------------------------------------------------------------------------------------


class TextExtractor(HTMLParser):
    SKIP = ("script", "style", "head")

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skipping = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skipping += 1
        elif tag in ("p", "br", "li", "h1", "h2", "h3", "h4", "tr", "div"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skipping = max(0, self.skipping - 1)

    def handle_data(self, data):
        if not self.skipping:
            self.parts.append(data)


def html_to_text(source: str) -> str:
    parser = TextExtractor()
    parser.feed(source)
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)


def doc_units(module: str, version: str, origin: str, rel: str, source: str) -> list[CodeUnit]:
    text = html_to_text(source) if rel.endswith(".html") else source.strip()
    if not text:
        return []
    return [CodeUnit(
        "doc", module, version, origin, "", rel, rel, 1, len(split_lines(source)), text, f"{module} > {rel}\n{text}",
    )]
