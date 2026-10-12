"""Structure of indexed Odoo addons: dependencies, model definitions and methods.

Kept apart from the vectors: overrides and adapter implementations are answered by exact lookup, not by similarity.
Only addons added to the graph are known; ``depends`` on anything else (Odoo core, other repositories) is reported by
``missing_depends`` and otherwise ignored. The graph is static analysis: a dynamic ``_inherit`` or a call by string name
is not in it, so every chain it returns is "as found in the source".
"""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rag import config
from rag.sources.odoo_module import CodeUnit

GRAPH_FILE = "graph.json"


def graph_path() -> Path:
    return config.DATA_DIR / "code" / GRAPH_FILE


@dataclass
class ModuleInfo:
    name: str
    version: str
    depends: list[str]
    origin: str  # core | connector | customer
    ref: str  # git ref the module was read at; "" for a working tree

    @classmethod
    def from_units(cls, name: str, units: list[CodeUnit], origin: str, ref: str = "") -> ModuleInfo:
        manifest = next((u for u in units if u.kind == "manifest"), None)
        return cls(
            name, manifest.module_version if manifest else "",
            list(manifest.extra["depends"]) if manifest else [], origin, ref,
        )


@dataclass
class ModelDef:
    model: str
    module: str
    class_name: str
    path: str
    line_start: int
    extends: bool  # False: the module defines the model, True: it extends an existing one


@dataclass
class MethodRef:
    module: str
    model: str
    name: str
    path: str
    line_start: int
    line_end: int
    calls_super: bool
    class_name: str
    adapter: bool
    abstract: bool


@dataclass
class AdapterGroup:
    abstract: list[MethodRef]
    impls: dict[str, list[MethodRef]]  # module -> implementations of the method in adapter classes


@dataclass
class CodeGraph:
    modules: dict[str, ModuleInfo] = field(default_factory=dict)
    models: list[ModelDef] = field(default_factory=list)
    methods: list[MethodRef] = field(default_factory=list)

    def add_module(self, info: ModuleInfo, units: list[CodeUnit]) -> None:
        """Add a module, replacing an earlier copy of it."""
        if info.name in self.modules:
            self.remove_module(info.name)
        self.modules[info.name] = info
        for u in units:
            if u.kind == "model":
                self.models.append(ModelDef(
                    u.model, u.module, u.extra["class_name"], u.path, u.line_start, u.extra["extends"],
                ))
            elif u.kind == "method":
                self.methods.append(MethodRef(
                    u.module, u.model, u.name, u.path, u.line_start, u.line_end, u.extra["calls_super"],
                    u.extra["class_name"], u.extra["adapter"], u.extra["abstract"],
                ))

    def remove_module(self, name: str) -> None:
        del self.modules[name]
        self.models = [m for m in self.models if m.module != name]
        self.methods = [m for m in self.methods if m.module != name]

    def missing_depends(self, name: str) -> list[str]:
        return [d for d in self.modules[name].depends if d not in self.modules]

    def depends_closure(self, name: str) -> set[str]:
        """The known modules `name` depends on, directly or not (not `name` itself)."""
        seen: set[str] = set()
        todo = [d for d in self.modules[name].depends if d in self.modules]
        while todo:
            d = todo.pop()
            if d not in seen and d != name:
                seen.add(d)
                todo += [x for x in self.modules[d].depends if x in self.modules]
        return seen

    def order(self) -> list[str]:
        """Module names, dependencies first (by depth among known modules, then by name). Cycles do not loop."""
        depth: dict[str, int] = {}

        def depth_of(name: str, seen: frozenset[str] = frozenset()) -> int:
            if name not in depth:
                parents = [d for d in self.modules[name].depends if d in self.modules and d not in seen]
                depth[name] = 1 + max((depth_of(d, seen | {name}) for d in parents), default=-1)
            return depth[name]

        return sorted(self.modules, key=lambda n: (depth_of(n), n))

    def model_defs(self, model: str) -> list[ModelDef]:
        rank = {n: i for i, n in enumerate(self.order())}
        return sorted((m for m in self.models if m.model == model), key=lambda m: (rank[m.module], m.path, m.line_start))

    def definitions(self, method: str, model: str | None = None, only: Collection[str] | None = None) -> list[MethodRef]:
        """Every definition of ``method`` (on ``model`` when given, any model or plain class when not), base first;
        ``only`` limits it to the given modules."""
        rank = {n: i for i, n in enumerate(self.order())}
        found = [
            m for m in self.methods
            if m.name == method and (model is None or m.model == model) and (only is None or m.module in only)
        ]
        return sorted(found, key=lambda m: (rank[m.module], m.path, m.line_start))

    def chain(self, model: str, method: str, only: Collection[str] | None = None) -> list[MethodRef]:
        """Definitions of ``method`` on ``model``, base first; ``only`` limits it to the given modules."""
        return self.definitions(method, model, only)

    def adapter_group(self, method: str) -> AdapterGroup:
        """The abstract declaration of ``method`` and its implementation in each module's adapter class."""
        rank = {n: i for i, n in enumerate(self.order())}
        named = sorted(
            (m for m in self.methods if m.name == method), key=lambda m: (rank[m.module], m.path, m.line_start)
        )
        impls: dict[str, list[MethodRef]] = {}
        for m in named:
            if m.adapter:
                impls.setdefault(m.module, []).append(m)
        return AdapterGroup([m for m in named if m.abstract], impls)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "modules": [asdict(m) for m in self.modules.values()],
            "models": [asdict(m) for m in self.models],
            "methods": [asdict(m) for m in self.methods],
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> CodeGraph:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(
            {m["name"]: ModuleInfo(**m) for m in data["modules"]},
            [ModelDef(**m) for m in data["models"]],
            [MethodRef(**m) for m in data["methods"]],
        )
