#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]


# -------------------------
# Utility helpers
# -------------------------


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> str:
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()


def list_repo_files(roots: List[Path]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                files.append(p)
    return files


def normalize_path(p: Path) -> Path:
    try:
        return p.resolve().relative_to(REPO_ROOT)
    except Exception:
        return p


# -------------------------
# Dependency graph builders
# -------------------------


CPP_INCLUDE_RE = re.compile(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]")
PY_IMPORT_RE = re.compile(r"^\s*(?:from\s+([a-zA-Z0-9_\.]+)\s+import|import\s+([a-zA-Z0-9_\.]+))")


def is_repo_header(path_str: str) -> bool:
    return path_str.startswith(("tt_metal/", "tt_stl/", "ttnn/", "tests/", "tt-train/"))


def resolve_include(from_file: Path, include_path: str) -> Optional[Path]:
    # Try relative to including file
    rel = (from_file.parent / include_path).resolve()
    if rel.exists():
        try:
            return rel.relative_to(REPO_ROOT)
        except Exception:
            return None
    # Try from repo root for in-repo includes
    root_try = (REPO_ROOT / include_path).resolve()
    if root_try.exists():
        try:
            return root_try.relative_to(REPO_ROOT)
        except Exception:
            return None
    return None


def parse_cpp_includes(file_path: Path) -> Set[Path]:
    deps: Set[Path] = set()
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = CPP_INCLUDE_RE.match(line)
                if not m:
                    continue
                inc = m.group(1)
                if "/" not in inc and not inc.endswith((".h", ".hpp", ".hh")):
                    # Likely a system header
                    continue
                if is_repo_header(inc):
                    resolved = resolve_include(file_path, inc)
                    if resolved:
                        deps.add(resolved)
    except Exception:
        pass
    return deps


def module_to_paths(module: str) -> List[Path]:
    parts = module.split(".")
    p = REPO_ROOT.joinpath(*parts)
    candidates = []
    if (p.with_suffix(".py")).exists():
        candidates.append((p.with_suffix(".py")).relative_to(REPO_ROOT))
    pkg_init = p / "__init__.py"
    if pkg_init.exists():
        candidates.append(pkg_init.relative_to(REPO_ROOT))
    return candidates


def parse_python_imports(file_path: Path) -> Set[Path]:
    deps: Set[Path] = set()
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = PY_IMPORT_RE.match(line)
                if not m:
                    continue
                mod = m.group(1) or m.group(2)
                if not mod:
                    continue
                # Only track imports within this monorepo namespaces
                if not mod.startswith(("ttnn", "tt_metal", "tt_train", "models", "tests")):
                    continue
                for cand in module_to_paths(mod):
                    deps.add(cand)
    except Exception:
        pass
    return deps


def build_dependency_graph(files: Iterable[Path]) -> Dict[Path, Set[Path]]:
    graph: Dict[Path, Set[Path]] = defaultdict(set)
    for f in files:
        rel = normalize_path(f)
        if str(rel).endswith((".c", ".cc", ".cpp", ".h", ".hpp", ".hh")):
            deps = parse_cpp_includes(REPO_ROOT / rel)
            if deps:
                graph[rel].update(deps)
        elif str(rel).endswith(".py"):
            deps = parse_python_imports(REPO_ROOT / rel)
            if deps:
                graph[rel].update(deps)
    return graph


def build_reverse_graph(graph: Dict[Path, Set[Path]]) -> Dict[Path, Set[Path]]:
    rev: Dict[Path, Set[Path]] = defaultdict(set)
    for src, deps in graph.items():
        for dep in deps:
            rev[dep].add(src)
    return rev


def downstream_impact(changed: Iterable[Path], rev_graph: Dict[Path, Set[Path]]) -> Set[Path]:
    impacted: Set[Path] = set()
    dq: deque[Path] = deque()
    for c in changed:
        dq.append(c)
    while dq:
        node = dq.popleft()
        if node in impacted:
            continue
        impacted.add(node)
        for nxt in rev_graph.get(node, set()):
            if nxt not in impacted:
                dq.append(nxt)
    return impacted


# -------------------------
# Component mapping & workflow mapping
# -------------------------


COMPONENT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("cmake", re.compile(r"^(CMakeLists\\.txt|cmake/|tt_metal/sfpi-version\\.sh)")),
    ("docs", re.compile(r"^(docs/|tech_reports/|.*\\.(md|rst)$)")),
    ("tt-metalium", re.compile(r"^(tt_metal/|tt_stl/)")),
    ("ttnn", re.compile(r"^(ttnn/)")),
    ("tt-train", re.compile(r"^(tt-train/)")),
    ("models", re.compile(r"^(models/)")),
    ("tests", re.compile(r"^(tests/)")),
]


def file_to_components(paths: Iterable[Path]) -> Set[str]:
    comps: Set[str] = set()
    for p in paths:
        s = str(p)
        for name, pat in COMPONENT_PATTERNS:
            if pat.search(s):
                comps.add(name)
    return comps


def discover_workflows() -> Dict[str, Path]:
    workflows_dir = REPO_ROOT / ".github" / "workflows"
    mapping: Dict[str, Path] = {}
    if workflows_dir.exists():
        for wf in workflows_dir.glob("*.yaml"):
            mapping[wf.name] = wf
    return mapping


def recommend_workflows(components: Set[str], workflows: Dict[str, Path]) -> List[str]:
    # Heuristic mapping based on filename keywords
    recs: Set[str] = set()
    comp_keywords = {
        "ttnn": ["ttnn", "tt-nn"],
        "tt-metalium": ["metal", "tt-metal"],
        "tt-train": ["tt-train", "train"],
        "docs": ["doc"],
        "models": ["model"],
        "cmake": ["build", "cpp-post-commit", "asan", "unit-tests"],
        "tests": ["unit-tests", "smoke", "frequent", "nightly"],
    }
    for wf_name in workflows.keys():
        lowered = wf_name.lower()
        for comp in components:
            for kw in comp_keywords.get(comp, []):
                if kw in lowered:
                    recs.add(wf_name)
        # Broad workflows
        if any(k in lowered for k in ["pr-gate", "merge-gate", "all-post-commit", "build-and-unit-tests", "cpp-post-commit"]):
            if components & {"tt-metalium", "ttnn", "tt-train", "tests", "cmake"}:
                recs.add(wf_name)
    # Prefer wrapper workflows where both wrapper and impl exist
    wrappers = [n for n in recs if n.endswith("-wrapper.yaml")]
    if wrappers:
        return sorted(set(wrappers))
    return sorted(recs)


# -------------------------
# Main logic
# -------------------------


def get_changed_files(base_ref: Optional[str], head_ref: Optional[str], explicit: List[str]) -> List[Path]:
    if explicit:
        return [normalize_path((REPO_ROOT / e)) for e in explicit]
    # Default to merge-base with origin/main
    base = base_ref or run_cmd(["git", "merge-base", "origin/main", "HEAD"])  # noqa: S603
    head = head_ref or "HEAD"
    out = run_cmd(["git", "diff", "--name-only", "--diff-filter=ACMRT", f"{base}..{head}"])  # noqa: S603
    changed = []
    for line in out.splitlines():
        p = (REPO_ROOT / line).resolve()
        try:
            changed.append(p.relative_to(REPO_ROOT))
        except Exception:
            pass
    return changed


@dataclass
class ImpactResult:
    changed_files: List[str]
    impacted_files: List[str]
    impacted_components: List[str]
    recommended_workflows: List[str]


def compute_impact(changed_files: List[Path]) -> ImpactResult:
    # Scope graph to relevant subtrees for speed
    scope_roots = [
        REPO_ROOT / "tt_metal",
        REPO_ROOT / "tt_stl",
        REPO_ROOT / "ttnn",
        REPO_ROOT / "tt-train",
        REPO_ROOT / "tests",
        REPO_ROOT / "models",
        REPO_ROOT / "docs",
    ]
    files = list_repo_files(scope_roots)
    graph = build_dependency_graph(files)
    rev = build_reverse_graph(graph)

    impacted = downstream_impact(changed_files, rev)
    impacted |= set(changed_files)

    impacted_components = sorted(file_to_components(impacted))
    workflows = discover_workflows()
    recommended = recommend_workflows(set(impacted_components), workflows)

    return ImpactResult(
        changed_files=sorted(str(p) for p in changed_files),
        impacted_files=sorted(str(p) for p in impacted),
        impacted_components=impacted_components,
        recommended_workflows=recommended,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Recommend CI workflows based on changed files and dependency impact.")
    parser.add_argument("--base", help="Base git ref for diff", default=None)
    parser.add_argument("--head", help="Head git ref for diff", default=None)
    parser.add_argument("--changed-file", action="append", default=[], help="Explicit changed file (can be repeated)")
    parser.add_argument("--output-json", default=None, help="Write JSON output to path")
    parser.add_argument("--print-markdown", action="store_true", help="Print markdown summary")

    args = parser.parse_args(argv)

    changed = get_changed_files(args.base, args.head, args.changed_file)
    result = compute_impact(changed)

    out_obj = {
        "changed_files": result.changed_files,
        "impacted_files": result.impacted_files,
        "impacted_components": result.impacted_components,
        "recommended_workflows": result.recommended_workflows,
    }

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(out_obj, indent=2) + "\n", encoding="utf-8")

    if args.print_markdown:
        recs_md = "\n".join(f"- {wf}" for wf in result.recommended_workflows) or "- (none)"
        comps_md = ", ".join(result.impacted_components) or "(none)"
        print(f"**Impacted components**: {comps_md}\n\n**Recommended workflows**:\n{recs_md}")
    else:
        print(json.dumps(out_obj))

    # Emit GitHub Actions outputs if present
    gha_out = os.environ.get("GITHUB_OUTPUT")
    if gha_out:
        with open(gha_out, "a", encoding="utf-8") as f:
            f.write(f"recommended-workflows={json.dumps(result.recommended_workflows)}\n")
            f.write(f"impacted-components={json.dumps(result.impacted_components)}\n")
            f.write(f"impacted-files-count={len(result.impacted_files)}\n")

    # Also write to step summary for convenience
    gha_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if gha_summary:
        with open(gha_summary, "a", encoding="utf-8") as f:
            f.write("## CI Impact Recommendation\n\n")
            f.write(f"Impacted components: {', '.join(result.impacted_components) or '(none)'}\n\n")
            if result.recommended_workflows:
                f.write("Recommended workflows to run:\n")
                for wf in result.recommended_workflows:
                    f.write(f"- {wf}\n")
            else:
                f.write("No specific workflows recommended.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())


