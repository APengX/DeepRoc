from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvaluationConfig:
    command: list[str]
    metric_pattern: str
    goal: str
    timeout_seconds: int = 300


@dataclass
class ProviderConfig:
    type: str
    path: str | None = None
    command: list[str] | None = None


@dataclass
class LabConfig:
    name: str
    program_path: str
    mutable_paths: list[str]
    evaluation: EvaluationConfig
    provider: ProviderConfig
    journal_dir: str = ".deeproc"


@dataclass
class Change:
    action: str
    path: str
    content: str | None = None
    search: str | None = None
    replace: str | None = None


@dataclass
class Proposal:
    summary: str
    rationale: str
    changes: list[Change]


@dataclass
class RunRecord:
    timestamp: float
    commit: str
    status: str
    metric: float | None
    best_metric: float | None
    kept: bool
    summary: str
    rationale: str
    output_path: str


class DeepRocError(RuntimeError):
    """Base error for DeepRoc failures."""


class ConfigError(DeepRocError):
    """Raised when the config is invalid."""


class ProviderError(DeepRocError):
    """Raised when a provider fails."""


class ProposalApplicationError(DeepRocError):
    """Raised when a proposal cannot be applied."""


def load_config(path: Path) -> LabConfig:
    raw = json.loads(path.read_text())
    evaluation = EvaluationConfig(**raw["evaluation"])
    provider = ProviderConfig(**raw["provider"])
    return LabConfig(
        name=raw["name"],
        program_path=raw["program_path"],
        mutable_paths=raw["mutable_paths"],
        evaluation=evaluation,
        provider=provider,
        journal_dir=raw.get("journal_dir", ".deeproc"),
    )


def _run(command: list[str], cwd: Path, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


class GitRepo:
    def __init__(self, root: Path):
        self.root = root

    def _git(self, *args: str) -> str:
        result = _run(["git", *args], self.root)
        if result.returncode != 0:
            raise DeepRocError(result.stderr.strip() or result.stdout.strip())
        return result.stdout.strip()

    def ensure_clean(self, ignored_paths: list[str] | None = None) -> None:
        ignored_paths = ignored_paths or []
        status = self._git("status", "--porcelain")
        remaining = []
        for line in status.splitlines():
            if not line:
                continue
            rel_path = line[3:]
            if any(rel_path == ignored or rel_path.startswith(f"{ignored}/") for ignored in ignored_paths):
                continue
            remaining.append(line)
        if remaining:
            raise DeepRocError(
                "Working tree is dirty. Commit or stash changes before running DeepRoc."
            )

    def head(self) -> str:
        return self._git("rev-parse", "HEAD")

    def short_head(self) -> str:
        return self._git("rev-parse", "--short", "HEAD")

    def commit_paths(self, message: str, paths: list[str]) -> str:
        self._git("add", "--", *paths)
        self._git("commit", "-m", message)
        return self.short_head()

    def reset_hard(self, ref: str) -> None:
        self._git("reset", "--hard", ref)


class ProposalProvider:
    def generate(self, context: dict[str, Any]) -> Proposal:
        raise NotImplementedError


class ReplayProvider(ProposalProvider):
    def __init__(self, path: Path):
        self.path = path

    def generate(self, context: dict[str, Any]) -> Proposal:
        raw = json.loads(self.path.read_text())
        return Proposal(
            summary=raw["summary"],
            rationale=raw.get("rationale", ""),
            changes=[Change(**item) for item in raw.get("changes", [])],
        )


class ExternalCommandProvider(ProposalProvider):
    def __init__(self, command: list[str], cwd: Path):
        self.command = command
        self.cwd = cwd

    def generate(self, context: dict[str, Any]) -> Proposal:
        result = subprocess.run(
            self.command,
            cwd=str(self.cwd),
            text=True,
            input=json.dumps(context),
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise ProviderError(result.stderr.strip() or result.stdout.strip())
        try:
            raw = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ProviderError("provider output must be valid JSON") from exc
        return Proposal(
            summary=raw["summary"],
            rationale=raw.get("rationale", ""),
            changes=[Change(**item) for item in raw.get("changes", [])],
        )


def build_provider(config: LabConfig, root: Path) -> ProposalProvider:
    if config.provider.type == "replay":
        if not config.provider.path:
            raise ConfigError("replay provider requires a path")
        return ReplayProvider(root / config.provider.path)
    if config.provider.type == "external-command":
        if not config.provider.command:
            raise ConfigError("external-command provider requires a command")
        return ExternalCommandProvider(config.provider.command, root)
    raise ConfigError(f"unsupported provider type: {config.provider.type}")


def journal_paths(root: Path, config: LabConfig) -> tuple[Path, Path]:
    journal_dir = root / config.journal_dir
    journal_dir.mkdir(parents=True, exist_ok=True)
    return journal_dir / "results.jsonl", journal_dir / "state.json"


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text())


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.write_text(json.dumps(state, indent=2) + "\n")


def append_record(results_path: Path, record: RunRecord) -> None:
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")


def ensure_mutable(root: Path, mutable_paths: list[str], relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    allowed = [(root / entry).resolve() for entry in mutable_paths]
    for allowed_path in allowed:
        if candidate == allowed_path or allowed_path in candidate.parents:
            return candidate
    raise ProposalApplicationError(f"path is outside mutable scope: {relative_path}")


def apply_change(root: Path, mutable_paths: list[str], change: Change) -> None:
    target = ensure_mutable(root, mutable_paths, change.path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if change.action == "write":
        if change.content is None:
            raise ProposalApplicationError("write action requires content")
        target.write_text(change.content)
        return
    if change.action == "replace":
        if change.search is None or change.replace is None:
            raise ProposalApplicationError("replace action requires search and replace")
        if not target.exists():
            raise ProposalApplicationError(f"target does not exist: {change.path}")
        original = target.read_text()
        if change.search not in original:
            raise ProposalApplicationError(f"search text not found in {change.path}")
        updated = original.replace(change.search, change.replace, 1)
        target.write_text(updated)
        return
    if change.action == "append":
        if change.content is None:
            raise ProposalApplicationError("append action requires content")
        prefix = target.read_text() if target.exists() else ""
        target.write_text(prefix + change.content)
        return
    raise ProposalApplicationError(f"unsupported change action: {change.action}")


def apply_proposal(root: Path, config: LabConfig, proposal: Proposal) -> None:
    for change in proposal.changes:
        apply_change(root, config.mutable_paths, change)


def gather_context(root: Path, config: LabConfig, state: dict[str, Any]) -> dict[str, Any]:
    files = []
    for rel_path in config.mutable_paths:
        path = root / rel_path
        if path.is_file():
            files.append({"path": rel_path, "content": path.read_text()})
    program = (root / config.program_path).read_text()
    return {
        "lab": config.name,
        "program": program,
        "mutable_paths": config.mutable_paths,
        "files": files,
        "state": state,
        "response_schema": {
            "summary": "string",
            "rationale": "string",
            "changes": [
                {
                    "action": "write|replace|append",
                    "path": "relative path inside mutable scope",
                    "content": "string for write/append",
                    "search": "string for replace",
                    "replace": "replacement string for replace",
                }
            ],
        },
    }


def run_evaluation(root: Path, evaluation: EvaluationConfig, output_path: Path) -> tuple[str, float]:
    result = _run(evaluation.command, root, timeout=evaluation.timeout_seconds)
    output = result.stdout + result.stderr
    output_path.write_text(output)
    if result.returncode != 0:
        raise DeepRocError(f"evaluation command failed with exit code {result.returncode}")
    match = re.search(evaluation.metric_pattern, output)
    if not match:
        raise DeepRocError("metric pattern did not match evaluation output")
    metric = float(match.group(1))
    return output, metric


def is_improvement(metric: float, best_metric: float | None, goal: str) -> bool:
    if best_metric is None:
        return True
    if goal == "min":
        return metric < best_metric
    if goal == "max":
        return metric > best_metric
    raise ConfigError(f"unsupported goal: {goal}")


def run_iteration(config_path: str | Path) -> RunRecord:
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = load_config(config_path)
    repo = GitRepo(root)
    repo.ensure_clean(ignored_paths=[config.journal_dir])
    provider = build_provider(config, root)
    results_path, state_path = journal_paths(root, config)
    state = load_state(state_path)
    best_metric = state.get("best_metric")
    baseline_commit = repo.head()
    proposal = provider.generate(gather_context(root, config, state))
    apply_proposal(root, config, proposal)
    commit = repo.commit_paths(proposal.summary, config.mutable_paths)
    output_path = results_path.parent / f"run-{int(time.time())}.log"
    status = "keep"
    kept = False
    metric = None
    try:
        _, metric = run_evaluation(root, config.evaluation, output_path)
        kept = is_improvement(metric, best_metric, config.evaluation.goal)
        if kept:
            status = "keep"
            state["best_metric"] = metric
            state["best_commit"] = commit
        else:
            status = "discard"
            repo.reset_hard(baseline_commit)
    except Exception:
        status = "crash"
        repo.reset_hard(baseline_commit)
        raise
    finally:
        final_best = state.get("best_metric", best_metric)
        record = RunRecord(
            timestamp=time.time(),
            commit=commit,
            status=status,
            metric=metric,
            best_metric=final_best,
            kept=kept,
            summary=proposal.summary,
            rationale=proposal.rationale,
            output_path=str(output_path.relative_to(root)),
        )
        append_record(results_path, record)
        save_state(state_path, state)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a DeepRoc agent iteration.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="run one agent iteration")
    run_parser.add_argument("--config", default="deeproc.json", help="path to lab config")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        record = run_iteration(args.config)
        metric = "n/a" if record.metric is None else f"{record.metric:.6f}"
        print(
            json.dumps(
                {
                    "status": record.status,
                    "metric": metric,
                    "kept": record.kept,
                    "commit": record.commit,
                    "output_path": record.output_path,
                }
            )
        )
        return 0
    parser.error("unknown command")
    return 2
