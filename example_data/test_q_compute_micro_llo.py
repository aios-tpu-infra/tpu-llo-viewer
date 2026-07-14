# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Opt-in LLO collection for individual _q_compute_loop primitive calls."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
MICRO_BENCH = REPO_ROOT / "scripts/vllm/benchmarking/bench_pcp_q_compute_micro.py"

DEFAULT_CASES = (
    # q_slice @ k.T in _q_compute_loop.
    ("qk_dot_general", 64, 16, 256, 256),
    ("qk_dot_general", 256, 16, 256, 256),
    ("qk_dot_general", 512, 16, 256, 256),
    # Causal masking around scores.
    ("causal_mask_where", 64, 16, 256, 256),
    ("causal_mask_where", 512, 16, 256, 256),
    # jnp.max(scores, axis=2, keepdims=True).
    ("max_axis2", 64, 16, 256, 256),
    ("max_axis2", 512, 16, 256, 256),
    # jnp.exp(scores - broadcast(m_next)).
    ("exp_scores", 64, 16, 256, 256),
    ("exp_scores", 512, 16, 256, 256),
    # jnp.sum(p, axis=2, keepdims=True).
    ("sum_axis2", 64, 16, 256, 256),
    ("sum_axis2", 512, 16, 256, 256),
    # p @ v in _q_compute_loop.
    ("pv_dot_general", 64, 16, 256, 256),
    ("pv_dot_general", 256, 16, 256, 256),
    ("pv_dot_general", 512, 16, 256, 256),
    # alpha * acc + pv.
    ("acc_update", 64, 16, 256, 256),
    ("acc_update", 512, 16, 256, 256),
)


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _case_id(op: str, q_size: int, q_per_kv: int, kv_tokens: int,
             head_dim: int) -> str:
    return f"{op}_q{q_size}_g{q_per_kv}_kv{kv_tokens}_d{head_dim}"


def _kernel_name(op: str, q_size: int, q_per_kv: int, kv_tokens: int,
                 head_dim: int) -> str:
    return f"pcp_q_compute_micro_{_case_id(op, q_size, q_per_kv, kv_tokens, head_dim)}"


def _parse_cases(raw: str | None):
    if not raw:
        return DEFAULT_CASES
    cases = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        op, shape = item.split(":", 1)
        q_size, q_per_kv, kv_tokens, head_dim = (
            int(part) for part in shape.split(","))
        cases.append((op, q_size, q_per_kv, kv_tokens, head_dim))
    if not cases:
        raise ValueError("PCP_Q_COMPUTE_MICRO_CASES must not be empty.")
    return tuple(cases)


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src}:{old_pythonpath}" if old_pythonpath else src
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("SKIP_JAX_PRECOMPILE", "1")
    env.setdefault("MODEL_IMPL_TYPE", "vllm")
    env.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    env.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    env.setdefault("TMPDIR", str(REPO_ROOT / "data" / "tmp"))
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def _parse_micro_result(output: str) -> dict[str, object]:
    for line in output.splitlines():
        if line.startswith("MICRO_RESULT "):
            return json.loads(line[len("MICRO_RESULT "):])
    raise AssertionError(f"missing MICRO_RESULT line in child output:\n{output}")


def _first_file(root: Path,
                pattern: str,
                *,
                exclude_name_parts: tuple[str, ...] = ()) -> Path:
    matches = sorted(
        path for path in root.rglob(pattern)
        if path.is_file()
        and not any(part in path.name for part in exclude_name_parts))
    if not matches:
        raise AssertionError(f"missing dump file {pattern!r} under {root}")
    return matches[0]


def _parse_schedule_summary(path: Path) -> dict[str, int | None]:
    text = path.read_text(encoding="utf-8", errors="replace")

    def _match(label: str) -> int | None:
        match = re.search(rf"{re.escape(label)}:\s+(\d+)", text)
        return int(match.group(1)) if match else None

    return {
        "total_scheduled_bundles": _match("total scheduled bundles"),
        "empty_scheduled_bundles": _match("empty scheduled bundles"),
        "non_empty_scheduled_bundles": _match("non empty scheduled bundles"),
    }


def _collect_dumps(jf_dir: Path, kernel_name: str) -> dict[str, object]:
    final_bundles = _first_file(
        jf_dir,
        f"*{kernel_name}*final_bundles.txt",
        exclude_name_parts=("schedule-analysis", ),
    )
    schedule = _first_file(
        jf_dir, f"*{kernel_name}*schedule-analysis_final_bundles.txt")
    utilization = _first_file(
        jf_dir, f"*{kernel_name}*final_hlo-static-per-bundle-utilization.txt")
    return {
        "final_bundles": str(final_bundles),
        "schedule_analysis_final_bundles": str(schedule),
        "final_hlo_static_per_bundle_utilization": str(utilization),
        "schedule_summary": _parse_schedule_summary(schedule),
    }


def _run_case(*, python: Path, output_root: Path, case_index: int, op: str,
              q_size: int, q_per_kv: int, kv_tokens: int,
              head_dim: int) -> dict[str, object]:
    case_name = f"{case_index:02d}_{_case_id(op, q_size, q_per_kv, kv_tokens, head_dim)}"
    case_root = output_root / case_name
    jf_dir = case_root / "jf"
    mosaic_dir = case_root / "mosaic"
    log_path = case_root / "run.log"
    jf_dir.mkdir(parents=True, exist_ok=True)
    mosaic_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(python),
        str(MICRO_BENCH),
        "--op",
        op,
        "--q-size",
        str(q_size),
        "--q-per-kv",
        str(q_per_kv),
        "--kv-tokens",
        str(kv_tokens),
        "--head-dim",
        str(head_dim),
        "--jf-dump-dir",
        str(jf_dir),
        "--mosaic-dump-dir",
        str(mosaic_dir),
    ]
    timeout_s = int(os.environ.get("PCP_Q_COMPUTE_MICRO_TIMEOUT_S",
                                   15 * 60))
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=_child_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise AssertionError(
            f"{case_name} failed with exit {proc.returncode}. "
            f"Full log: {log_path}\n\n{proc.stdout}")

    result = _parse_micro_result(proc.stdout)
    kernel_name = _kernel_name(op, q_size, q_per_kv, kv_tokens, head_dim)
    return {
        **result,
        "case_name": case_name,
        "case_root": str(case_root),
        "log": str(log_path),
        "dumps": _collect_dumps(jf_dir, kernel_name),
    }


@pytest.mark.skipif(
    os.environ.get("RUN_PCP_Q_COMPUTE_MICRO_LLO") != "1",
    reason="intentional TPU/LLO micro collection; set "
    "RUN_PCP_Q_COMPUTE_MICRO_LLO=1",
)
def test_q_compute_loop_primitives_generate_small_llo_dumps():
    python = Path(os.environ.get("PCP_Q_COMPUTE_MICRO_PYTHON",
                                 sys.executable))
    output_root = Path(
        os.environ.get(
            "PCP_Q_COMPUTE_MICRO_OUTPUT_ROOT",
            str(REPO_ROOT / "data" / "pcp_q_compute_micro_llo" /
                _timestamp()),
        ))
    output_root.mkdir(parents=True, exist_ok=True)

    results = [
        _run_case(
            python=python,
            output_root=output_root,
            case_index=index,
            op=op,
            q_size=q_size,
            q_per_kv=q_per_kv,
            kv_tokens=kv_tokens,
            head_dim=head_dim,
        )
        for index, (op, q_size, q_per_kv, kv_tokens,
                    head_dim) in enumerate(
                        _parse_cases(
                            os.environ.get("PCP_Q_COMPUTE_MICRO_CASES")))
    ]

    summary = {
        "created_utc": _timestamp(),
        "repo_root": str(REPO_ROOT),
        "micro_benchmark": str(MICRO_BENCH),
        "case_format": "op:q_size,q_per_kv,kv_tokens,head_dim",
        "results": results,
    }
    summary_path = output_root / "summary.json"
    hw_path = output_root / "hardware_utilization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True),
                            encoding="utf-8")
    hw_path.write_text(
        json.dumps(
            {
                "created_utc":
                summary["created_utc"],
                "results": [{
                    "case_name": result["case_name"],
                    "op": result["op"],
                    "shape": result["shape"],
                    "static_cycle_proxy_total_scheduled_bundles":
                    result["dumps"]["schedule_summary"][
                        "total_scheduled_bundles"],
                    "final_hlo_static_per_bundle_utilization":
                    result["dumps"][
                        "final_hlo_static_per_bundle_utilization"],
                    "schedule_analysis_final_bundles":
                    result["dumps"]["schedule_analysis_final_bundles"],
                    "final_bundles":
                    result["dumps"]["final_bundles"],
                } for result in results],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"PCP q_compute micro LLO summary: {summary_path}")
    print(f"PCP q_compute micro hardware utilization summary: {hw_path}")
