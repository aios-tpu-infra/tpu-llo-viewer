#!/usr/bin/env python3
"""Generate small Pallas LLO dumps for _q_compute_loop primitives."""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

LLO_FLAGS = (
    "--xla_jf_collect_llo_stack_trace=true",
    "--xla_mosaic_enable_llo_source_annotations=true",
    "--xla_mosaic_enable_dump_debug_info=true",
    "--xla_jf_debug_level=2",
    "--xla_tpu_include_hlo_statistics_in_llo_dump",
    "--xla_tpu_impure_track_debug_metadata",
    "--xla_jf_log_scopes",
    "--xla_jf_line_info_in_symbol_table",
    "--xla_jf_emit_annotations",
    "--xla_jf_module_tracemarks",
    "--xla_jf_dump_debug_info",
    "--xla_tpu_add_llo_regions_to_symbol_table",
    "--xla_tpu_use_enhanced_launch_barrier=true",
    "--xla_jf_lsra_v2_annotate",
)
PATH_FLAG_PREFIXES = (
    "--xla_jf_dump_to=",
    "--xla_mosaic_dump_to=",
)
OP_CHOICES = (
    "qk_dot_general",
    "causal_mask_where",
    "max_axis2",
    "exp_scores",
    "sum_axis2",
    "pv_dot_general",
    "acc_update",
)


def _append_libtpu_init_args(*new_flags: str,
                             replace_prefixes: tuple[str, ...] = ()) -> None:
    existing = os.environ.get("LIBTPU_INIT_ARGS", "")
    flags = [
        flag for flag in existing.split()
        if not any(flag.startswith(prefix) for prefix in replace_prefixes)
    ]
    for flag in new_flags:
        if flag not in flags:
            flags.append(flag)
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(flags)


def _enable_llo_dump(jf_dump_dir: Path, mosaic_dump_dir: Path | None) -> None:
    jf_dump_dir.mkdir(parents=True, exist_ok=True)
    flags = [*LLO_FLAGS, f"--xla_jf_dump_to={jf_dump_dir}"]
    replace_prefixes = [PATH_FLAG_PREFIXES[0]]
    if mosaic_dump_dir is not None:
        mosaic_dump_dir.mkdir(parents=True, exist_ok=True)
        flags.append(f"--xla_mosaic_dump_to={mosaic_dump_dir}")
        replace_prefixes.append(PATH_FLAG_PREFIXES[1])
    _append_libtpu_init_args(*flags,
                             replace_prefixes=tuple(replace_prefixes))


def _kernel_name(op: str, q_size: int, q_per_kv: int, kv_tokens: int,
                 head_dim: int) -> str:
    return (f"pcp_q_compute_micro_{op}_q{q_size}_g{q_per_kv}"
            f"_kv{kv_tokens}_d{head_dim}")


def _grid_spec(in_count: int):
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    return pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[vmem_spec for _ in range(in_count)],
        out_specs=vmem_spec,
        scratch_shapes=(),
        grid=(1, ),
    )


def _compiler_params() -> pltpu.CompilerParams:
    return pltpu.CompilerParams(
        vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes)


def _qk_dot_general_kernel(q_ref, k_ref, out_ref):
    q = q_ref[...].astype(jnp.float32)
    k = k_ref[...].astype(jnp.float32)
    out_ref[...] = lax.dot_general(
        q,
        k,
        (((2, ), (1, )), ((), ())),
        preferred_element_type=jnp.float32,
    )


def _causal_mask_where_kernel(scores_ref, out_ref, *, q_global_start: int,
                              kv_global_start: int, pcp_size: int,
                              page_size: int, q_tile_size: int,
                              kv_valid_len: int, req_id: int):
    scores = scores_ref[...]
    q_size, q_per_kv, kv_tokens = scores.shape
    q_row = lax.broadcasted_iota(jnp.int32, (q_size, q_per_kv, 1), 0)
    q_chunk_idx = lax.div(q_row, page_size)
    q_chunk_offset = lax.rem(q_row, page_size)
    q_pos = q_global_start + q_chunk_idx * pcp_size * page_size + q_chunk_offset

    kv_local_pos = lax.broadcasted_iota(jnp.int32,
                                        (1, 1, kv_tokens),
                                        2)
    kv_page_offset = lax.div(kv_local_pos, page_size)
    kv_token_offset = lax.rem(kv_local_pos, page_size)
    kv_pos = kv_global_start + kv_page_offset * pcp_size * page_size + kv_token_offset

    kv_valid = kv_local_pos < kv_valid_len
    q_valid = q_row < q_tile_size
    row_active = jnp.logical_and(jnp.asarray(req_id != -1), q_valid)
    mask = jnp.logical_and(jnp.logical_and(q_pos >= kv_pos, kv_valid),
                           q_valid)
    masked = jnp.where(mask, scores, -jnp.inf)
    out_ref[...] = jnp.where(row_active, masked, 0.0)


def _max_axis2_kernel(scores_ref, out_ref):
    out_ref[...] = jnp.max(scores_ref[...], axis=2, keepdims=True)


def _exp_scores_kernel(scores_ref, m_ref, out_ref):
    scores = scores_ref[...]
    m = jnp.broadcast_to(m_ref[...], scores.shape)
    out_ref[...] = jnp.exp(scores - m)


def _sum_axis2_kernel(p_ref, out_ref):
    out_ref[...] = jnp.sum(p_ref[...], axis=2, keepdims=True)


def _pv_dot_general_kernel(p_ref, v_ref, out_ref):
    p = p_ref[...]
    v = v_ref[...].astype(jnp.float32)
    out_ref[...] = lax.dot_general(
        p,
        v,
        (((2, ), (0, )), ((), ())),
        preferred_element_type=jnp.float32,
    )


def _acc_update_kernel(alpha_ref, acc_ref, pv_ref, out_ref):
    alpha = jnp.broadcast_to(alpha_ref[...], acc_ref.shape)
    out_ref[...] = alpha * acc_ref[...] + pv_ref[...]


def _make_micro_call(op: str, q_size: int, q_per_kv: int, kv_tokens: int,
                     head_dim: int, *, pcp_size: int, page_size: int):
    name = _kernel_name(op, q_size, q_per_kv, kv_tokens, head_dim)
    if op == "qk_dot_general":
        return pl.pallas_call(
            _qk_dot_general_kernel,
            out_shape=jax.ShapeDtypeStruct((q_size, q_per_kv, kv_tokens),
                                           jnp.float32),
            grid_spec=_grid_spec(2),
            compiler_params=_compiler_params(),
            name=name,
        )
    if op == "causal_mask_where":
        return pl.pallas_call(
            functools.partial(
                _causal_mask_where_kernel,
                q_global_start=126976,
                kv_global_start=126976,
                pcp_size=pcp_size,
                page_size=page_size,
                q_tile_size=q_size,
                kv_valid_len=kv_tokens,
                req_id=0,
            ),
            out_shape=jax.ShapeDtypeStruct((q_size, q_per_kv, kv_tokens),
                                           jnp.float32),
            grid_spec=_grid_spec(1),
            compiler_params=_compiler_params(),
            name=name,
        )
    if op == "max_axis2":
        return pl.pallas_call(
            _max_axis2_kernel,
            out_shape=jax.ShapeDtypeStruct((q_size, q_per_kv, 1),
                                           jnp.float32),
            grid_spec=_grid_spec(1),
            compiler_params=_compiler_params(),
            name=name,
        )
    if op == "exp_scores":
        return pl.pallas_call(
            _exp_scores_kernel,
            out_shape=jax.ShapeDtypeStruct((q_size, q_per_kv, kv_tokens),
                                           jnp.float32),
            grid_spec=_grid_spec(2),
            compiler_params=_compiler_params(),
            name=name,
        )
    if op == "sum_axis2":
        return pl.pallas_call(
            _sum_axis2_kernel,
            out_shape=jax.ShapeDtypeStruct((q_size, q_per_kv, 1),
                                           jnp.float32),
            grid_spec=_grid_spec(1),
            compiler_params=_compiler_params(),
            name=name,
        )
    if op == "pv_dot_general":
        return pl.pallas_call(
            _pv_dot_general_kernel,
            out_shape=jax.ShapeDtypeStruct((q_size, q_per_kv, head_dim),
                                           jnp.float32),
            grid_spec=_grid_spec(2),
            compiler_params=_compiler_params(),
            name=name,
        )
    if op == "acc_update":
        return pl.pallas_call(
            _acc_update_kernel,
            out_shape=jax.ShapeDtypeStruct((q_size, q_per_kv, head_dim),
                                           jnp.float32),
            grid_spec=_grid_spec(3),
            compiler_params=_compiler_params(),
            name=name,
        )
    raise ValueError(f"unsupported op: {op}")


def _make_inputs(op: str, q_size: int, q_per_kv: int, kv_tokens: int,
                 head_dim: int) -> tuple[jax.Array, ...]:
    if op == "qk_dot_general":
        q = jnp.ones((q_size, q_per_kv, head_dim), dtype=jnp.bfloat16) * 0.01
        k = jnp.ones((kv_tokens, head_dim), dtype=jnp.bfloat16) * 0.02
        return q, k
    if op in ("causal_mask_where", "max_axis2", "sum_axis2"):
        return (jnp.ones((q_size, q_per_kv, kv_tokens),
                         dtype=jnp.float32) * 0.01, )
    if op == "exp_scores":
        scores = jnp.ones((q_size, q_per_kv, kv_tokens),
                          dtype=jnp.float32) * 0.01
        m = jnp.ones((q_size, q_per_kv, 1), dtype=jnp.float32) * 0.001
        return scores, m
    if op == "pv_dot_general":
        p = jnp.ones((q_size, q_per_kv, kv_tokens), dtype=jnp.float32) * 0.01
        v = jnp.ones((kv_tokens, head_dim), dtype=jnp.bfloat16) * 0.03
        return p, v
    if op == "acc_update":
        alpha = jnp.ones((q_size, q_per_kv, 1), dtype=jnp.float32) * 0.5
        acc = jnp.ones((q_size, q_per_kv, head_dim), dtype=jnp.float32) * 0.2
        pv = jnp.ones((q_size, q_per_kv, head_dim), dtype=jnp.float32) * 0.3
        return alpha, acc, pv
    raise ValueError(f"unsupported op: {op}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile/run one _q_compute_loop primitive as a tiny "
        "Pallas kernel and dump its LLO.")
    parser.add_argument("--op", choices=OP_CHOICES, required=True)
    parser.add_argument("--q-size", type=int, required=True)
    parser.add_argument("--q-per-kv", type=int, default=16)
    parser.add_argument("--kv-tokens", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--pcp-size", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--jf-dump-dir", type=Path, required=True)
    parser.add_argument("--mosaic-dump-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = _parse_args()
    if args.q_size <= 0 or args.q_per_kv <= 0 or args.kv_tokens <= 0:
        raise ValueError("q-size, q-per-kv and kv-tokens must be positive.")
    if args.head_dim <= 0:
        raise ValueError("head-dim must be positive.")

    _enable_llo_dump(args.jf_dump_dir, args.mosaic_dump_dir)
    devices = jax.local_devices()
    if not devices or devices[0].platform != "tpu":
        raise RuntimeError("This micro LLO benchmark requires a TPU backend.")

    call = _make_micro_call(
        args.op,
        args.q_size,
        args.q_per_kv,
        args.kv_tokens,
        args.head_dim,
        pcp_size=args.pcp_size,
        page_size=args.page_size,
    )
    inputs = _make_inputs(args.op, args.q_size, args.q_per_kv,
                          args.kv_tokens, args.head_dim)

    @jax.jit
    def run_once(*run_inputs):
        return call(*run_inputs)

    compile_start = time.perf_counter()
    compiled = run_once.lower(*inputs).compile()
    compile_s = time.perf_counter() - compile_start

    run_start = time.perf_counter()
    out = compiled(*inputs)
    out.block_until_ready()
    run_s = time.perf_counter() - run_start

    result = {
        "op": args.op,
        "kernel_name": _kernel_name(args.op, args.q_size, args.q_per_kv,
                                    args.kv_tokens, args.head_dim),
        "shape": {
            "q_size": args.q_size,
            "q_per_kv": args.q_per_kv,
            "kv_tokens": args.kv_tokens,
            "head_dim": args.head_dim,
        },
        "output_shape": tuple(int(dim) for dim in out.shape),
        "output_dtype": str(out.dtype),
        "compile_s": compile_s,
        "run_s": run_s,
        "jf_dump_dir": str(args.jf_dump_dir),
        "mosaic_dump_dir": (str(args.mosaic_dump_dir)
                            if args.mosaic_dump_dir is not None else None),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
    }
    print("MICRO_RESULT " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
