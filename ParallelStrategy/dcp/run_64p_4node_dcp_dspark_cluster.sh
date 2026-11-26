#!/usr/bin/env bash
#
# Four-node Kimi-K3 NPU DCP + DSpark launcher.
#
# This script intentionally does not enable prefill context parallelism (PCP).
# It runs a mixed prefill/decode instance with CP=1. DCP_SIZE=1 is accepted as
# the A/B baseline; DCP_SIZE>1 enables DCP without introducing PCP topology or
# synchronization effects.
#
# Examples:
#   ./run_64p_4node_dcp_dspark_cluster.sh config
#   ./run_64p_4node_dcp_dspark_cluster.sh start
#   ./run_64p_4node_dcp_dspark_cluster.sh status
#   ./run_64p_4node_dcp_dspark_cluster.sh stop
#
# The DCP runtime is isolated from PCP and uses the same upstream revisions as
# the community baseline by default:
#   container: k3-dcp
#   SGLang:    /home/q00886407/K3-DCP/sglang-pr37802-17319cb90d
#   kernel:    /home/q00886407/K3-DCP/sgl-kernel-npu-community-base
#   DeepEP:    /home/q00886407/K3-DCP/sgl-kernel-npu-community-base

set -euo pipefail

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Keep orchestration in this directory while allowing a clean worktree to be
# selected as the server-side Python source tree.
REPO_ROOT="${SGLANG_SOURCE_ROOT:-/home/q00886407/K3-DCP/sglang-pr37802-17319cb90d}"

usage() {
    cat >&2 <<EOF
Usage: $0 {start|status|stop|config}

Required topology for this validated path:
  CP_SIZE=1, DCP_SIZE>=1 (1 is the A/B baseline),
  SGLANG_RAGGED_VERIFY_MODE=static. DCP_SIZE>1 requires ASCEND_USE_FIA=1.
  Both DCP_SIZE=1 and DCP_SIZE>1 support NPUGraph.
EOF
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

validate_kda_kernel_api() {
    local python_bin="$1"
    local kernel_python_root="$2"

    SGLANG_EXPECTED_KERNEL_PYTHON_ROOT="${kernel_python_root}" "${python_bin}" -c '
import inspect
import os
from pathlib import Path

from sgl_kernel_npu.fla.kda_prefill import recompute_w_u_fwd_npu
from sgl_kernel_npu.fla.kda_target_verify import kda_target_verify_npu


def validate_keyword_call(function, supplied):
    signature = inspect.signature(function)
    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    required = {
        name
        for name, parameter in parameters.items()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and parameter.default is inspect.Parameter.empty
    }
    missing = required - supplied
    unexpected = set() if accepts_kwargs else supplied - set(parameters)
    if missing or unexpected:
        raise RuntimeError(
            f"Incompatible {function.__name__}{signature}: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return signature


recompute_signature = validate_keyword_call(
    recompute_w_u_fwd_npu,
    {"k", "v", "beta", "A", "gk", "cu_seqlens", "chunk_indices"},
)
target_verify_signature = validate_keyword_call(
    kda_target_verify_npu,
    {
        "A_log",
        "dt_bias",
        "q",
        "k",
        "v",
        "a",
        "b",
        "initial_state_source",
        "initial_state_indices",
        "intermediate_states_buffer",
        "intermediate_state_indices",
        "cache_steps",
        "gates_are_preactivated",
        "lower_bound",
    },
)

source_file = Path(inspect.getsourcefile(recompute_w_u_fwd_npu)).resolve()
expected_root = Path(os.environ["SGLANG_EXPECTED_KERNEL_PYTHON_ROOT"]).resolve()
if expected_root not in source_file.parents:
    raise RuntimeError(
        "sgl_kernel_npu was imported from an unexpected location: "
        f"source={source_file}, expected_under={expected_root}"
    )

print(
    "KDA kernel API preflight OK: "
    f"source={source_file}, recompute={recompute_signature}, "
    f"target_verify={target_verify_signature}"
)
'
}

validate_deepep_runtime() {
    local python_bin="$1"
    local deepep_python_root="$2"
    local deepep_opp_root="$3"

    SGLANG_EXPECTED_DEEPEP_PYTHON_ROOT="${deepep_python_root}" \
    SGLANG_EXPECTED_DEEPEP_OPP_ROOT="${deepep_opp_root}" \
        "${python_bin}" -c '
import os
from pathlib import Path

import deep_ep
import deep_ep_cpp

package_file = Path(deep_ep.__file__).resolve()
extension_file = Path(deep_ep_cpp.__file__).resolve()
expected_python_root = Path(os.environ["SGLANG_EXPECTED_DEEPEP_PYTHON_ROOT"]).resolve()
expected_opp_root = Path(os.environ["SGLANG_EXPECTED_DEEPEP_OPP_ROOT"]).resolve()
opp_paths = [Path(item).resolve() for item in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if item]

if expected_python_root not in package_file.parents:
    raise RuntimeError(
        "deep_ep was imported from stale site-packages: "
        f"source={package_file}, expected_under={expected_python_root}"
    )
if expected_python_root not in extension_file.parents:
    raise RuntimeError(
        "deep_ep_cpp was imported from stale site-packages: "
        f"source={extension_file}, expected_under={expected_python_root}"
    )
if expected_opp_root not in opp_paths:
    raise RuntimeError(
        "rebuilt DeepEP custom OPP is not active: "
        f"expected={expected_opp_root}, actual={opp_paths}"
    )

print(
    "DeepEP runtime preflight OK: "
    f"package={package_file}, extension={extension_file}, opp={expected_opp_root}"
)
'
}

node_main() {
    local node_rank="${1:-}"
    local python_bin="${PYTHON_BIN:-python3}"
    local config_only="${CONFIG_ONLY:-0}"
    local nnodes="${NNODES:-4}"
    local node_ips="${NODE_IPS:-192.168.25.209,192.168.25.212,192.168.25.216,192.168.25.217}"
    local tp_size="${TP_SIZE:-64}"
    local dp_size="${DP_SIZE:-4}"
    local cp_size="${CP_SIZE:-1}"
    local dcp_size="${DCP_SIZE:-2}"
    local model_path="${MODEL_PATH:-/home/weights/Kimi-K3-w4a8-int-moe}"
    local draft_model_path="${DRAFT_MODEL_PATH:-/home/weights/Kimi-K3-DSpark}"
    local net_iface="${NET_IFACE:-enp196s0f0}"
    local dist_port="${DIST_PORT:-30143}"
    local host="${HOST:-0.0.0.0}"
    local port="${PORT:-15010}"
    # An explicitly empty value omits the corresponding CLI option and lets
    # upstream SGLang use its native default. This is useful for reproducing a
    # community launch command without changing the DCP launcher's defaults.
    local page_size="${PAGE_SIZE-}"
    local chunked_prefill_size="${CHUNKED_PREFILL_SIZE:-8192}"
    local max_total_tokens="${MAX_TOTAL_TOKENS-}"
    local max_running_requests="${MAX_RUNNING_REQUESTS:-64}"
    local max_mamba_cache_size="${MAX_MAMBA_CACHE_SIZE-}"
    local mem_fraction_static="${MEM_FRACTION_STATIC:-0.72}"
    local kv_cache_dtype="${KV_CACHE_DTYPE-}"
    local draft_kv_cache_dtype="${DSPARK_DRAFT_KV_CACHE_DTYPE-}"
    # Keep the Base/DCP capacity A/B on the same DSpark attention path.
    local speculative_attention_mode="${DSPARK_ATTENTION_MODE:-decode}"
    local linear_attn_verify_backend="${LINEAR_ATTN_VERIFY_BACKEND:-triton}"
    local disable_custom_all_reduce="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
    local disable_cuda_graph="${DISABLE_CUDA_GRAPH:-0}"
    local cuda_graph_bs="${CUDA_GRAPH_BS:-1 4 16}"
    local ascend_use_fia="${ASCEND_USE_FIA-1}"
    local deepep_mode="${DEEPEP_MODE:-auto}"
    local deep_use_mode
    if [[ "${dcp_size}" == "1" ]]; then
        # The community baseline must not exercise the DCP adaptation's
        # mc2_alltoall DeepEP path.  Keep its MoE routing on the unchanged
        # upstream default strategy so TPOT is not contaminated by that change.
        if [[ -n "${DEEP_USE_MODE:-}" && "${DEEP_USE_MODE}" != "default" ]]; then
            echo "DCP_SIZE=1 community baseline requires DEEP_USE_MODE=default (got ${DEEP_USE_MODE})." >&2
            exit 2
        fi
        deep_use_mode="default"
    else
        deep_use_mode="${DEEP_USE_MODE:-default}"
    fi
    local dspark_block_size="${DSPARK_BLOCK_SIZE:-7}"
    local draft_attention_backend="${DSPARK_DRAFT_ATTENTION_BACKEND:-ascend}"
    local draft_quantization="${DSPARK_DRAFT_QUANTIZATION:-unquant}"
    # Capacity benchmark defaults: let memory profiling size the KV pool,
    # expose its logical capacity, and avoid retained radix-cache tokens.
    local enable_prefix_cache="${ENABLE_PREFIX_CACHE:-0}"
    local enable_metrics="${ENABLE_METRICS:-1}"
    local expert_balancedness_report_mode="${EXPERT_BALANCEDNESS_REPORT_MODE:-off}"
    local expert_distribution_recorder_buffer_size="${EXPERT_DISTRIBUTION_RECORDER_BUFFER_SIZE-}"
    local kernel_code_root="${KERNEL_CODE_ROOT:-/home/q00886407/K3-DCP/sgl-kernel-npu-community-base}"
    local deepep_code_root="${DEEPEP_CODE_ROOT:-/home/q00886407/K3-DCP/sgl-kernel-npu-community-base}"
    local kernel_python_root="${kernel_code_root}/python/sgl_kernel_npu"
    local deepep_python_root="${deepep_code_root}/python/deep_ep"
    local deepep_package_root="${deepep_python_root}/deep_ep"
    local deepep_opp_root="${deepep_package_root}/vendors/hwcomputing"
    local expected_kernel_api_fingerprint="${EXPECTED_KERNEL_API_FINGERPRINT:-c27d78f317d39405540cbe74615f01947a5558527f27dbfd8ac7da90d250965f}"
    local run_tag="${RUN_TAG:-kimi_k3_dcp${dcp_size}_dp${dp_size}_dspark_static_graph}"
    local log_dir="${LOG_DIR:-${REPO_ROOT}/logs/kimi_k3_4node_dcp_dspark}"

    IFS=',' read -r -a node_ip_array <<< "${node_ips}"
    if ! is_positive_integer "${nnodes}" || (( nnodes != 4 )); then
        echo "NNODES must be 4 for this launcher (got ${nnodes})." >&2
        exit 2
    fi
    if (( ${#node_ip_array[@]} != nnodes )); then
        echo "NODE_IPS must contain exactly ${nnodes} comma-separated addresses." >&2
        exit 2
    fi
    if [[ ! "${node_rank}" =~ ^[0-3]$ ]]; then
        echo "Internal node rank must be 0, 1, 2, or 3 (got ${node_rank:-empty})." >&2
        exit 2
    fi
    for value in "${tp_size}" "${dp_size}" "${dcp_size}"; do
        if ! is_positive_integer "${value}"; then
            echo "TP_SIZE, DP_SIZE, and DCP_SIZE must be positive integers." >&2
            exit 2
        fi
    done
    if [[ -n "${page_size}" ]] && ! is_positive_integer "${page_size}"; then
        echo "PAGE_SIZE must be empty or a positive integer." >&2
        exit 2
    fi
    if [[ -n "${max_total_tokens}" ]] && ! is_positive_integer "${max_total_tokens}"; then
        echo "MAX_TOTAL_TOKENS must be empty or a positive integer." >&2
        exit 2
    fi
    if (( cp_size != 1 )); then
        echo "This is a DCP-only validation launcher: CP_SIZE must be 1 (got ${cp_size})." >&2
        exit 2
    fi
    if (( tp_size % nnodes != 0 )); then
        echo "TP_SIZE must be divisible by NNODES." >&2
        exit 2
    fi
    if (( tp_size % dp_size != 0 )); then
        echo "TP_SIZE must be divisible by DP_SIZE." >&2
        exit 2
    fi
    local attn_tp_size=$((tp_size / dp_size))
    if (( attn_tp_size % dcp_size != 0 )); then
        echo "attention-TP (${attn_tp_size}) must be divisible by DCP_SIZE (${dcp_size})." >&2
        exit 2
    fi
    if [[ "${SGLANG_RAGGED_VERIFY_MODE:-static}" != "static" ]]; then
        echo "Kimi-K3 NPU DCP + DSpark currently requires SGLANG_RAGGED_VERIFY_MODE=static." >&2
        exit 2
    fi
    case "${DEEPEP_MODE:-auto}" in
        auto|normal|low_latency) ;;
        *) echo "DEEPEP_MODE must be auto, normal, or low_latency." >&2; exit 2 ;;
    esac
    case "${deep_use_mode}" in
        default|ops|alltoall|mc2_alltoall) ;;
        *) echo "DEEP_USE_MODE must be default, ops, alltoall, or mc2_alltoall." >&2; exit 2 ;;
    esac
    # Kimi-K3 SP-MoE reduce-scatters the DP-local token rows over attention-TP
    # before EP dispatch. Size DeepEP for the per-attention-TP-rank verify
    # workload, matching vLLM-Ascend's MC2 capacity convention.
    local dspark_verify_token_capacity=$((max_running_requests * (dspark_block_size + 1)))
    local dspark_verify_rank_capacity=$((
        (dspark_verify_token_capacity + attn_tp_size - 1) / attn_tp_size
    ))
    local deepep_default_capacity=128
    local deepep_max_dispatch_tokens="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-${deepep_default_capacity}}"
    if ! is_positive_integer "${deepep_max_dispatch_tokens}" || (( deepep_max_dispatch_tokens > 1024 )); then
        echo "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK must be in [1, 1024]." >&2
        exit 2
    fi
    if [[ "${deepep_mode}" != "normal" ]] && (( deepep_max_dispatch_tokens < dspark_verify_rank_capacity )); then
        echo "DeepEP low-latency per-rank capacity (${deepep_max_dispatch_tokens}) is below the SP-MoE DSpark verify requirement (${dspark_verify_rank_capacity}; global=${dspark_verify_token_capacity}, attention-TP=${attn_tp_size})." >&2
        echo "Raise SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK or reduce MAX_RUNNING_REQUESTS/DSPARK_BLOCK_SIZE." >&2
        exit 2
    fi
    # Kimi-K3 low-latency DeepEP V2's stricter Combine tiling uses 7680 bytes
    # per dispatch token and 7168 bytes per combine token with EP=64,
    # 14 local experts and top-k=16. Keep this check aligned with the stricter
    # kernel formula so an undersized HCCL window fails before model execution.
    # Mixed serving also reuses this HCCL group for normal-mode chunked
    # prefill. The custom low-latency operator needs a registered remote
    # window; the alltoall-v strategy does not and uses the ordinary HCCL
    # collective buffer instead.
    # Keep the ordinary HCCL window small. DeepEP/EP64 gets its larger window
    # through the existing DEEPEP_HCCL_BUFFSIZE override. DCP does not need a
    # custom environment variable: with no group-local override it inherits
    # HCCL_BUFFSIZE, just like the other ordinary HCCL process groups.
    local hccl_buffsize="${HCCL_BUFFSIZE:-200}"
    local deepep_hccl_buffsize="${DEEPEP_HCCL_BUFFSIZE:-1800}"
    if ! is_positive_integer "${hccl_buffsize}"; then
        echo "HCCL_BUFFSIZE must be a positive integer in MB." >&2
        exit 2
    fi
    if ! is_positive_integer "${deepep_hccl_buffsize}"; then
        echo "DEEPEP_HCCL_BUFFSIZE must be a positive integer in MB." >&2
        exit 2
    fi
    local deepep_required_buffsize=$((
        deepep_max_dispatch_tokens *
        (7680 * 64 * 14 + 7168 * 16) * 2 /
        (1024 * 1024) + 1
    ))
    if [[ "${deep_use_mode}" != "alltoall" && "${deepep_mode}" != "normal" ]] && (( deepep_hccl_buffsize < deepep_required_buffsize )); then
        echo "DEEPEP_HCCL_BUFFSIZE=${deepep_hccl_buffsize}MB is too small for DeepEP low-latency capacity ${deepep_max_dispatch_tokens}; need at least ${deepep_required_buffsize}MB." >&2
        exit 2
    fi
    case "${enable_prefix_cache}" in
        0|1) ;;
        *) echo "ENABLE_PREFIX_CACHE must be 0 or 1." >&2; exit 2 ;;
    esac
    case "${expert_balancedness_report_mode}" in
        off|server_log|prometheus|both) ;;
        *) echo "EXPERT_BALANCEDNESS_REPORT_MODE must be off, server_log, prometheus, or both." >&2; exit 2 ;;
    esac
    local -a expert_distribution_buffer_args=()
    if [[ -n "${expert_distribution_recorder_buffer_size}" ]]; then
        if [[ "${expert_distribution_recorder_buffer_size}" != "-1" && ! "${expert_distribution_recorder_buffer_size}" =~ ^[1-9][0-9]*$ ]]; then
            echo "EXPERT_DISTRIBUTION_RECORDER_BUFFER_SIZE must be -1 or a positive integer." >&2
            exit 2
        fi
        expert_distribution_buffer_args=(
            --expert-distribution-recorder-buffer-size
            "${expert_distribution_recorder_buffer_size}"
        )
    fi
    case "${config_only}" in
        0|1) ;;
        *) echo "CONFIG_ONLY must be 0 or 1." >&2; exit 2 ;;
    esac
    case "${ascend_use_fia}" in
        ""|0|false|False|FALSE|1|true|True|TRUE) ;;
        *) echo "ASCEND_USE_FIA must be empty or a boolean value." >&2; exit 2 ;;
    esac
    if (( dcp_size > 1 )); then
        case "${ascend_use_fia}" in
            1|true|True|TRUE) ;;
            *) echo "Kimi-K3 NPU DCP_SIZE>1 requires ASCEND_USE_FIA=1." >&2; exit 2 ;;
        esac
    fi
    case "${disable_cuda_graph}" in
        0|1) ;;
        *) echo "DISABLE_CUDA_GRAPH must be 0 or 1." >&2; exit 2 ;;
    esac
    case "${disable_custom_all_reduce}" in
        0|1) ;;
        *) echo "DISABLE_CUSTOM_ALL_REDUCE must be 0 or 1." >&2; exit 2 ;;
    esac
    if [[ ! -f "${kernel_python_root}/sgl_kernel_npu/activation/situ.py" ]]; then
        echo "Invalid KERNEL_CODE_ROOT; activation/situ.py was not found under ${kernel_python_root}." >&2
        echo "The container-installed sgl_kernel_npu is too old for Kimi-K3 ModelSlim MoE." >&2
        exit 2
    fi
    if [[ ! -f "${kernel_python_root}/sgl_kernel_npu/lib/libsgl_kernel_npu.so" ]]; then
        echo "Invalid KERNEL_CODE_ROOT; libsgl_kernel_npu.so was not found under ${kernel_python_root}." >&2
        exit 2
    fi
    if ! compgen -G "${deepep_package_root}/deep_ep_cpp.cpython-311-*.so" >/dev/null; then
        echo "Invalid DEEPEP_CODE_ROOT; the rebuilt deep_ep_cpp CPython 3.11 library was not found under ${deepep_package_root}." >&2
        exit 2
    fi
    if [[ ! -f "${deepep_opp_root}/op_api/lib/libcust_opapi.so" ]]; then
        echo "Invalid DEEPEP_CODE_ROOT; the rebuilt DeepEP custom OPP was not found under ${deepep_opp_root}." >&2
        exit 2
    fi
    if ! grep -Eq 'MOE_EXPERT_MAX_NUM[[:space:]]*=[[:space:]]*1024' \
        "${deepep_code_root}/csrc/deepep/ops/op_host/moe_distribute_dispatch_v2_tiling.cpp"; then
        echo "The selected DeepEP dispatch OPP does not declare support for Kimi-K3's 896 experts." >&2
        exit 2
    fi
    local kernel_package_root="${kernel_python_root}/sgl_kernel_npu"
    local actual_kernel_api_fingerprint
    actual_kernel_api_fingerprint="$(
        (
            cd "${kernel_package_root}"
            sha256sum \
                lib/libsgl_kernel_npu.so \
                fla/utils.py \
                fla/solve_tril.py \
                fla/kda_prefill.py \
                fla/kda_chunk_delta_h.py \
                fla/kda_gate.py \
                fla/kda_target_verify.py \
                mamba/causal_conv1d.py \
                mamba/causal_conv1d_verify.py \
                mamba/mamba_state_update_triton.py \
                mamba/speculative_state_scatter.py \
                activation/situ.py
        ) | sha256sum | awk '{print $1}'
    )"
    if [[ "${actual_kernel_api_fingerprint}" != "${expected_kernel_api_fingerprint}" ]]; then
        echo "sgl-kernel-npu KDA API fingerprint mismatch: expected=${expected_kernel_api_fingerprint}, actual=${actual_kernel_api_fingerprint}." >&2
        echo "Set EXPECTED_KERNEL_API_FINGERPRINT only after validating a deliberate kernel update." >&2
        exit 2
    fi

    if [[ "${config_only}" != "1" ]]; then
        for checkpoint in "${model_path}" "${draft_model_path}"; do
            if [[ ! -f "${checkpoint}/config.json" ]]; then
                echo "Invalid checkpoint; config.json not found: ${checkpoint}" >&2
                exit 2
            fi
        done
    fi

    local dist_init_addr="${DIST_INIT_ADDR:-${node_ip_array[0]}:${dist_port}}"
    local local_world_size=$((tp_size / nnodes))
    local log_stamp
    log_stamp="$(date '+%Y-%m-%d_%H-%M-%S')"
    local log_file="${log_dir}/${run_tag}_rank${node_rank}_${log_stamp}.log"

    local fia_label=0
    case "${ascend_use_fia}" in
        1|true|True|TRUE)
            export ASCEND_USE_FIA=1
            fia_label=1
            ;;
        *) unset ASCEND_USE_FIA ;;
    esac
    export SGLANG_RAGGED_VERIFY_MODE=static
    export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
    export HCCL_SOCKET_IFNAME="${net_iface}"
    export GLOO_SOCKET_IFNAME="${net_iface}"
    export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-18000}"
    export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-18000-18100}"
    export HCCL_BUFFSIZE="${hccl_buffsize}"
    export DEEPEP_HCCL_BUFFSIZE="${deepep_hccl_buffsize}"
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${deepep_max_dispatch_tokens}"
    # Match vLLM-Ascend's A3 policy: in mc2_alltoall mode, mixed/prefill extend
    # batches use dynamic AllToAllV and decode-only batches use MC2.  MC2 leaves
    # comm_alg unset by default; hierarchy is not valid for K3's 896 experts.
    if [[ "${deep_use_mode}" == "default" ]]; then
        unset DEEP_USE_MODE
    else
        export DEEP_USE_MODE="${deep_use_mode}"
    fi
    if [[ -n "${DEEPEP_MC2_COMM_ALG:-}" ]]; then
        export DEEPEP_MC2_COMM_ALG
    else
        unset DEEPEP_MC2_COMM_ALG
    fi
    unset MOE_ENABLE_TOPK_NEG_ONE
    export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
    export DEEP_NORMAL_MODE_USE_INT8_QUANT="${DEEP_NORMAL_MODE_USE_INT8_QUANT:-1}"
    export DEEPEP_NORMAL_LONG_SEQ_ROUND="${DEEPEP_NORMAL_LONG_SEQ_ROUND:-64}"
    export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS="${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-512}"
    export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
    export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
    export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS="${SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS:-1}"

    unset SGLANG_NPU_FUSED_MOE_MODE
    unset ENABLE_PROFILING
    unset SGLANG_K3_TRACE_STATE_FILE SGLANG_K3_TRACE_HIDDEN_FILE
    unset ASCEND_RT_VISIBLE_DEVICES

    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
    export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,${node_ips}"
    export no_proxy="${NO_PROXY}"
    if [[ "${ASCEND_LAUNCH_BLOCKING:-0}" == "1" ]]; then
        export ASCEND_LAUNCH_BLOCKING=1
    else
        unset ASCEND_LAUNCH_BLOCKING
    fi
    unset SGLANG_ENABLE_CP_V2
    # Speculative V2 is unconditional in this SGLang revision; the retired
    # SGLANG_ENABLE_SPEC_V2 switch only emits a warning.
    unset SGLANG_ENABLE_SPEC_V2
    unset SGLANG_MAMBA_SSM_DTYPE SGLANG_MAMBA_CONV_DTYPE

    local -a mamba_cache_args=()
    if [[ -n "${max_mamba_cache_size}" ]]; then
        if ! is_positive_integer "${max_mamba_cache_size}"; then
            echo "MAX_MAMBA_CACHE_SIZE must be empty or a positive integer." >&2
            exit 2
        fi
        mamba_cache_args=(--max-mamba-cache-size "${max_mamba_cache_size}")
    fi

    local -a page_size_args=()
    if [[ -n "${page_size}" ]]; then
        page_size_args=(--page-size "${page_size}")
    fi

    local -a max_total_tokens_args=()
    if [[ -n "${max_total_tokens}" ]]; then
        max_total_tokens_args=(--max-total-tokens "${max_total_tokens}")
    fi

    local -a kv_cache_dtype_args=()
    if [[ -n "${kv_cache_dtype}" ]]; then
        kv_cache_dtype_args=(--kv-cache-dtype "${kv_cache_dtype}")
    fi

    local -a draft_kv_cache_dtype_args=()
    if [[ -n "${draft_kv_cache_dtype}" ]]; then
        draft_kv_cache_dtype_args=(--speculative-draft-kv-cache-dtype "${draft_kv_cache_dtype}")
    fi

    local -a speculative_attention_mode_args=()
    if [[ -n "${speculative_attention_mode}" ]]; then
        speculative_attention_mode_args=(--speculative-attention-mode "${speculative_attention_mode}")
    fi

    local -a linear_verify_args=()
    if [[ -n "${linear_attn_verify_backend}" ]]; then
        linear_verify_args=(--linear-attn-verify-backend "${linear_attn_verify_backend}")
    fi

    local -a custom_all_reduce_args=()
    if [[ "${disable_custom_all_reduce}" == "1" ]]; then
        custom_all_reduce_args=(--disable-custom-all-reduce)
    fi

    local -a graph_args=()
    if [[ "${disable_cuda_graph}" == "1" ]]; then
        graph_args=(--disable-cuda-graph)
    elif [[ -n "${cuda_graph_bs}" ]]; then
        local -a graph_batch_sizes=()
        read -r -a graph_batch_sizes <<< "${cuda_graph_bs}"
        local graph_bs
        for graph_bs in "${graph_batch_sizes[@]}"; do
            if ! is_positive_integer "${graph_bs}"; then
                echo "CUDA_GRAPH_BS must contain positive integers." >&2
                exit 2
            fi
        done
        # Newer SGLang splits graph batch sizes by phase.  The legacy
        # --cuda-graph-bs setting used by this launcher controls decode graph
        # capture, so bind it explicitly to the decode phase.
        graph_args=(--cuda-graph-bs-decode "${graph_batch_sizes[@]}")
    fi

    local -a cache_args=()
    if [[ "${enable_prefix_cache}" == "0" ]]; then
        cache_args=(--disable-radix-cache)
    fi

    local -a metrics_args=()
    if [[ "${enable_metrics}" == "1" ]]; then
        metrics_args=(--enable-metrics)
    elif [[ "${enable_metrics}" != "0" ]]; then
        echo "ENABLE_METRICS must be 0 or 1 (got ${enable_metrics})." >&2
        exit 2
    fi

    local -a dcp_args=(--dcp-size "${dcp_size}")
    local dcp_comm_label="disabled (baseline)"
    if (( dcp_size > 1 )); then
        dcp_args+=(--dcp-comm-backend a2a)
        dcp_comm_label="a2a"
    fi

    local graph_label="disabled"
    if [[ "${disable_cuda_graph}" == "0" ]]; then
        graph_label="enabled"
        if [[ -n "${cuda_graph_bs}" ]]; then
            graph_label+=" (bs=${cuda_graph_bs})"
        fi
    fi

    local -a server_args=(
        --model-loader-extra-config '{"enable_multithread_load": true}'
        --dist-init-addr "${dist_init_addr}"
        --nnodes "${nnodes}"
        --node-rank "${node_rank}"
        --model-path "${model_path}"
        --tokenizer-path "${model_path}"
        --trust-remote-code
        --attention-backend ascend
        --device npu
        --quantization modelslim
        --dtype bfloat16
        "${kv_cache_dtype_args[@]}"
        --tp-size "${tp_size}"
        --enable-dp-attention
        --dp-size "${dp_size}"
        --enable-dp-lm-head
        # The current K3 EP-a2a default replicates every shared-expert MLP on
        # each rank. Ascend's proven layout shards these weights over attn-TP
        # and gathers/scatters only the shared branch, avoiding a ~12 GB/rank
        # weight regression seen during DCP bring-up.
        --enable-shared-experts-attn-tp
        --enable-dense-mlp-attn-tp
        "${dcp_args[@]}"
        "${page_size_args[@]}"
        --mem-fraction-static "${mem_fraction_static}"
        --chunked-prefill-size "${chunked_prefill_size}"
        "${max_total_tokens_args[@]}"
        --max-running-requests "${max_running_requests}"
        "${mamba_cache_args[@]}"
        --reasoning-parser kimi_k3
        --moe-a2a-backend deepep
        --deepep-mode "${deepep_mode}"
        --expert-balancedness-report-mode "${expert_balancedness_report_mode}"
        "${expert_distribution_buffer_args[@]}"
        "${cache_args[@]}"
        "${metrics_args[@]}"
        "${graph_args[@]}"
        "${custom_all_reduce_args[@]}"
        --speculative-algorithm DSPARK
        --speculative-draft-model-path "${draft_model_path}"
        --speculative-dspark-block-size "${dspark_block_size}"
        "${speculative_attention_mode_args[@]}"
        --speculative-draft-attention-backend "${draft_attention_backend}"
        "${draft_kv_cache_dtype_args[@]}"
        "${linear_verify_args[@]}"
        --speculative-eagle-topk 1
        --speculative-draft-model-quantization "${draft_quantization}"
        --watchdog-timeout 9000
        --host "${host}"
        --port "${port}"
    )

    echo "Kimi-K3 four-node NPU DCP + DSpark:"
    echo "  rank=${node_rank}/${nnodes}, local-world=${local_world_size}"
    echo "  TP=${tp_size}, DP=${dp_size}, CP=1, attention-TP=${attn_tp_size}, DCP=${dcp_size}"
    echo "  target=${model_path}, draft=${draft_model_path}, block=${dspark_block_size}"
    echo "  target-KV=${kv_cache_dtype:-upstream-default}, draft-KV=${draft_kv_cache_dtype:-upstream-default}, FIA=${fia_label}, ragged=${SGLANG_RAGGED_VERIFY_MODE}"
    echo "  DCP-comm=${dcp_comm_label}, DeepEP-strategy=${DEEP_USE_MODE:-default}, DeepEP-capacity=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK}"
    echo "  HCCL-window=${HCCL_BUFFSIZE}MB, DeepEP-window=${DEEPEP_HCCL_BUFFSIZE}MB"
    echo "  HCCL-expansion=${HCCL_OP_EXPANSION_MODE}, graph=${graph_label}, prefix-cache=${enable_prefix_cache}, metrics=${enable_metrics}"
    echo "  dist=${dist_init_addr}, service=${host}:${port}, interface=${net_iface}"
    echo "  source=${REPO_ROOT}/python"
    echo "  kernel-source=${kernel_python_root}"
    echo "  deepep-source=${deepep_python_root}"

    if [[ "${config_only}" == "1" ]]; then
        printf '  command:'
        printf ' %q' "${python_bin}" -m sglang.launch_server "${server_args[@]}"
        printf '\n'
        return 0
    fi

    mkdir -p "${log_dir}"
    for env_script in \
        /usr/local/Ascend/ascend-toolkit/set_env.sh \
        /usr/local/Ascend/nnal/atb/set_env.sh; do
        if [[ -f "${env_script}" ]]; then
            set +u
            # shellcheck disable=SC1090
            source "${env_script}"
            set -u
        fi
    done
    # Load the rebuilt DeepEP package and its bundled custom OPP. Merely adding
    # sgl_kernel_npu to PYTHONPATH leaves the container's stale site-packages
    # DeepEP active (its dispatch tiling is limited to 512 experts).
    export ASCEND_CUSTOM_OPP_PATH="${deepep_opp_root}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
    export LD_LIBRARY_PATH="${deepep_opp_root}/op_api/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    # deep_ep currently imports deep_ep_cpp as a top-level extension module,
    # so both the package parent and the package directory must precede the
    # container's stale site-packages paths.
    export PYTHONPATH="${REPO_ROOT}/python:${kernel_python_root}:${deepep_python_root}:${deepep_package_root}:${PYTHONPATH:-}"
    validate_deepep_runtime "${python_bin}" "${deepep_python_root}" "${deepep_opp_root}"
    # The exact KDA Python sources were already checked above with the pinned
    # fingerprint.  Do not import torch_npu a second time in a throwaway
    # process here: on the four-node Ascend launcher that short-lived runtime
    # can be terminated during teardown before launch_server is exec'd.  The
    # real server process performs the authoritative KDA import immediately
    # afterwards and reports any import/API error in the service log.
    echo "KDA kernel source preflight OK: fingerprint=${actual_kernel_api_fingerprint}"

    set +e
    "${python_bin}" -m sglang.launch_server "${server_args[@]}" 2>&1 | tee "${log_file}"
    local server_rc="${PIPESTATUS[0]}"
    set -e
    return "${server_rc}"
}

if [[ "${1:-}" == "_node" ]]; then
    node_main "${2:-}"
    exit $?
fi

ACTION="${1:-}"
case "${ACTION}" in
    start|status|stop|config) ;;
    *) usage; exit 2 ;;
esac

if [[ "${ACTION}" == "config" ]]; then
    CONFIG_ONLY=1 node_main 0
    exit $?
fi

REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-${SCRIPT_ROOT}}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-${REMOTE_REPO_ROOT}/logs/kimi_k3_4node_dcp_dspark/control}"
NODE_IPS="${NODE_IPS:-192.168.25.209,192.168.25.212,192.168.25.216,192.168.25.217}"
IFS=',' read -r -a NODES <<< "${NODE_IPS}"
if (( ${#NODES[@]} != 4 )); then
    echo "NODE_IPS must contain exactly four comma-separated addresses." >&2
    exit 2
fi

CONTAINER_NAME="${CONTAINER_NAME:-k3-dcp}"
SSH_USER="${SSH_USER:-root}"
SSH_BIN="${SSH_BIN:-ssh}"
SSH_KEY="${SSH_KEY:-}"
LOCAL_IPS=" $(hostname -I 2>/dev/null || true) "

run_on() {
    local remote_host="$1"
    shift
    local command="$*"

    if [[ -n "${CONTAINER_NAME}" ]]; then
        printf -v command '%q ' docker exec "${CONTAINER_NAME}" bash -c "${command}"
    fi

    if [[ "${LOCAL_IPS}" == *" ${remote_host} "* ]]; then
        bash -c "${command}"
    else
        local -a ssh_args=(
            -o BatchMode=yes
            -o ConnectTimeout="${SSH_CONNECT_TIMEOUT:-10}"
            -o StrictHostKeyChecking=accept-new
        )
        if [[ -n "${SSH_KEY}" ]]; then
            ssh_args+=(
                -i "${SSH_KEY}"
                -o IdentitiesOnly=yes
            )
        fi
        printf '%s\n' "${command}" | \
            "${SSH_BIN}" "${ssh_args[@]}" \
                "${SSH_USER}@${remote_host}" bash -s
    fi
}

status_command() {
    local rank="$1"
    local pid_file="${REMOTE_LOG_DIR}/server_rank${rank}.pid"
    local exit_file="${REMOTE_LOG_DIR}/server_rank${rank}.exit"
    printf '%s' \
        "pid_file=$(printf '%q' "${pid_file}"); exit_file=$(printf '%q' "${exit_file}"); " \
        'pid=$(cat "${pid_file}" 2>/dev/null || true); ' \
        'if [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null; then echo "RUNNING pid=${pid}"; ' \
        'else rc=$(cat "${exit_file}" 2>/dev/null || true); if [[ "${rc}" =~ ^[0-9]+$ ]]; then echo "STOPPED rc=${rc}"; else echo "STOPPED rc=unknown"; fi; fi'
}

stop_command() {
    local rank="$1"
    local pid_file="${REMOTE_LOG_DIR}/server_rank${rank}.pid"
    local pgid_file="${REMOTE_LOG_DIR}/server_rank${rank}.pgid"
    local exit_file="${REMOTE_LOG_DIR}/server_rank${rank}.exit"
    printf '%s' \
        "pid_file=$(printf '%q' "${pid_file}"); pgid_file=$(printf '%q' "${pgid_file}"); exit_file=$(printf '%q' "${exit_file}"); " \
        'pid=$(cat "${pid_file}" 2>/dev/null || true); pgid=$(cat "${pgid_file}" 2>/dev/null || true); ' \
        'if [[ "${pgid}" =~ ^[1-9][0-9]*$ ]]; then kill -TERM -- "-${pgid}" 2>/dev/null || true; ' \
        'for _ in {1..20}; do [[ ! "${pid}" =~ ^[1-9][0-9]*$ ]] || ! kill -0 "${pid}" 2>/dev/null || { sleep 0.5; continue; }; break; done; ' \
        'if [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null; then kill -KILL -- "-${pgid}" 2>/dev/null || true; fi; fi; ' \
        'rm -f "${pid_file}" "${pgid_file}" "${exit_file}"; echo STOPPED'
}

start_available_command() {
    local rank="$1"
    local lock_file="${REMOTE_LOG_DIR}/server_rank${rank}.lock"
    printf '%s' \
        "mkdir -p $(printf '%q' "${REMOTE_LOG_DIR}"); " \
        "exec 8>$(printf '%q' "${lock_file}"); " \
        "if ! flock -n 8; then echo 'rank=${rank} is already running; use stop before start' >&2; exit 73; fi"
}

if [[ "${ACTION}" == "status" || "${ACTION}" == "stop" ]]; then
    for rank in 0 1 2 3; do
        node_host="${NODES[$rank]}"
        printf 'rank=%s host=%s: ' "${rank}" "${node_host}"
        if [[ "${ACTION}" == "status" ]]; then
            run_on "${node_host}" "$(status_command "${rank}")"
        else
            run_on "${node_host}" "$(stop_command "${rank}")"
        fi
    done
    exit 0
fi

# Fail before creating any distributed workers when a selected source worktree
# has not been synchronized into every node's container.
source_marker="${REPO_ROOT}/python/sglang/__init__.py"
for rank in 0 1 2 3; do
    node_host="${NODES[$rank]}"
    if ! run_on "${node_host}" \
        "test -r $(printf '%q' "${source_marker}") || { echo $(printf '%q' "Missing SGLANG_SOURCE_ROOT on rank=${rank}: ${source_marker}") >&2; exit 2; }"; then
        echo "Start aborted: source tree is unavailable on rank=${rank} host=${node_host}." >&2
        exit 2
    fi
done

# Serialize the four-node orchestration and refuse to replace a live service.
# A restart must be explicit: `stop` followed by `start`.
mkdir -p "${REMOTE_LOG_DIR}"
exec 9>"${REMOTE_LOG_DIR}/cluster_start.lock"
if ! flock -n 9; then
    echo "Another DCP cluster start is already in progress." >&2
    exit 73
fi
for rank in 0 1 2 3; do
    node_host="${NODES[$rank]}"
    if ! run_on "${node_host}" "$(start_available_command "${rank}")"; then
        echo "Start aborted: rank=${rank} host=${node_host} already has a live launcher." >&2
        exit 73
    fi
done

controller_dcp_size="${DCP_SIZE:-2}"
if [[ "${controller_dcp_size}" == "1" ]]; then
    if [[ -n "${DEEP_USE_MODE:-}" && "${DEEP_USE_MODE}" != "default" ]]; then
        echo "DCP_SIZE=1 community baseline requires DEEP_USE_MODE=default (got ${DEEP_USE_MODE})." >&2
        exit 2
    fi
    controller_deep_use_mode="default"
else
    controller_deep_use_mode="${DEEP_USE_MODE:-default}"
fi
controller_hccl_buffsize="${HCCL_BUFFSIZE:-200}"
controller_deepep_hccl_buffsize="${DEEPEP_HCCL_BUFFSIZE:-1800}"

COMMON_ENV=(
    "SGLANG_SOURCE_ROOT=${REPO_ROOT}"
    "NODE_IPS=${NODE_IPS}"
    "NNODES=4"
    "TP_SIZE=${TP_SIZE:-64}"
    "DP_SIZE=${DP_SIZE:-4}"
    "CP_SIZE=1"
    "DCP_SIZE=${controller_dcp_size}"
    "DIST_PORT=${DIST_PORT:-30143}"
    "PORT=${PORT:-15010}"
    "PAGE_SIZE=${PAGE_SIZE-}"
    "CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}"
    "MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS-}"
    "MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-64}"
    "MAX_MAMBA_CACHE_SIZE=${MAX_MAMBA_CACHE_SIZE-}"
    "MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.72}"
    "KV_CACHE_DTYPE=${KV_CACHE_DTYPE-}"
    "DSPARK_DRAFT_KV_CACHE_DTYPE=${DSPARK_DRAFT_KV_CACHE_DTYPE-}"
    "DSPARK_ATTENTION_MODE=${DSPARK_ATTENTION_MODE:-decode}"
    "LINEAR_ATTN_VERIFY_BACKEND=${LINEAR_ATTN_VERIFY_BACKEND:-triton}"
    "DISABLE_CUSTOM_ALL_REDUCE=${DISABLE_CUSTOM_ALL_REDUCE:-1}"
    "HCCL_BUFFSIZE=${controller_hccl_buffsize}"
    "DEEPEP_HCCL_BUFFSIZE=${controller_deepep_hccl_buffsize}"
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
    "DEEP_USE_MODE=${controller_deep_use_mode}"
    "DEEPEP_MC2_COMM_ALG=${DEEPEP_MC2_COMM_ALG:-}"
    "PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
    "ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING:-0}"
    "HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-18000}"
    "HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-18000-18100}"
    "DEEPEP_MODE=${DEEPEP_MODE:-auto}"
    "HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}"
    "SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE=${SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE:-1}"
    "SGLANG_K3_SHARED_EXPERTS_ATTN_TP=${SGLANG_K3_SHARED_EXPERTS_ATTN_TP:-1}"
    "SGLANG_K3_DENSE_MLP_ATTN_TP=${SGLANG_K3_DENSE_MLP_ATTN_TP:-1}"
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-1}"
    "SGLANG_DSPARK_FOLDED_PROPOSAL=${SGLANG_DSPARK_FOLDED_PROPOSAL:-0}"
    "SGLANG_DSPARK_FOLDED_SAMPLING=${SGLANG_DSPARK_FOLDED_SAMPLING:-0}"
    "SGLANG_DSPARK_STACKED_CTX_KV=${SGLANG_DSPARK_STACKED_CTX_KV:-0}"
    "SGLANG_DSPARK_EMBED_IN_GRAPH=${SGLANG_DSPARK_EMBED_IN_GRAPH:-0}"
    "DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-0}"
    "CUDA_GRAPH_BS=${CUDA_GRAPH_BS:-1 4 16}"
    "ENABLE_PREFIX_CACHE=${ENABLE_PREFIX_CACHE:-0}"
    "ENABLE_METRICS=${ENABLE_METRICS:-1}"
    "EXPERT_BALANCEDNESS_REPORT_MODE=${EXPERT_BALANCEDNESS_REPORT_MODE:-off}"
    "EXPERT_DISTRIBUTION_RECORDER_BUFFER_SIZE=${EXPERT_DISTRIBUTION_RECORDER_BUFFER_SIZE-}"
    "ASCEND_USE_FIA=${ASCEND_USE_FIA-1}"
    "SGLANG_RAGGED_VERIFY_MODE=static"
    "DSPARK_BLOCK_SIZE=${DSPARK_BLOCK_SIZE:-7}"
    "DSPARK_DRAFT_ATTENTION_BACKEND=${DSPARK_DRAFT_ATTENTION_BACKEND:-ascend}"
    "DSPARK_DRAFT_QUANTIZATION=${DSPARK_DRAFT_QUANTIZATION:-unquant}"
    "EXPECTED_KERNEL_API_FINGERPRINT=${EXPECTED_KERNEL_API_FINGERPRINT:-c27d78f317d39405540cbe74615f01947a5558527f27dbfd8ac7da90d250965f}"
    "RUN_TAG=${RUN_TAG:-kimi_k3_dcp${DCP_SIZE:-2}_dp${DP_SIZE:-4}_dspark_static_graph}"
)
[[ -z "${DIST_INIT_ADDR:-}" ]] || COMMON_ENV+=("DIST_INIT_ADDR=${DIST_INIT_ADDR}")
[[ -z "${PYTHON_BIN:-}" ]] || COMMON_ENV+=("PYTHON_BIN=${PYTHON_BIN}")

MODEL_PATH="${MODEL_PATH:-/home/weights/Kimi-K3-w4a8-int-moe}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/home/weights/Kimi-K3-DSpark}"
NET_IFACE="${NET_IFACE:-enp196s0f0}"
KERNEL_CODE_ROOT="${KERNEL_CODE_ROOT:-/home/q00886407/K3-DCP/sgl-kernel-npu-community-base}"
DEEPEP_CODE_ROOT="${DEEPEP_CODE_ROOT:-/home/q00886407/K3-DCP/sgl-kernel-npu-community-base}"

echo "Launching Kimi-K3 DCP + DSpark: nodes=${NODE_IPS}, DCP=${DCP_SIZE:-2}, DP=${DP_SIZE:-4}, CP=1"
for rank in 1 2 3 0; do
    node_host="${NODES[$rank]}"
    model_var="MODEL_PATH_RANK${rank}"
    draft_var="DRAFT_MODEL_PATH_RANK${rank}"
    net_iface_var="NET_IFACE_RANK${rank}"
    kernel_var="KERNEL_CODE_ROOT_RANK${rank}"
    deepep_var="DEEPEP_CODE_ROOT_RANK${rank}"
    rank_model_path="${MODEL_PATH}"
    rank_draft_model_path="${DRAFT_MODEL_PATH}"
    rank_net_iface="${NET_IFACE}"
    rank_kernel_code_root="${KERNEL_CODE_ROOT}"
    rank_deepep_code_root="${DEEPEP_CODE_ROOT}"
    [[ -z "${!model_var:-}" ]] || rank_model_path="${!model_var}"
    [[ -z "${!draft_var:-}" ]] || rank_draft_model_path="${!draft_var}"
    [[ -z "${!net_iface_var:-}" ]] || rank_net_iface="${!net_iface_var}"
    [[ -z "${!kernel_var:-}" ]] || rank_kernel_code_root="${!kernel_var}"
    [[ -z "${!deepep_var:-}" ]] || rank_deepep_code_root="${!deepep_var}"

    env_args=(
        "${COMMON_ENV[@]}"
        "MODEL_PATH=${rank_model_path}"
        "DRAFT_MODEL_PATH=${rank_draft_model_path}"
        "NET_IFACE=${rank_net_iface}"
        "KERNEL_CODE_ROOT=${rank_kernel_code_root}"
        "DEEPEP_CODE_ROOT=${rank_deepep_code_root}"
    )

    pid_file="${REMOTE_LOG_DIR}/server_rank${rank}.pid"
    pgid_file="${REMOTE_LOG_DIR}/server_rank${rank}.pgid"
    exit_file="${REMOTE_LOG_DIR}/server_rank${rank}.exit"
    lock_file="${REMOTE_LOG_DIR}/server_rank${rank}.lock"
    launcher_log="${REMOTE_LOG_DIR}/dcp_rank${rank}_launcher.log"

    printf -v node_command '%q ' \
        env "${env_args[@]}" "./${SCRIPT_NAME}" _node "${rank}"
    printf -v pid_file_q '%q' "${pid_file}"
    printf -v pgid_file_q '%q' "${pgid_file}"
    printf -v exit_file_q '%q' "${exit_file}"
    printf -v lock_file_q '%q' "${lock_file}"
    wrapper_command="exec 8>${lock_file_q}; flock -n 8 || { echo 'rank=${rank} is already running; refusing duplicate start' >&2; exit 73; }; rm -f ${exit_file_q}; printf '%s\\n' \"\$\$\" >${pid_file_q}; pgid=\$(ps -o pgid= -p \"\$\$\" 2>/dev/null | tr -d ' ' || true); printf '%s\\n' \"\${pgid}\" >${pgid_file_q}; ${node_command}; rc=\$?; printf '%s\\n' \"\${rc}\" >${exit_file_q}; exit \"\${rc}\""
    printf -v detached_command '%q ' nohup setsid bash -c "${wrapper_command}"

    command="cd $(printf '%q' "${REMOTE_REPO_ROOT}") || exit 1; "
    command+="mkdir -p $(printf '%q' "${REMOTE_LOG_DIR}"); "
    command+="${detached_command}>>$(printf '%q' "${launcher_log}") 2>&1 < /dev/null &"

    run_on "${node_host}" "${command}"
    echo "rank=${rank} host=${node_host} STARTED log=${launcher_log}"
done

echo "All four launch commands were sent."
echo "Rank-0 launcher log: ${REMOTE_LOG_DIR}/dcp_rank0_launcher.log"
echo "Service URL after model loading: http://${NODES[0]}:${PORT:-15010}"
