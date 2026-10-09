#!/usr/bin/env bash
# Run the Simt-family ST oneshots under ST_VF_MODE=simt and/or simd.
# Same kernels/T.Parallel source; vf_region picks T.SimtVF or T.SimdVF.
# kernels_ptodsl/ is not part of this regress.
#
# SimdVF may FAIL some cases. The script still prints a PASS/FAIL table
# and exits non-zero only when an oneshot script itself fails (missing env,
# etc.), not because a case cell is FAIL.
#
# Default (ST_VF_MODE and ST_VF_MODES unset): both modes.
# ST_VF_MODES="simd"           one or more modes (comma or space separated)
# ST_VF_MODE=simt              single mode when ST_VF_MODES is unset
# ST_VF_ONESHOTS="oneshot_sv1_sv9.sh"   subset of family oneshots
# ST_VF_REGRESS_OUT=/tmp/...   root; each mode writes $root/<mode>/
# ST_VF_SUMMARIZE_ONLY=1       reprint the table, skip oneshots
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)

if [[ -z "${PY:-}" && -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
fi
TABLE_PY="${PY:-python3}"

canon_mode() {
  local raw="${1,,}"
  case "$raw" in
    simt|simtvf) printf '%s\n' simt ;;
    simd|simdvf) printf '%s\n' simd ;;
    *)
      echo "error: unknown VF mode '$1' (expected simt|simd or alias simtvf|simdvf)" >&2
      return 1
      ;;
  esac
}

if [[ -n "${ST_VF_MODES:-}" ]]; then
  normalized="${ST_VF_MODES//,/ }"
  # shellcheck disable=SC2206
  RAW_MODES=($normalized)
elif [[ -n "${ST_VF_MODE:-}" ]]; then
  RAW_MODES=("$ST_VF_MODE")
else
  RAW_MODES=(simt simd)
fi

declare -A SEEN=()
MODES=()
for raw in "${RAW_MODES[@]}"; do
  [[ -z "$raw" ]] && continue
  mode=$(canon_mode "$raw")
  if [[ -n "${SEEN[$mode]:-}" ]]; then
    continue
  fi
  SEEN[$mode]=1
  MODES+=("$mode")
done
if [[ "${#MODES[@]}" -eq 0 ]]; then
  echo "error: no VF modes to run" >&2
  exit 1
fi

if [[ -n "${ST_VF_ONESHOTS:-}" ]]; then
  normalized_scripts="${ST_VF_ONESHOTS//,/ }"
  # shellcheck disable=SC2206
  ONESHOTS=($normalized_scripts)
else
  ONESHOTS=(oneshot_sv1_sv9.sh oneshot_cf1_cf6.sh oneshot_sp1_sp6.sh)
fi
for script in "${ONESHOTS[@]}"; do
  if [[ ! -f "$HERE/$script" ]]; then
    echo "error: oneshot not found: $HERE/$script" >&2
    exit 1
  fi
done

BASE="${ST_VF_REGRESS_OUT:-/tmp/t_parallel_st_suite_vf_mode}"
mkdir -p "$BASE"
LOG="$BASE/oneshot_vf_mode_regress.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== VF mode regress $(date -Is) modes=${MODES[*]} oneshots=${ONESHOTS[*]} OUT=$BASE ==="
echo "SimdVF FAIL cells are recorded. This regress does not require every case to PASS."

rc_all=0
if [[ "${ST_VF_SUMMARIZE_ONLY:-0}" == "1" ]]; then
  echo "ST_VF_SUMMARIZE_ONLY=1 — skipping oneshots"
else
  for mode in "${MODES[@]}"; do
    export ST_VF_MODE="$mode"
    export ST_SIMTVF_OUT="$BASE/$mode"
    export ST_SIMTVF_REPORTS="$ST_SIMTVF_OUT/reports"
    mkdir -p "$ST_SIMTVF_OUT"
    for script in "${ONESHOTS[@]}"; do
      echo "==== VF_MODE=$mode BEGIN $script $(date -Is) ===="
      set +e
      bash "$HERE/$script"
      rc=$?
      set -e
      echo "==== VF_MODE=$mode END $script rc=$rc ===="
      if [[ "$rc" -ne 0 ]]; then
        rc_all=$rc
      fi
    done
  done
fi

echo "==== VF_MODE PASS/FAIL TABLE ===="
set +e
"$TABLE_PY" "$HERE/vf_mode_pass_table.py" --root "$BASE"
table_rc=$?
set -e
if [[ "$table_rc" -ne 0 && "$rc_all" -eq 0 ]]; then
  rc_all=$table_rc
fi
echo "=== VF mode regress DONE rc=$rc_all $(date -Is) ==="
exit "$rc_all"
