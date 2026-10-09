#!/usr/bin/env bash
# Run the Simt-family ST oneshots under the four ST_VF_MODE frame×target pairs.
# Same kernels/T.Parallel source; vf_region picks T.SimtVF or T.SimdVF and
# compile_target picks ascend or pto. kernels_ptodsl/ is not part of this regress.
#
#   simt-asc   T.SimtVF  target=ascend   (aliases: simt, simtvf, simt-ascend)
#   simt-pto   T.SimtVF  target=pto       (alias: simtvf-pto)
#   simd-asc   T.SimdVF  target=ascend    (aliases: simd, simdvf, simd-ascend)
#   simd-pto   T.SimdVF  target=pto       (alias: simdvf-pto)
#
# SimdVF and PTO modes may FAIL some cases. The script still prints a PASS/FAIL
# table and exits non-zero only when an oneshot script itself fails (missing
# env, etc.), not because a case cell is FAIL. A missing PTO backend is
# FAIL compile; it is not remapped to ascend.
#
# Default (ST_VF_MODE and ST_VF_MODES unset): all four modes.
# ST_VF_MODES="simd-pto simt-asc"   one or more modes (comma or space separated)
# ST_VF_MODE=simt-asc               single mode when ST_VF_MODES is unset
# ST_VF_ONESHOTS="oneshot_sv1_sv9.sh"   subset of family oneshots
# ST_VF_REGRESS_OUT=/tmp/...        root; each mode writes $root/<mode>/
# ST_VF_SUMMARIZE_ONLY=1            reprint the table, skip oneshots
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)

if [[ -z "${PY:-}" && -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
fi
TABLE_PY="${PY:-python3}"

# Canonical names come from vf_mode.resolve_vf_mode so shell aliases cannot drift.
canon_mode() {
  "$TABLE_PY" - "$HERE" "$1" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from vf_mode import resolve_vf_mode

try:
    print(resolve_vf_mode(sys.argv[2]))
except ValueError as exc:
    print(f"error: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

if [[ -n "${ST_VF_MODES:-}" ]]; then
  normalized="${ST_VF_MODES//,/ }"
  # shellcheck disable=SC2206
  RAW_MODES=($normalized)
elif [[ -n "${ST_VF_MODE:-}" ]]; then
  RAW_MODES=("$ST_VF_MODE")
else
  RAW_MODES=(simt-asc simt-pto simd-asc simd-pto)
fi

declare -A SEEN=()
MODES=()
for raw in "${RAW_MODES[@]}"; do
  [[ -z "$raw" ]] && continue
  mode=$(canon_mode "$raw") || exit 1
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
echo "SimdVF and PTO FAIL cells are recorded. This regress does not require every case to PASS."

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
