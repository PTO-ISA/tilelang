#!/usr/bin/env bash
set -euo pipefail
OUT=/tmp/st_simtvf_parallel/ptodsl_sv1_sv9d
cd "$OUT"
echo -e "tag\tstatus\twall_us\tmaxabs\tnote"
for f in *.log; do
  tag="${f%.log}"
  status=UNKNOWN
  wall=NA
  maxabs=NA
  note=
  if grep -q "^PASS ${tag}" "$f" 2>/dev/null; then
    status=PASS
    maxabs=$(grep -oE "maxabs=[0-9.eE+-]+" "$f" | head -1 | cut -d= -f2 || true)
  elif grep -qE "exceeded vf stack|ensure_layout|Traceback \(most recent" "$f" 2>/dev/null; then
    status=COMPILE_FAIL
    note=$(grep -E "exceeded vf stack|ensure_layout|Error:" "$f" | head -1 | tr '\t' ' ' | cut -c1-120 || true)
  elif grep -qiE "^FAIL " "$f" 2>/dev/null; then
    status=FAIL
  fi
  wall=$(awk '/core0\.veccore0/ {print $2; exit}' "$f" 2>/dev/null || true)
  if [[ -z "$wall" || "$wall" == "duration_time(us)" ]]; then
    wall=$(grep -A1 'core_name' "$f" | grep veccore0 | awk '{print $2}' | head -1 || true)
  fi
  [[ -z "$wall" ]] && wall=NA
  echo -e "${tag}\t${status}\t${wall}\t${maxabs:-NA}\t${note}"
done
