# Relay Phase-1 PTO-DSL oneshot to pto-b10.
$ErrorActionPreference = "Stop"
ssh -p 8022 -o StrictHostKeyChecking=no happybot@pto-b10.ddns.net @'
set -euo pipefail
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel/examples/ascend/st_simtvf_parallel
ROOT=/tmp/st_simtvf_parallel
if [ -d "$WT/kernels_ptodsl" ]; then ROOT=$WT; fi
cd "$ROOT"
bash oneshot_ptodsl_phase1.sh
'@
