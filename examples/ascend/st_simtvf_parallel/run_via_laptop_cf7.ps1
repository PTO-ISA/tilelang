# Relay oneshot_cf7.sh to pto-b10 (Simt CF7).
$ErrorActionPreference = "Stop"
ssh -o BatchMode=yes -o ConnectTimeout=90 pto-b10 @"
set -euo pipefail
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel/examples/ascend/st_simtvf_parallel
export SUITE=\$WT
bash \$SUITE/oneshot_cf7.sh
"@
