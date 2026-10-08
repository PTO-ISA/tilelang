# Relay oneshot_cf1v_cf6v.sh to pto-b10 (VMI / overlay).
$ErrorActionPreference = "Stop"
$SuiteLocal = Join-Path $env:USERPROFILE "st_simtvf_parallel"
if (-not (Test-Path (Join-Path $SuiteLocal "oneshot_cf1v_cf6v.sh"))) {
  Write-Host "Missing oneshot_cf1v_cf6v.sh under $SuiteLocal — sync suite first"
  exit 1
}
ssh -o BatchMode=yes -o ConnectTimeout=90 pto-b10 @"
set -euo pipefail
mkdir -p /tmp/st_simtvf_parallel_suite
# Prefer worktree suite if present
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel/examples/ascend/st_simtvf_parallel
if [[ -f \$WT/oneshot_cf1v_cf6v.sh ]]; then
  SUITE=\$WT
else
  SUITE=/tmp/st_simtvf_parallel_suite
fi
export SUITE
bash \$SUITE/oneshot_cf1v_cf6v.sh
"@
