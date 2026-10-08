# Relay Phase-2 PTO-DSL oneshot to pto-b10 via SSH (run from laptop).
$ErrorActionPreference = "Stop"
$remote = "happybot@pto-b10.ddns.net"
$port = 8022
$wt = "/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel/examples/ascend/st_simtvf_parallel"
Write-Host "Sync kernels_ptodsl + oneshot to worktree then run oneshot_ptodsl_phase2.sh"
