#!/usr/bin/env bash
set +e
echo ===FIND_SV4===
find /tmp -maxdepth 3 -type d -name 'opsim_sv4_*' 2>/dev/null | head -40
echo ===FIND_KEEP_IDX===
find /tmp -maxdepth 4 -type d -name '*keep_idx*' 2>/dev/null | head -40
echo ===WT_SUITE_KERNEL===
head -25 /mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel/examples/ascend/st_simtvf_parallel/kernels/sv2_eltwise_bcast_rf.py 2>/dev/null
rg -n "input_keep|_ARMS" /mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel/examples/ascend/st_simtvf_parallel/kernels/sv2_eltwise_bcast_rf.py 2>/dev/null | head -10
