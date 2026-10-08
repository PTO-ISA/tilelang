#!/bin/bash
set +e
# Unpack into /tmp suite for oneshot, and into worktree for commit
rm -rf /tmp/sv2a_fold_unpack && mkdir -p /tmp/sv2a_fold_unpack
tar xzf /tmp/sv2a_fold.tgz -C /tmp/sv2a_fold_unpack
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel
ST=$WT/examples/ascend/st_simtvf_parallel
# Copy into worktree (do NOT touch ~/projects/tilelang-deepseek)
mkdir -p "$ST/kernels" "$ST/reports"
cp -f /tmp/sv2a_fold_unpack/kernels/sv2_eltwise_bcast_rf.py "$ST/kernels/"
cp -f /tmp/sv2a_fold_unpack/kernels/sv2v_eltwise_bcast_rf.py "$ST/kernels/"
cp -f /tmp/sv2a_fold_unpack/run_opsim_generic.py "$ST/"
cp -f /tmp/sv2a_fold_unpack/oneshot_sv2.sh "$ST/"
cp -f /tmp/sv2a_fold_unpack/reports/ST_RF_KEEP_STREAM_AUDIT.md "$ST/reports/"
cp -f /tmp/sv2a_fold_unpack/reports/ST_PTODSL_PHASE2_COMPARE.md "$ST/reports/"
chmod +x "$ST/oneshot_sv2.sh"

# Also stage a /tmp suite for oneshot (self-contained)
SUITE=/tmp/st_simtvf_parallel_suite
mkdir -p "$SUITE/kernels" "$SUITE/reports"
# Prefer worktree as suite root if it has harness; else copy essentials
if [[ -f "$ST/common_asc_harness.py" ]]; then
  SUITE=$ST
else
  cp -f /tmp/sv2a_fold_unpack/kernels/*.py "$SUITE/kernels/"
  cp -f /tmp/sv2a_fold_unpack/run_opsim_generic.py "$SUITE/"
  cp -f /tmp/sv2a_fold_unpack/oneshot_sv2.sh "$SUITE/"
  # harness must already exist in suite from prior deploys
fi
export SUITE
echo "SUITE=$SUITE"
ls -la "$SUITE/kernels/sv2_eltwise_bcast_rf.py" "$SUITE/oneshot_sv2.sh" "$SUITE/common_asc_harness.py" 2>&1 | head -20

# Run SV2 oneshot
bash "$SUITE/oneshot_sv2.sh"
echo "==== SUMMARY ===="
cat /tmp/st_simtvf_parallel/SUMMARY_sv2_raw.txt
echo SV2A_FOLD_DONE
