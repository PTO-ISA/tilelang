#!/bin/bash
set -euo pipefail
rm -rf /tmp/sv2a && mkdir -p /tmp/sv2a
tar xzf /tmp/ptodsl_sv2d_audit.tgz -C /tmp/sv2a
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel
ST=$WT/examples/ascend/st_simtvf_parallel
cp -f /tmp/sv2a/kernels_ptodsl/sv2d_eltwise_bcast_rf.py "$ST/kernels_ptodsl/"
cp -f /tmp/sv2a/oneshot_ptodsl_phase2.sh "$ST/"
cp -f /tmp/sv2a/reports/ST_PTODSL_PHASE2_COMPARE.md \
      /tmp/sv2a/reports/ST_PTODSL_PLAN.md \
      /tmp/sv2a/reports/ST_RF_KEEP_STREAM_AUDIT.md \
      "$ST/reports/"
chmod +x "$ST/oneshot_ptodsl_phase2.sh"
cd "$WT"
git checkout feat/st-simtvf-parallel-sv1-sv9
git add examples/ascend/st_simtvf_parallel/kernels_ptodsl/sv2d_eltwise_bcast_rf.py \
        examples/ascend/st_simtvf_parallel/oneshot_ptodsl_phase2.sh \
        examples/ascend/st_simtvf_parallel/reports/ST_PTODSL_PHASE2_COMPARE.md \
        examples/ascend/st_simtvf_parallel/reports/ST_PTODSL_PLAN.md \
        examples/ascend/st_simtvf_parallel/reports/ST_RF_KEEP_STREAM_AUDIT.md
git commit -m "$(cat <<'MSG'
st: rewrite SV2d KEEP-scales/STREAM-rows + RF audit

SV2d primary arms scale_keep_stream / full_reload (row_keep_tile negative).
Add ST_RF_KEEP_STREAM_AUDIT.md for SV/CF/SP A+B+D keep-vs-stream review.
Opsim: scale_keep_stream 1.62µs, full_reload 1.33µs, row_keep_tile 1.06µs.
MSG
)"
git push origin HEAD:feat/st-simtvf-parallel-sv1-sv9
echo SHA:$(git rev-parse HEAD)
git log -1 --oneline
echo PUSH_DONE
