#!/bin/bash
set -euo pipefail
rm -rf /tmp/sv2_ik_unpack && mkdir -p /tmp/sv2_ik_unpack
tar xzf /tmp/sv2_input_keep.tgz -C /tmp/sv2_ik_unpack
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel
ST=$WT/examples/ascend/st_simtvf_parallel
cp -f /tmp/sv2_ik_unpack/kernels/sv2_eltwise_bcast_rf.py "$ST/kernels/"
cp -f /tmp/sv2_ik_unpack/kernels/sv2v_eltwise_bcast_rf.py "$ST/kernels/"
cp -f /tmp/sv2_ik_unpack/kernels_ptodsl/sv2d_eltwise_bcast_rf.py "$ST/kernels_ptodsl/"
cp -f /tmp/sv2_ik_unpack/run_opsim_generic.py "$ST/"
cp -f /tmp/sv2_ik_unpack/oneshot_sv2.sh "$ST/"
cp -f /tmp/sv2_ik_unpack/oneshot_ptodsl_phase2.sh "$ST/"
cp -f /tmp/sv2_ik_unpack/_run_sv2d_input.sh "$ST/"
cp -f /tmp/sv2_ik_unpack/reports/ST_RF_KEEP_STREAM_AUDIT.md "$ST/reports/"
cp -f /tmp/sv2_ik_unpack/reports/ST_PTODSL_PHASE2_COMPARE.md "$ST/reports/"
chmod +x "$ST/oneshot_sv2.sh" "$ST/oneshot_ptodsl_phase2.sh" "$ST/_run_sv2d_input.sh"
cd "$WT"
git checkout feat/st-simtvf-parallel-sv1-sv9
git config user.name "peanutchan" || true
git config user.email "peanutchan@users.noreply.github.com" || true
git add examples/ascend/st_simtvf_parallel/kernels/sv2_eltwise_bcast_rf.py \
        examples/ascend/st_simtvf_parallel/kernels/sv2v_eltwise_bcast_rf.py \
        examples/ascend/st_simtvf_parallel/kernels_ptodsl/sv2d_eltwise_bcast_rf.py \
        examples/ascend/st_simtvf_parallel/run_opsim_generic.py \
        examples/ascend/st_simtvf_parallel/oneshot_sv2.sh \
        examples/ascend/st_simtvf_parallel/oneshot_ptodsl_phase2.sh \
        examples/ascend/st_simtvf_parallel/_run_sv2d_input.sh \
        examples/ascend/st_simtvf_parallel/reports/ST_RF_KEEP_STREAM_AUDIT.md \
        examples/ascend/st_simtvf_parallel/reports/ST_PTODSL_PHASE2_COMPARE.md
git status -sb
git commit -m "$(cat <<'MSG'
st: SV2 input KEEP vs STREAM — scale ALWAYS KEEP

Contract: folded scale always stays in RF (A alloc_fragment /
D VL vmax). Turning point is input by R: input_keep @R≈16
(x+scale live) vs input_stream @R≥32 (stream x, scale KEEP).
Demote scale-remat arms (reload / fold_scale_reload).

Opsim Ascend950PR_9599: see harvest / SUMMARY_sv2_raw.
MSG
)"
git push origin HEAD:feat/st-simtvf-parallel-sv1-sv9
echo SHA:$(git rev-parse HEAD)
git log -1 --oneline
echo PUSH_DONE
