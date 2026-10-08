#!/bin/bash
set -euo pipefail
rm -rf /tmp/sv2d_fold_unpack && mkdir -p /tmp/sv2d_fold_unpack
tar xzf /tmp/sv2d_fold.tgz -C /tmp/sv2d_fold_unpack
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel
ST=$WT/examples/ascend/st_simtvf_parallel
cp -f /tmp/sv2d_fold_unpack/kernels_ptodsl/sv2d_eltwise_bcast_rf.py "$ST/kernels_ptodsl/"
cp -f /tmp/sv2d_fold_unpack/oneshot_ptodsl_phase2.sh "$ST/"
cp -f /tmp/sv2d_fold_unpack/_run_sv2d_fold.sh "$ST/"
cp -f /tmp/sv2d_fold_unpack/reports/ST_PTODSL_PHASE2_COMPARE.md "$ST/reports/"
cp -f /tmp/sv2d_fold_unpack/reports/ST_RF_KEEP_STREAM_AUDIT.md "$ST/reports/"
chmod +x "$ST/oneshot_ptodsl_phase2.sh" "$ST/_run_sv2d_fold.sh"
cd "$WT"
git checkout feat/st-simtvf-parallel-sv1-sv9
# peanutchan identity for git remotes (as in prior work)
git config user.name "peanutchan" || true
git config user.email "peanutchan@users.noreply.github.com" || true
git add examples/ascend/st_simtvf_parallel/kernels_ptodsl/sv2d_eltwise_bcast_rf.py \
        examples/ascend/st_simtvf_parallel/oneshot_ptodsl_phase2.sh \
        examples/ascend/st_simtvf_parallel/_run_sv2d_fold.sh \
        examples/ascend/st_simtvf_parallel/reports/ST_PTODSL_PHASE2_COMPARE.md \
        examples/ascend/st_simtvf_parallel/reports/ST_RF_KEEP_STREAM_AUDIT.md
git status -sb
git commit -m "$(cat <<'MSG'
st: SV2d fold_scale_keep — vmax-only per-lane scale KEEP

Rework ST-SV2d into folding per-channel quant: scale[lane] =
max(EPS, max_ch abs(X[i,ch*VL+lane])) via vmax only — no vcmax,
no pack lane-select, no size-1 scalar remat.

Arms: fold_scale_keep (primary, scale VL stays in RF) /
fold_scale_reload (VL scale→UB + mem_bar VST_VLD contrast).
Demote old vcmax+scalar arms from oneshot.

Opsim Ascend950PR_9599: C64 keep 1.01µs; C128 keep 1.34µs vs
reload 1.87µs (RVECEX tied 2578; reload RVECLD 3224 vs 864).
MSG
)"
git push origin HEAD:feat/st-simtvf-parallel-sv1-sv9
echo SHA:$(git rev-parse HEAD)
git log -1 --oneline
echo PUSH_DONE
