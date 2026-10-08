#!/bin/bash
set -euo pipefail
rm -rf /tmp/p2extract
mkdir -p /tmp/p2extract
tar xzf /tmp/ptodsl_phase2.tgz -C /tmp/p2extract
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel
ST=$WT/examples/ascend/st_simtvf_parallel
mkdir -p "$ST/kernels_ptodsl" "$ST/reports"
cp -f /tmp/p2extract/kernels_ptodsl/sv2d_eltwise_bcast_rf.py \
      /tmp/p2extract/kernels_ptodsl/sv4d_index_gather_psum.py \
      /tmp/p2extract/kernels_ptodsl/sv9d_topk_e2e.py \
      "$ST/kernels_ptodsl/"
cp -f /tmp/p2extract/oneshot_ptodsl_phase2.sh "$ST/"
cp -f /tmp/p2extract/run_via_laptop_ptodsl_phase2.ps1 "$ST/" 2>/dev/null || true
cp -f /tmp/p2extract/reports/ST_PTODSL_PLAN.md \
      /tmp/p2extract/reports/ST_PTODSL_PHASE2_COMPARE.md \
      "$ST/reports/"
chmod +x "$ST/oneshot_ptodsl_phase2.sh"
cd "$WT"
git checkout feat/st-simtvf-parallel-sv1-sv9
git add examples/ascend/st_simtvf_parallel/kernels_ptodsl/sv2d_eltwise_bcast_rf.py \
        examples/ascend/st_simtvf_parallel/kernels_ptodsl/sv4d_index_gather_psum.py \
        examples/ascend/st_simtvf_parallel/kernels_ptodsl/sv9d_topk_e2e.py \
        examples/ascend/st_simtvf_parallel/oneshot_ptodsl_phase2.sh \
        examples/ascend/st_simtvf_parallel/reports/ST_PTODSL_PLAN.md \
        examples/ascend/st_simtvf_parallel/reports/ST_PTODSL_PHASE2_COMPARE.md
# optional ps1
git add examples/ascend/st_simtvf_parallel/run_via_laptop_ptodsl_phase2.ps1 2>/dev/null || true
git status -sb
git diff --cached --stat
git commit -m "$(cat <<'MSG'
st: Phase2 PTO-DSL RF ladder SV2d/SV4d/SV9d

Add explicit ptodsl twins stressing RF keep vs remat/reload:
SV2d frag_live/reload, SV4d keep_idx/remat_idx, SV9d keep+remat arms.
Opsim Ascend950PR_9599: all 7 primary tags PASS (wall µs in PHASE2_COMPARE).
MSG
)"
git push origin HEAD:feat/st-simtvf-parallel-sv1-sv9
echo SHA:$(git rev-parse HEAD)
git log -1 --oneline
echo PUSH_DONE
