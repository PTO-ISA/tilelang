#!/bin/bash
set -euo pipefail
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel
SUITE=$WT/examples/ascend/st_simtvf_parallel
tar xzf /tmp/st_simtvf_parallel_ptodsl_phase1.tgz -C "$SUITE"
# also refresh /tmp suite
mkdir -p /tmp/st_simtvf_parallel
tar xzf /tmp/st_simtvf_parallel_ptodsl_phase1.tgz -C /tmp/st_simtvf_parallel
cd "$WT"
echo "branch=$(git branch --show-current)"
git status -sb | head -40
git add \
  examples/ascend/st_simtvf_parallel/kernels_ptodsl \
  examples/ascend/st_simtvf_parallel/reports/ST_PTODSL_PLAN.md \
  examples/ascend/st_simtvf_parallel/reports/ST_PTODSL_PHASE1_COMPARE.md \
  examples/ascend/st_simtvf_parallel/oneshot_ptodsl_phase1.sh \
  examples/ascend/st_simtvf_parallel/run_via_laptop_ptodsl_phase1.ps1 \
  examples/ascend/st_simtvf_parallel/README.md
git status -sb | head -30
if git diff --cached --quiet; then
  echo "NO_STAGED_CHANGES"
  git log -1 --oneline
  exit 0
fi
git -c user.name='peanutchan' -c user.email='peanutchan@users.noreply.github.com' commit -m "st: Phase1 PTO-DSL twins SV1d/CF1d/SP1d

Layer-D explicit ptodsl twins of Simt SV1/CF1/SP1(keep_pos) with RF/UB/CF
maps. Opsim Ascend950PR_9599: SV1d 0.89us PASS, CF1d 0.9us PASS; SP1d
emit-mlir OK but device compile FAIL (IR blow-up). Docs + oneshot included."
echo COMMIT_SHA=$(git rev-parse HEAD)
# Prefer peanutchan remote if configured
git remote -v | head -10
if git remote | grep -qx peanutchan; then
  git push peanutchan HEAD:feat/st-simtvf-parallel-sv1-sv9
else
  git push origin HEAD:feat/st-simtvf-parallel-sv1-sv9
fi
echo PUSH_DONE
git log -1 --oneline
