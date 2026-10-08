#!/bin/bash
set -euo pipefail
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel
SUITE=$WT/examples/ascend/st_simtvf_parallel
tar xzf /tmp/st_simtvf_parallel_ptodsl_phase1.tgz -C "$SUITE"
cd "$WT"
git status -sb
git branch --show-current
# Only add Phase1 deliverables (avoid unrelated dirty tree noise)
git add \
  examples/ascend/st_simtvf_parallel/kernels_ptodsl \
  examples/ascend/st_simtvf_parallel/reports/ST_PTODSL_PLAN.md \
  examples/ascend/st_simtvf_parallel/reports/ST_PTODSL_PHASE1_COMPARE.md \
  examples/ascend/st_simtvf_parallel/oneshot_ptodsl_phase1.sh \
  examples/ascend/st_simtvf_parallel/run_via_laptop_ptodsl_phase1.ps1 \
  examples/ascend/st_simtvf_parallel/README.md
git status -sb
git -c user.name='peanutchan' -c user.email='peanutchan@users.noreply.github.com' \
  commit -m "$(cat <<'EOF'
st: Phase1 PTO-DSL twins SV1d/CF1d/SP1d

Layer-D explicit ptodsl twins of Simt SV1/CF1/SP1(keep_pos) with RF/UB/CF
maps. Opsim Ascend950PR_9599: SV1d 0.89us PASS, CF1d 0.9us PASS; SP1d
emit-mlir OK but device compile FAIL (IR blow-up). Docs + oneshot included.
EOF
)"
SHA=$(git rev-parse HEAD)
echo COMMIT_SHA=$SHA
# Push as peanutchan to PR #272 branch (no force)
git push origin HEAD:feat/st-simtvf-parallel-sv1-sv9
echo PUSH_RC:$?
git log -1 --oneline
