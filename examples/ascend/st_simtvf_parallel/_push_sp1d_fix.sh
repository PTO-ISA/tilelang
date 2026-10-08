#!/bin/bash
set -euo pipefail
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel
ST=$WT/examples/ascend/st_simtvf_parallel
# stage files already on host at /tmp
mkdir -p "$ST/kernels_ptodsl" "$ST/reports"
cp -f /tmp/sp1d_dual_scatter_keep_pos.py "$ST/kernels_ptodsl/"
cp -f /tmp/ST_PTODSL_PHASE1_COMPARE.md "$ST/reports/"
# also refresh box-synced copies under /tmp tree
mkdir -p /tmp/st_simtvf_parallel/kernels_ptodsl /tmp/st_simtvf_parallel/reports
cp -f /tmp/sp1d_dual_scatter_keep_pos.py /tmp/st_simtvf_parallel/kernels_ptodsl/
cp -f /tmp/ST_PTODSL_PHASE1_COMPARE.md /tmp/st_simtvf_parallel/reports/

cd "$WT"
git status -sb
git branch --show-current
git remote -v | head -5
# ensure on right branch
git checkout feat/st-simtvf-parallel-sv1-sv9
git pull --ff-only peanutchan feat/st-simtvf-parallel-sv1-sv9 2>/dev/null || git pull --ff-only origin feat/st-simtvf-parallel-sv1-sv9 2>/dev/null || true
git add examples/ascend/st_simtvf_parallel/kernels_ptodsl/sp1d_dual_scatter_keep_pos.py \
        examples/ascend/st_simtvf_parallel/reports/ST_PTODSL_PHASE1_COMPARE.md
git status -sb
git diff --cached --stat
git commit -m "$(cat <<'MSG'
st: fix SP1d ptodsl scatter via pto.for_/vscatter

Rewrite SP1d keep_pos to avoid Python-unrolled T×K×Eexp IR blow-up.
Use pto.for_ + vcmps/vsel for V; Sf via VL-padded scratch + size=1 pack.
Opsim Ascend950PR_9599 PASS core0.veccore0 22.84 µs.
MSG
)"
# push peanutchan
if git remote | grep -q peanutchan; then
  git push peanutchan HEAD:feat/st-simtvf-parallel-sv1-sv9
else
  git push origin HEAD:feat/st-simtvf-parallel-sv1-sv9
fi
git rev-parse HEAD
git log -1 --oneline
echo PUSH_DONE
