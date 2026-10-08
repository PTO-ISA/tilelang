#!/bin/bash
set -euo pipefail
tar xzf /tmp/ptodsl_phase2.tgz -C /tmp
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel/examples/ascend/st_simtvf_parallel
mkdir -p "$WT/kernels_ptodsl" "$WT/reports" /tmp/st_simtvf_parallel
# Prefer syncing into a /tmp working copy that mirrors oneshot ROOT
rm -rf /tmp/st_simtvf_parallel_phase2
mkdir -p /tmp/st_simtvf_parallel_phase2
cp -a /tmp/kernels_ptodsl /tmp/oneshot_ptodsl_phase2.sh /tmp/reports /tmp/st_simtvf_parallel_phase2/ 2>/dev/null || true
# tarball extracts flat relative paths under /tmp
mkdir -p /tmp/st_simtvf_parallel_phase2/kernels_ptodsl /tmp/st_simtvf_parallel_phase2/reports
cp -f /tmp/kernels_ptodsl/*.py /tmp/st_simtvf_parallel_phase2/kernels_ptodsl/ 2>/dev/null || \
  (cd /tmp && tar xzf /tmp/ptodsl_phase2.tgz && cp -f kernels_ptodsl/*.py /tmp/st_simtvf_parallel_phase2/kernels_ptodsl/ && cp -f oneshot_ptodsl_phase2.sh /tmp/st_simtvf_parallel_phase2/ && cp -f reports/*.md /tmp/st_simtvf_parallel_phase2/reports/)

# Also sync to PR worktree (no commit yet)
cp -f /tmp/st_simtvf_parallel_phase2/kernels_ptodsl/sv2d_*.py /tmp/st_simtvf_parallel_phase2/kernels_ptodsl/sv4d_*.py /tmp/st_simtvf_parallel_phase2/kernels_ptodsl/sv9d_*.py "$WT/kernels_ptodsl/"
cp -f /tmp/st_simtvf_parallel_phase2/oneshot_ptodsl_phase2.sh "$WT/"
cp -f /tmp/st_simtvf_parallel_phase2/reports/ST_PTODSL_PLAN.md /tmp/st_simtvf_parallel_phase2/reports/ST_PTODSL_PHASE2_COMPARE.md "$WT/reports/"
chmod +x /tmp/st_simtvf_parallel_phase2/oneshot_ptodsl_phase2.sh "$WT/oneshot_ptodsl_phase2.sh"
ls /tmp/st_simtvf_parallel_phase2/kernels_ptodsl/
echo "==== RUN PHASE2 ===="
cd /tmp/st_simtvf_parallel_phase2
bash oneshot_ptodsl_phase2.sh
echo PHASE2_ONESHOT_DONE
