#!/bin/bash
set -euo pipefail
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel
SUITE=$WT/examples/ascend/st_simtvf_parallel
TGZ=/tmp/cf_pr272_sync_20261007.tgz

ls -la "$TGZ"
tar -xzf "$TGZ" -C "$SUITE"
# strip CR if any from shell scripts
sed -i 's/\r$//' "$SUITE"/oneshot_ptodsl_cf*.sh "$SUITE"/oneshot_cf6b_cf6bd.sh 2>/dev/null || true
chmod +x "$SUITE"/oneshot_ptodsl_cf*.sh "$SUITE"/oneshot_cf6b_cf6bd.sh 2>/dev/null || true

cd "$WT"
echo "branch=$(git branch --show-current)"
echo "HEAD_BEFORE=$(git rev-parse HEAD)"

# Verify key files landed
test -f "$SUITE/kernels_ptodsl/cf1d_pred_thresh_keep.py"
test -f "$SUITE/kernels_ptodsl/cf6bd_fat_hard_newton.py"
test -f "$SUITE/kernels/cf6b_fat_hard_newton.py"
test -f "$SUITE/reports/SIMT_SIMD_E2E_COMPARE_CF_20261007.md"
test -f "$SUITE/reports/cf6b_cf6bd_20261007/RESULT.md"
grep -q 'CF2 vs CF3' "$SUITE/reports/SIMT_SIMD_E2E_COMPARE_CF_20261007.md"
grep -q 'cf6b_' "$SUITE/run_opsim_generic.py"
echo "LANDING_OK"

# Stage only CF Layer-D / CF6b deliverables (avoid unrelated dirty tree)
git add \
  examples/ascend/st_simtvf_parallel/kernels_ptodsl/cf1d_pred_thresh_keep.py \
  examples/ascend/st_simtvf_parallel/kernels_ptodsl/cf2d_remat_thresh_kill_shared.py \
  examples/ascend/st_simtvf_parallel/kernels_ptodsl/cf3d_remat_idx.py \
  examples/ascend/st_simtvf_parallel/kernels_ptodsl/cf4d_nested_if.py \
  examples/ascend/st_simtvf_parallel/kernels_ptodsl/cf5d_div_ulp_branch.py \
  examples/ascend/st_simtvf_parallel/kernels_ptodsl/cf6d_newton_branch.py \
  examples/ascend/st_simtvf_parallel/kernels_ptodsl/cf6bd_fat_hard_newton.py \
  examples/ascend/st_simtvf_parallel/kernels/cf6b_fat_hard_newton.py \
  examples/ascend/st_simtvf_parallel/oneshot_ptodsl_cf1d_cf6d.sh \
  examples/ascend/st_simtvf_parallel/oneshot_ptodsl_cf2d_cf4d.sh \
  examples/ascend/st_simtvf_parallel/oneshot_ptodsl_cf5d_cf6d.sh \
  examples/ascend/st_simtvf_parallel/oneshot_cf6b_cf6bd.sh \
  examples/ascend/st_simtvf_parallel/reports/SIMT_SIMD_E2E_COMPARE_CF_20261007.md \
  examples/ascend/st_simtvf_parallel/reports/simt_simd_e2e_cf_20261007.tsv \
  examples/ascend/st_simtvf_parallel/reports/ST_CF_FULL_COMPARE.md \
  examples/ascend/st_simtvf_parallel/reports/ST_CF1D_CF6D_SUMMARY.md \
  examples/ascend/st_simtvf_parallel/reports/ST_CF2D_CF4D.md \
  examples/ascend/st_simtvf_parallel/reports/ST_CF5D_CF6D.md \
  examples/ascend/st_simtvf_parallel/reports/ST_CF5_CF6_DIVERGENCE.md \
  examples/ascend/st_simtvf_parallel/reports/ST_CF6_DIVERGENCE_CASE.md \
  examples/ascend/st_simtvf_parallel/reports/ST_CF6B_RESULT.md \
  examples/ascend/st_simtvf_parallel/reports/cf6b_cf6bd_20261007/ \
  examples/ascend/st_simtvf_parallel/run_opsim_generic.py

echo "--- staged ---"
git status -sb | head -60
git diff --cached --stat

if git diff --cached --quiet; then
  echo "NO_STAGED_CHANGES"
  git log -1 --oneline
  exit 0
fi

git -c user.name='peanutchan' -c user.email='peanutchan@users.noreply.github.com' commit -m "$(cat <<'EOF'
st: CF Layer-D twins CF1d–CF6d, CF6b fat-hard, E2E compare

PTO-DSL *d twins of Simt CF1–CF6 plus CF6b/CF6bd fat-hard Newton
divergence probe. Opsim Ascend950PR_9599 walls + SIMT/SIMD E2E CF
compare (2026-10-07) with CF2 vs CF3 sensitivity clarity. Includes
oneshots, run_opsim_generic cf6b_ tags, and cf6b_cf6bd RESULT.
EOF
)"

SHA=$(git rev-parse HEAD)
echo "COMMIT_SHA=$SHA"
git log -1 --oneline
git remote -v

# Prefer peanutchan remote if present; else origin (non-force)
if git remote | grep -qx peanutchan; then
  echo "PUSHING peanutchan HEAD:feat/st-simtvf-parallel-sv1-sv9"
  git push peanutchan HEAD:feat/st-simtvf-parallel-sv1-sv9
else
  echo "PUSHING origin HEAD:feat/st-simtvf-parallel-sv1-sv9"
  git push origin HEAD:feat/st-simtvf-parallel-sv1-sv9
fi
echo "PUSH_RC=$?"
echo "HEAD_AFTER=$(git rev-parse HEAD)"
git status -sb | head -20
