#!/bin/bash
set -euo pipefail
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel
SUITE=$WT/examples/ascend/st_simtvf_parallel
TGZ=/tmp/cf_primary_cut_20261007.tgz

ls -la "$TGZ"
# extract into suite root (tarball paths are reports/... and SUMMARY.md)
tar -xzf "$TGZ" -C "$SUITE"

cd "$WT"
echo "branch=$(git branch --show-current)"
echo "HEAD_BEFORE=$(git rev-parse HEAD)"

test -f "$SUITE/reports/SIMT_SIMD_E2E_COMPARE_CF_20261007.md"
test -f "$SUITE/reports/ST_CF_FULL_COMPARE.md"
test -f "$SUITE/reports/simt_simd_e2e_cf_20261007.tsv"
grep -q 'Product cut' "$SUITE/reports/SIMT_SIMD_E2E_COMPARE_CF_20261007.md"
grep -q 'CF1–CF3 + CF6' "$SUITE/reports/ST_CF1_CF6_SLICES.md"
grep -q 'divergence experiments' "$SUITE/reports/SIMT_SIMD_E2E_COMPARE_CF_20261007.md"
echo "LANDING_OK"

git add \
  examples/ascend/st_simtvf_parallel/reports/SIMT_SIMD_E2E_COMPARE_CF_20261007.md \
  examples/ascend/st_simtvf_parallel/reports/ST_CF_FULL_COMPARE.md \
  examples/ascend/st_simtvf_parallel/reports/simt_simd_e2e_cf_20261007.tsv \
  examples/ascend/st_simtvf_parallel/reports/ST_CF1_CF6_SLICES.md \
  examples/ascend/st_simtvf_parallel/reports/SUMMARY.md \
  examples/ascend/st_simtvf_parallel/SUMMARY.md

echo "--- staged ---"
git status -sb | head -40
git diff --cached --stat

if git diff --cached --quiet; then
  echo "NO_STAGED_CHANGES"
  git log -1 --oneline
  exit 0
fi

git -c user.name='peanutchan' -c user.email='peanutchan@users.noreply.github.com' commit -m "$(cat <<'EOFMSG'
st: slim primary CF compare to CF1–CF3 + CF6

Primary E2E table/TSV and ST_CF_FULL_COMPARE keep CF1 (k1+k8), CF2,
CF3, CF6 (pnear5; pnear25 flat). Frame *d as SIMD rooftop and when
SIMD beats Simt (CF2 remat, CF3, CF6 *d); annotate CF6 *v ABI tax.
CF4/CF5/CF6b moved to appendix (files kept). Product-cut notes in
ST_CF1_CF6_SLICES + SUMMARY.
EOFMSG
)"

SHA=$(git rev-parse HEAD)
echo "COMMIT_SHA=$SHA"
git log -1 --oneline
git remote -v | head -8

if git remote | grep -qx peanutchan; then
  echo "PUSHING peanutchan HEAD:feat/st-simtvf-parallel-sv1-sv9"
  git push peanutchan HEAD:feat/st-simtvf-parallel-sv1-sv9
else
  echo "PUSHING origin HEAD:feat/st-simtvf-parallel-sv1-sv9"
  git push origin HEAD:feat/st-simtvf-parallel-sv1-sv9
fi
echo "PUSH_RC=$?"
echo "HEAD_AFTER=$(git rev-parse HEAD)"
git status -sb | head -15
