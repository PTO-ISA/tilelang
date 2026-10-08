#!/bin/bash
set -euo pipefail
rm -rf /tmp/sv2a_fold_unpack && mkdir -p /tmp/sv2a_fold_unpack
tar xzf /tmp/sv2a_fold.tgz -C /tmp/sv2a_fold_unpack
WT=/mnt/fluxdata/happybot/projects/tilelang-st-simtvf-parallel
ST=$WT/examples/ascend/st_simtvf_parallel
cp -f /tmp/sv2a_fold_unpack/kernels/sv2_eltwise_bcast_rf.py "$ST/kernels/"
cp -f /tmp/sv2a_fold_unpack/kernels/sv2v_eltwise_bcast_rf.py "$ST/kernels/"
cp -f /tmp/sv2a_fold_unpack/run_opsim_generic.py "$ST/"
cp -f /tmp/sv2a_fold_unpack/oneshot_sv2.sh "$ST/"
cp -f /tmp/sv2a_fold_unpack/reports/ST_RF_KEEP_STREAM_AUDIT.md "$ST/reports/"
cp -f /tmp/sv2a_fold_unpack/reports/ST_PTODSL_PHASE2_COMPARE.md "$ST/reports/"
chmod +x "$ST/oneshot_sv2.sh"
# also drop run/push helpers if present in tarball later
cd "$WT"
git checkout feat/st-simtvf-parallel-sv1-sv9
git config user.name "peanutchan" || true
git config user.email "peanutchan@users.noreply.github.com" || true
git add examples/ascend/st_simtvf_parallel/kernels/sv2_eltwise_bcast_rf.py \
        examples/ascend/st_simtvf_parallel/kernels/sv2v_eltwise_bcast_rf.py \
        examples/ascend/st_simtvf_parallel/run_opsim_generic.py \
        examples/ascend/st_simtvf_parallel/oneshot_sv2.sh \
        examples/ascend/st_simtvf_parallel/reports/ST_RF_KEEP_STREAM_AUDIT.md \
        examples/ascend/st_simtvf_parallel/reports/ST_PTODSL_PHASE2_COMPARE.md
git status -sb
git commit -m "$(cat <<'MSG'
st: SV2 A fold — alloc_fragment scale under T.Parallel

Reshape Simt SV2 to match D fold gold: per-lane vmax fold into
scale[R,LANES] fragment (LANES=threads). Consumer STREAMs x from
x_ub; KEEP only scale fragment (frag_live) or remat via shared
(reload). Alias fold_frag_keep. Twin SV2v same fold, shared working
(VMI ABI). Opsim gold updated; oneshot adds C64/C128.

Mapping: A frag_live ↔ D fold_scale_keep; A reload ↔ D fold_scale_reload.
MSG
)"
git push origin HEAD:feat/st-simtvf-parallel-sv1-sv9
echo SHA:$(git rev-parse HEAD)
git log -1 --oneline
echo PUSH_DONE
