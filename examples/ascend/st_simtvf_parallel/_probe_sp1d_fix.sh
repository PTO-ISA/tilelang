#!/bin/bash
set +e
# Gather scatter / for_ / vscatter patterns from pto-vmi
echo ====GATHER_FOR====
grep -n "pto.for_\|vscatter\|vecscope\|vgather\|masked_store\|vsts\|vsstb" \
  ~/projects/pto-vmi/dsl/GatherElementsVfKernel/GatherElementsVfKernel_case0_fp32_fp32_128_64.py \
  ~/projects/pto-vmi/dsl/ScatterNdOffsetVfKernel/*.py \
  ~/projects/pto-vmi/dsl/ScatterNdAddUniqueIdVfKernel/*.py \
  ~/projects/pto-vmi/dsl/SelectGatherKernel/*.py \
  2>/dev/null | head -60

echo ====SCATTER_FILES====
ls ~/projects/pto-vmi/dsl/ScatterNdOffsetVfKernel/ 2>/dev/null
ls ~/projects/pto-vmi/dsl/SelectGatherKernel/ 2>/dev/null

echo ====SCATTER_SNIP====
sed -n '1,180p' ~/projects/pto-vmi/dsl/ScatterNdOffsetVfKernel/*.py 2>/dev/null | head -180

echo ====SELECT_SNIP====
sed -n '1,160p' ~/projects/pto-vmi/dsl/SelectGatherKernel/*.py 2>/dev/null | head -160

echo ====VSCATTER_IN_DEPS====
DEPS=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018
grep -rn "def vscatter\|def for_\|vscatter" "$DEPS/ptodsl" 2>/dev/null | head -40
python3 - <<'PYEOF'
import sys
sys.path.insert(0, "/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018")
from ptodsl import pto
print("vscatter" in dir(pto.vmi), [x for x in dir(pto.vmi) if "scatter" in x.lower() or "gather" in x.lower()])
print("for_", hasattr(pto, "for_"))
import ptodsl._control_flow as cf
print("cf", [x for x in dir(cf) if not x.startswith("_")])
PYEOF
