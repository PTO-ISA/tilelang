#!/bin/bash
set +e
DEPS=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018
sed -n '1310,1360p' $DEPS/ptodsl/_vmi_namespace.py
echo ====OPS====
sed -n '730,780p' $DEPS/ptodsl/_ops_vmem.py
echo ====EXAMPLES====
grep -rn "vscatter\|vmi.vscatter" ~/projects/pto-vmi/dsl --include='*.py' 2>/dev/null | head -30
grep -rn "vscatter" $DEPS/ptodsl --include='*.py' | grep -i example | head
# find any test using vscatter
find $DEPS -name '*scatter*' 2>/dev/null | head
grep -rn "\.vscatter(" ~/projects/pto-vmi ~/projects/PTOAS-main/ptodsl --include='*.py' 2>/dev/null | head -20
echo ====FOR_DOC====
sed -n '200,280p' $DEPS/ptodsl/_control_flow.py
