#!/bin/bash
DEPS=/mnt/fluxdata/happybot/projects/tilelang-deepseek-pto-vmi-topk/.camodel_deps_vmi018
grep -rn "index_cast\|IndexCast\|cast.*i32\|as_i32\|to_i32\|trunc" $DEPS/ptodsl --include='*.py' 2>/dev/null | head -40
echo ====IF====
grep -rn "def if_\|class.*If" $DEPS/ptodsl/_control_flow.py | head -20
sed -n '1,100p' $DEPS/ptodsl/_control_flow.py | head -80
echo ====VMI_PAND====
grep -n "pand\|por\|pxor\|def vcmp" $DEPS/ptodsl/_vmi_namespace.py | head -30
# try small emit of index cast patterns from tests
grep -rn "vbrc(.*for_\|vbrc(i\|index_cast" $DEPS/ptodsl/tests --include='*.py' 2>/dev/null | head -20
grep -rn "vbrc(p\|vbrc(i" ~/projects/pto-vmi/dsl --include='*.py' 2>/dev/null | head -15
