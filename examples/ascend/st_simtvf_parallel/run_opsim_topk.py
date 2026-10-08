#!/usr/bin/env python3
"""Opsim runner for SV9 topk e2e (and archived sv2/sv3/sv4 topk micros)."""
from __future__ import annotations

import ctypes
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp")
import run_cf_mb_opsim as R  # noqa: E402

TAG = sys.argv[1]  # e.g. sv9_e256_k8_t32_keep
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/st_simtvf_parallel")
SO = OUT / "so" / f"{TAG}.so"

m = re.search(r"_e(\d+)_k(\d+)_t(\d+)", TAG)
if not m:
    raise SystemExit(f"bad tag {TAG}")
E, K, _T = int(m.group(1)), int(m.group(2)), int(m.group(3))

acl = R.load_acl()
R.check_acl(acl.aclInit(None), "init")
R.check_acl(acl.aclrtSetDevice(0), "dev")
stream = ctypes.c_void_p()
R.check_acl(acl.aclrtCreateStream(ctypes.byref(stream)), "stream")

scores = np.zeros(E, np.float32)
peaks = [144, 33, 200, 7, 99, 12, 180, 55][:K]
for rank, idx in enumerate(peaks):
    scores[idx] = np.float32(1000.0 - rank)
fill = np.float32(0.0)
for i in range(E):
    if i not in set(peaks):
        scores[i] = fill
        fill -= np.float32(1.0)
exp = np.array(peaks, np.int32)
out_h = np.zeros(K, np.int32)

ctypes.CDLL("libruntime.so", mode=ctypes.RTLD_GLOBAL)
for _cand in (
    "/home/happybot/projects/tilelang-pto-vmi-deps-stack/build/lib/libtvm_ffi.so",
):
    if Path(_cand).is_file():
        try:
            ctypes.CDLL(_cand, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            print("tvm_ffi preload skip", e)
lib = ctypes.CDLL(str(SO))
fn = getattr(lib, "call")
fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
d_in = ctypes.c_void_p()
d_out = ctypes.c_void_p()
R.check_acl(acl.aclrtMalloc(ctypes.byref(d_in), scores.nbytes, R.ACL_MEM_MALLOC_HUGE_FIRST), "mi")
R.check_acl(acl.aclrtMalloc(ctypes.byref(d_out), out_h.nbytes, R.ACL_MEM_MALLOC_HUGE_FIRST), "mo")
R.check_acl(
    acl.aclrtMemcpy(
        d_in, scores.nbytes, scores.ctypes.data_as(ctypes.c_void_p), scores.nbytes, R.ACL_MEMCPY_HOST_TO_DEVICE
    ),
    "h2d",
)
print("call rc", fn(d_in, d_out, stream))
R.check_acl(acl.aclrtSynchronizeStream(stream), "sync")
R.check_acl(
    acl.aclrtMemcpy(
        out_h.ctypes.data_as(ctypes.c_void_p), out_h.nbytes, d_out, out_h.nbytes, R.ACL_MEMCPY_DEVICE_TO_HOST
    ),
    "d2h",
)
ok = bool(np.array_equal(out_h, exp))
print(("PASS" if ok else "FAIL"), "got=", out_h.tolist(), "exp=", exp.tolist())
(OUT / f"opsim_{TAG}_result.txt").write_text(f"ok={ok} got={out_h.tolist()} exp={exp.tolist()}\n")
raise SystemExit(0 if ok else 2)
