from ptodsl import pto
print("pto OK", "f16", hasattr(pto, "f16"), "f32", hasattr(pto, "f32"))
vmi = pto.vmi
names = sorted(n for n in dir(vmi) if not n.startswith("_"))
print("vmi_ops", ",".join(names))
for n in [
    "vload", "vstore", "vmax", "vmin", "vmul", "vadd", "vdiv", "vcmax", "vcmin",
    "vbrc", "vcast", "cast", "vsel", "vcmp", "vgather", "create_mask", "vabs",
    "vexp", "vrec", "vsqrt",
]:
    print(n, hasattr(vmi, n))
print("pto.f16", getattr(pto, "f16", None))
print("mem_bar", hasattr(pto, "mem_bar"))
print("for_", hasattr(pto, "for_"))
