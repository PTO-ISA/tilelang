# sv2g_m32_vl64_k16_t32_keep

- status: **PASS**
- wall_us: 8.63
- 
- layout: launch_bounds=32; acc[] sizes=['64']; acc[64]; tir=sv2g_m32_vl64_k16_t32_keep_tir.txt; src=sv2g_m32_vl64_k16_t32_keep_source.txt
- instr: VF_SIMT us=6.6 cyc=11874 body_instr=4042 IPC_proxy=0.340 | top_opcodes: SIMT_LDS:1040, SIMT_FFMA:1024, SIMT_IADD_I:594, SIMT_F2F:528, SIMT_PRMT:512, SIMT_STS:64, SIMT_MOV:64, SIMT_LEA:33, MOV_XD_IMM:26, MOVK:20, SIMT_SHFI:18, SIMT_IADD:16 | pipes: RVECEX:2809, RVECLD:1040, SCALAR:95, RVECST:64, RVECLP:17, FLOWCTRL:5, PUSHQ:4, MTE2:4
