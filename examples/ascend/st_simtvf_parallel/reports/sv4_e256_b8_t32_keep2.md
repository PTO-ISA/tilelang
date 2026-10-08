# sv4_e256_b8_t32_keep2

- status: **PASS**
- wall_us: 3.79
- 
- layout: launch_bounds=32; scores[] sizes=['8']; idxs[] sizes=['8']; tir=sv4_e256_b8_t32_keep2_tir.txt; src=sv4_e256_b8_t32_keep2_source.txt; elems_per_thread≈8.00
- instr: VF_SIMT us=2.82 cyc=5069 body_instr=597 IPC_proxy=0.118 | top_opcodes: SIMT_LDS:128, SIMT_IADD_I:73, SIMT_STS:64, SIMT_FMUL:64, SIMT_BAR_THREAD_BLOCK:64, SIMT_IADD:32, SIMT_LEA:25, MOV_XD_IMM:25, SIMT_SHFI:18, MOVK:16, MOV_XD_SPR:13, SIMT_ISETP_I:8 | pipes: RVECEX:225, RVECLD:128, SCALAR:90, RVECST:64, RVECSU:64, RVECLP:9, FLOWCTRL:5, PUSHQ:4
