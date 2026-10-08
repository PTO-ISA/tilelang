# cf2_e256_k8_t32_remat

- status: **PASS**
- wall_us: 1.95
- 
- layout: launch_bounds=32; scores[] sizes=['8']; tir=cf2_e256_k8_t32_remat_tir.txt; src=cf2_e256_k8_t32_remat_source.txt; elems_per_thread≈8.00
- instr: VF_SIMT us=1.13 cyc=2041 body_instr=372 IPC_proxy=0.182 | top_opcodes: SIMT_LDS:80, SIMT_STS:72, SIMT_FSETP:64, MOV_XD_IMM:25, MOVK:16, MOV_XD_SPR:13, SIMT_IADD_I:10, SIMT_LEA:9, SIMT_ISETP_I:8, SIMT_BRANCH:8, SIMT_BAR_THREAD_BLOCK:8, MOV_SPR_XN:6 | pipes: RVECEX:96, SCALAR:90, RVECLD:80, RVECST:72, RVECLP:9, RVECSU:8, FLOWCTRL:5, PUSHQ:4
