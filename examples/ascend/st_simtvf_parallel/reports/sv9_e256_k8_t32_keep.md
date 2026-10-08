# sv9_e256_k8_t32_keep

- status: **PASS**
- wall_us: 1.48
- 
- layout: launch_bounds=32; scores[] sizes=['8']; idxs[] sizes=['8']; idx_cand[] sizes=['8']; amax[] sizes=['0', '1']; tir=sv9_e256_k8_t32_keep_tir.txt; src=sv9_e256_k8_t32_keep_source.txt; has AscendAllReduce; cross-thread reduce path; elems_per_thread≈8.00
- instr: VF_SIMT us=0.62 cyc=1122 body_instr=567 IPC_proxy=0.505 | top_opcodes: SIMT_SEL:128, SIMT_ISETP:64, SIMT_FSETP:64, SIMT_FMNMX:56, SIMT_IMNMX:56, MOV_XD_IMM:23, SIMT_REDUX:16, SIMT_LEA:15, MOVK:15, MOV_XD_SPR:13, SIMT_IADD_I:10, SIMT_MOVI:10 | pipes: RVECEX:440, SCALAR:86, RVECLP:9, RVECST:8, RVECLD:8, FLOWCTRL:5, PUSHQ:4, MTE3:3
