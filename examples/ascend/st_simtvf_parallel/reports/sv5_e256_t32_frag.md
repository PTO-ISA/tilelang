# sv5_e256_t32_frag

- status: **PASS**
- wall_us: 1.17
- 
- layout: launch_bounds=32; x[] sizes=['8']; amax[] sizes=['0', '1']; src=sv5_e256_t32_frag_source.txt; has AscendAllReduce; cross-thread reduce path; elems_per_thread≈8.00
- instr: VF_SIMT us=0.35 cyc=638 body_instr=169 IPC_proxy=0.265 | top_opcodes: MOV_XD_IMM:23, SIMT_STS:16, MOVK:14, MOV_XD_SPR:13, SIMT_LDS:8, SIMT_FMUL:8, SIMT_FSETP:8, SIMT_SEL:8, SIMT_FMNMX:7, MOV_SPR_XN:6, ADD_IMM:5, ADD:4 | pipes: SCALAR:86, RVECEX:41, RVECST:16, RVECLD:8, FLOWCTRL:5, MTE3:4, PUSHQ:4, MTE2:3
