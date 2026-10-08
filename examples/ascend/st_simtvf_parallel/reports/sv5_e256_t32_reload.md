# sv5_e256_t32_reload

- status: **PASS**
- wall_us: 1.15
- 
- layout: launch_bounds=32; x[] sizes=['8']; amax[] sizes=['0', '1']; src=sv5_e256_t32_reload_source.txt; has AscendAllReduce; cross-thread reduce path; elems_per_thread≈8.00
- instr: VF_SIMT us=0.3 cyc=532 body_instr=181 IPC_proxy=0.340 | top_opcodes: MOV_XD_IMM:23, SIMT_STS:17, MOVK:14, MOV_XD_SPR:13, SIMT_LDS:9, SIMT_SHFI:9, SIMT_FMUL:8, SIMT_FSETP:8, SIMT_SEL:8, SIMT_FMNMX:7, MOV_SPR_XN:6, ADD_IMM:5 | pipes: SCALAR:86, RVECEX:49, RVECST:17, RVECLD:9, FLOWCTRL:5, MTE3:4, PUSHQ:4, MTE2:3
