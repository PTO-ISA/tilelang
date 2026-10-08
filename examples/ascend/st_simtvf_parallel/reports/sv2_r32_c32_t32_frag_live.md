# sv2_r32_c32_t32_frag_live

- status: **PASS**
- wall_us: 4.93
- 
- layout: launch_bounds=32; x[] sizes=['32']; tir=sv2_r32_c32_t32_frag_live_tir.txt; src=sv2_r32_c32_t32_frag_live_source.txt
- instr: VF_SIMT us=4.1 cyc=7379 body_instr=753 IPC_proxy=0.102 | top_opcodes: SIMT_LEA:128, SIMT_IADD_I:66, SIMT_IADD:65, SIMT_F2F:64, SIMT_ISETP_I:64, SIMT_BRANCH:64, SIMT_STS:32, SIMT_LDK:32, SIMT_LDS:32, SIMT_STK:32, SIMT_FDIV:32, SIMT_FMNMX:32 | pipes: RVECEX:460, SCALAR:84, RVECLP:65, RVECST:64, RVECLD:64, FLOWCTRL:5, PUSHQ:4, MTE3:3
