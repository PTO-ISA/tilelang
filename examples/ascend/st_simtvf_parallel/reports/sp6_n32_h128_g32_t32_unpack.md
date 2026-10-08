# sp6_n32_h128_g32_t32_unpack

- status: **PASS**
- wall_us: 5.35
- 
- layout: launch_bounds=32; tir=sp6_n32_h128_g32_t32_unpack_tir.txt; src=sp6_n32_h128_g32_t32_unpack_source.txt
- instr: VF_SIMT us=4.34 cyc=7813 body_instr=902 IPC_proxy=0.115 | top_opcodes: SIMT_LDS:192, SIMT_IADD_I:145, SIMT_STS:128, SIMT_LOP3:128, SIMT_IADD:65, SIMT_SHFI:65, SIMT_LEA:64, MOV_XD_IMM:26, MOVK:17, MOV_XD_SPR:13, MOV_SPR_XN:6, ADD_IMM:5 | pipes: RVECEX:472, RVECLD:192, RVECST:128, SCALAR:92, FLOWCTRL:5, PUSHQ:4, MTE2:4, MTE3:3
