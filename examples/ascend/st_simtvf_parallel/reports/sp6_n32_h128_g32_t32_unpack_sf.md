# sp6_n32_h128_g32_t32_unpack_sf

- status: **PASS**
- wall_us: 11.31
- 
- layout: launch_bounds=32; tir=sp6_n32_h128_g32_t32_unpack_sf_tir.txt; src=sp6_n32_h128_g32_t32_unpack_sf_source.txt
- instr: VF_SIMT us=10.32 cyc=18574 body_instr=1666 IPC_proxy=0.090 | top_opcodes: SIMT_LDS:324, SIMT_STS:256, SIMT_IADD_I:216, SIMT_LEA:176, SIMT_FMUL:128, SIMT_LOP3:128, SIMT_SHFI:83, SIMT_IADD:65, SIMT_ISETP_I:64, SIMT_BRANCH:64, SIMT_IADD_X:32, MOV_XD_IMM:29 | pipes: RVECEX:905, RVECLD:324, RVECST:256, SCALAR:97, RVECLP:65, MTE2:5, FLOWCTRL:5, PUSHQ:4
