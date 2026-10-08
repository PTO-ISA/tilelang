# sv8_r64_c128_g16_t32_spill_dist

- status: **PASS**
- wall_us: 18.23
- 
- layout: launch_bounds=32; tir=sv8_r64_c128_g16_t32_spill_dist_tir.txt; src=sv8_r64_c128_g16_t32_spill_dist_source.txt
- instr: VF_SIMT us=16.969999 cyc=30545 body_instr=4220 IPC_proxy=0.138 | top_opcodes: SIMT_IADD_I:1217, SIMT_LDS:768, SIMT_F2F:768, SIMT_STS:512, SIMT_FMUL:256, SIMT_FMNMX:240, SIMT_LDK:150, SIMT_STK:150, SIMT_FMNMX_I:32, MOV_XD_IMM:22, SIMT_FDIV_I:16, MOVK:14 | pipes: RVECEX:2538, RVECLD:918, RVECST:662, SCALAR:84, FLOWCTRL:5, PUSHQ:4, MTE3:3, MTE2:3
