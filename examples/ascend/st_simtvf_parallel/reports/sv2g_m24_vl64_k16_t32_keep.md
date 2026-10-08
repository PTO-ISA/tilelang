# sv2g_m24_vl64_k16_t32_keep

- status: **PASS**
- wall_us: 6.79
- 
- layout: launch_bounds=32; acc[] sizes=['48']; acc[48]; tir=sv2g_m24_vl64_k16_t32_keep_tir.txt; src=sv2g_m24_vl64_k16_t32_keep_source.txt
- instr: VF_SIMT us=5.06 cyc=9105 body_instr=2951 IPC_proxy=0.324 | top_opcodes: SIMT_LDS:784, SIMT_FFMA:768, SIMT_F2F:400, SIMT_PRMT:384, SIMT_IADD_I:322, SIMT_STS:48, SIMT_MOV:48, SIMT_LEA:33, MOV_XD_IMM:26, MOVK:17, SIMT_IMAD_I:16, SIMT_ISETP_I:16 | pipes: RVECEX:1993, RVECLD:784, SCALAR:92, RVECST:48, RVECLP:17, FLOWCTRL:5, PUSHQ:4, MTE2:4
