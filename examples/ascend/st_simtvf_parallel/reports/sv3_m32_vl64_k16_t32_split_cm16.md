# sv3_m32_vl64_k16_t32_split_cm16

- status: **PASS**
- wall_us: 7.79
- 
- layout: launch_bounds=32; acc_c[] sizes=['32']; acc_c[32]; tir=sv3_m32_vl64_k16_t32_split_cm16_tir.txt; src=sv3_m32_vl64_k16_t32_split_cm16_source.txt
- instr: VF_SIMT us=5.77 cyc=10384 body_instr=3690 IPC_proxy=0.355 | top_opcodes: SIMT_LDS:1056, SIMT_FFMA:1024, SIMT_F2F:544, SIMT_PRMT:512, SIMT_IADD_I:98, SIMT_MOV:69, SIMT_LEA:66, SIMT_STS:64, SIMT_IADD:34, SIMT_SHFI:34, SIMT_BRANCH:34, SIMT_ISETP_I:32 | pipes: RVECEX:2423, RVECLD:1056, SCALAR:95, RVECST:64, RVECLP:35, FLOWCTRL:5, PUSHQ:4, MTE2:4
