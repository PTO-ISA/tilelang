# cf7_e256_t32_pfat5_pmid20

- status: **PASS**
- wall_us: 2.19
- 
- layout: launch_bounds=32; x[] sizes=['8']; tir=cf7_e256_t32_pfat5_pmid20_tir.txt; src=cf7_e256_t32_pfat5_pmid20_source.txt; elems_per_thread≈8.00
- instr: VF_SIMT us=1.39 cyc=2501 body_instr=448 IPC_proxy=0.179 | top_opcodes: SIMT_FFMA_I:64, SIMT_FMUL:64, SIMT_MOVI:58, SIMT_BRANCH:24, SIMT_END_DVG:22, SIMT_FSETP:22, MOV_XD_IMM:22, SIMT_FSETP_I:16, SIMT_START_DVG:16, SIMT_SEL:14, MOVK:14, MOV_XD_SPR:13 | pipes: RVECEX:269, SCALAR:84, RVECLP:63, RVECLD:8, RVECST:8, FLOWCTRL:5, PUSHQ:4, MTE3:3
