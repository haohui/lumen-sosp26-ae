
/root/.cache/torch_extensions/py312_cpu/ksearch_dense_qkv_prefill_causal_h8_kv1or8_d128_ext_src_v1/ksearch_dense_qkv_prefill_causal_h8_kv1or8_d128_ext_src_v1.so:	file format elf64-x86-64

Disassembly of section .text:

000000000002f210 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_>:
   2f210: 55                           	pushq	%rbp
   2f211: 41 57                        	pushq	%r15
   2f213: 41 56                        	pushq	%r14
   2f215: 41 55                        	pushq	%r13
   2f217: 41 54                        	pushq	%r12
   2f219: 53                           	pushq	%rbx
   2f21a: 48 81 ec b8 00 00 00         	subq	$0xb8, %rsp
   2f221: 48 89 cd                     	movq	%rcx, %rbp
   2f224: f3 0f 11 44 24 64            	movss	%xmm0, 0x64(%rsp)
   2f22a: 49 89 d7                     	movq	%rdx, %r15
   2f22d: 49 89 f4                     	movq	%rsi, %r12
   2f230: 49 89 fe                     	movq	%rdi, %r14
   2f233: 48 8b 3f                     	movq	(%rdi), %rdi
   2f236: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   2f23e: 0f 88 1f 0e 00 00            	js	0x30063 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xe53>
   2f244: 80 bf ac 00 00 00 01         	cmpb	$0x1, 0xac(%rdi)
   2f24b: 0f 85 20 0e 00 00            	jne	0x30071 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xe61>
   2f251: 80 bf aa 00 00 00 01         	cmpb	$0x1, 0xaa(%rdi)
   2f258: 0f 85 13 0e 00 00            	jne	0x30071 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xe61>
   2f25e: 49 8b 3c 24                  	movq	(%r12), %rdi
   2f262: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   2f26a: 0f 88 20 0e 00 00            	js	0x30090 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xe80>
   2f270: 80 bf ac 00 00 00 01         	cmpb	$0x1, 0xac(%rdi)
   2f277: 0f 85 21 0e 00 00            	jne	0x3009e <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xe8e>
   2f27d: 80 bf aa 00 00 00 01         	cmpb	$0x1, 0xaa(%rdi)
   2f284: 0f 85 14 0e 00 00            	jne	0x3009e <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xe8e>
   2f28a: 49 8b 3f                     	movq	(%r15), %rdi
   2f28d: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   2f295: 0f 88 22 0e 00 00            	js	0x300bd <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xead>
   2f29b: 80 bf ac 00 00 00 01         	cmpb	$0x1, 0xac(%rdi)
   2f2a2: 0f 85 23 0e 00 00            	jne	0x300cb <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xebb>
   2f2a8: 80 bf aa 00 00 00 01         	cmpb	$0x1, 0xaa(%rdi)
   2f2af: 0f 85 16 0e 00 00            	jne	0x300cb <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xebb>
   2f2b5: 48 8b 7d 00                  	movq	(%rbp), %rdi
   2f2b9: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   2f2c1: 0f 88 23 0e 00 00            	js	0x300ea <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xeda>
   2f2c7: 80 bf ac 00 00 00 01         	cmpb	$0x1, 0xac(%rdi)
   2f2ce: 0f 85 24 0e 00 00            	jne	0x300f8 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xee8>
   2f2d4: 80 bf aa 00 00 00 01         	cmpb	$0x1, 0xaa(%rdi)
   2f2db: 0f 85 17 0e 00 00            	jne	0x300f8 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xee8>
   2f2e1: 49 8b 3e                     	movq	(%r14), %rdi
   2f2e4: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   2f2ec: 0f 88 25 0e 00 00            	js	0x30117 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xf07>
   2f2f2: 80 bf ac 00 00 00 00         	cmpb	$0x0, 0xac(%rdi)
   2f2f9: 0f 84 c2 17 00 00            	je	0x30ac1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x18b1>
   2f2ff: 0f b7 9f aa 00 00 00         	movzwl	0xaa(%rdi), %ebx
   2f306: 49 8b 3c 24                  	movq	(%r12), %rdi
   2f30a: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   2f312: 0f 88 19 0e 00 00            	js	0x30131 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xf21>
   2f318: 80 bf ac 00 00 00 00         	cmpb	$0x0, 0xac(%rdi)
   2f31f: 0f 84 9c 17 00 00            	je	0x30ac1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x18b1>
   2f325: 0f b7 87 aa 00 00 00         	movzwl	0xaa(%rdi), %eax
   2f32c: 66 39 c3                     	cmpw	%ax, %bx
   2f32f: 0f 85 60 0e 00 00            	jne	0x30195 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xf85>
   2f335: 49 8b 3e                     	movq	(%r14), %rdi
   2f338: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   2f340: 0f 88 fc 0d 00 00            	js	0x30142 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xf32>
   2f346: 80 bf ac 00 00 00 00         	cmpb	$0x0, 0xac(%rdi)
   2f34d: 0f 84 6e 17 00 00            	je	0x30ac1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x18b1>
   2f353: 0f b7 9f aa 00 00 00         	movzwl	0xaa(%rdi), %ebx
   2f35a: 49 8b 3f                     	movq	(%r15), %rdi
   2f35d: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   2f365: 0f 88 f0 0d 00 00            	js	0x3015b <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xf4b>
   2f36b: 80 bf ac 00 00 00 00         	cmpb	$0x0, 0xac(%rdi)
   2f372: 0f 84 49 17 00 00            	je	0x30ac1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x18b1>
   2f378: 0f b7 87 aa 00 00 00         	movzwl	0xaa(%rdi), %eax
   2f37f: 66 39 c3                     	cmpw	%ax, %bx
   2f382: 0f 85 0d 0e 00 00            	jne	0x30195 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xf85>
   2f388: 49 8b 3e                     	movq	(%r14), %rdi
   2f38b: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   2f393: 0f 88 d3 0d 00 00            	js	0x3016c <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xf5c>
   2f399: 80 bf ac 00 00 00 00         	cmpb	$0x0, 0xac(%rdi)
   2f3a0: 0f 84 1b 17 00 00            	je	0x30ac1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x18b1>
   2f3a6: 0f b7 9f aa 00 00 00         	movzwl	0xaa(%rdi), %ebx
   2f3ad: 48 8b 7d 00                  	movq	(%rbp), %rdi
   2f3b1: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   2f3b9: 0f 88 c7 0d 00 00            	js	0x30186 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xf76>
   2f3bf: 80 bf ac 00 00 00 00         	cmpb	$0x0, 0xac(%rdi)
   2f3c6: 0f 84 f5 16 00 00            	je	0x30ac1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x18b1>
   2f3cc: 0f b7 87 aa 00 00 00         	movzwl	0xaa(%rdi), %eax
   2f3d3: 66 39 c3                     	cmpw	%ax, %bx
   2f3d6: 0f 85 b9 0d 00 00            	jne	0x30195 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xf85>
   2f3dc: 49 8b 3e                     	movq	(%r14), %rdi
   2f3df: 0f b7 87 a8 00 00 00         	movzwl	0xa8(%rdi), %eax
   2f3e6: 66 83 f8 2f                  	cmpw	$0x2f, %ax
   2f3ea: 0f 83 6c 0c 00 00            	jae	0x3005c <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xe4c>
   2f3f0: 66 83 f8 0f                  	cmpw	$0xf, %ax
   2f3f4: 0f 85 24 17 00 00            	jne	0x30b1e <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x190e>
   2f3fa: 49 8b 04 24                  	movq	(%r12), %rax
   2f3fe: 0f b7 80 a8 00 00 00         	movzwl	0xa8(%rax), %eax
   2f405: 66 83 f8 2f                  	cmpw	$0x2f, %ax
   2f409: 0f 83 4d 0c 00 00            	jae	0x3005c <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xe4c>
   2f40f: 66 83 f8 0f                  	cmpw	$0xf, %ax
   2f413: 0f 85 24 17 00 00            	jne	0x30b3d <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x192d>
   2f419: 49 8b 07                     	movq	(%r15), %rax
   2f41c: 0f b7 80 a8 00 00 00         	movzwl	0xa8(%rax), %eax
   2f423: 66 83 f8 2f                  	cmpw	$0x2f, %ax
   2f427: 0f 83 2f 0c 00 00            	jae	0x3005c <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xe4c>
   2f42d: 66 83 f8 0f                  	cmpw	$0xf, %ax
   2f431: 0f 85 25 17 00 00            	jne	0x30b5c <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x194c>
   2f437: 48 8b 45 00                  	movq	(%rbp), %rax
   2f43b: 0f b7 80 a8 00 00 00         	movzwl	0xa8(%rax), %eax
   2f442: 66 83 f8 2f                  	cmpw	$0x2f, %ax
   2f446: 0f 83 10 0c 00 00            	jae	0x3005c <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xe4c>
   2f44c: 66 83 f8 0f                  	cmpw	$0xf, %ax
   2f450: 0f 85 25 17 00 00            	jne	0x30b7b <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x196b>
   2f456: 0f b7 87 ad 00 00 00         	movzwl	0xad(%rdi), %eax
   2f45d: a9 00 0c 00 00               	testl	$0xc00, %eax            # imm = 0xC00
   2f462: 0f 85 4c 0d 00 00            	jne	0x301b4 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xfa4>
   2f468: a9 00 10 00 00               	testl	$0x1000, %eax           # imm = 0x1000
   2f46d: 0f 85 27 0a 00 00            	jne	0x2fe9a <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xc8a>
   2f473: a8 01                        	testb	$0x1, %al
   2f475: 0f 84 63 0a 00 00            	je	0x2fede <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xcce>
   2f47b: 49 8b 3c 24                  	movq	(%r12), %rdi
   2f47f: 0f b7 87 ad 00 00 00         	movzwl	0xad(%rdi), %eax
   2f486: a9 00 0c 00 00               	testl	$0xc00, %eax            # imm = 0xC00
   2f48b: 0f 85 37 0d 00 00            	jne	0x301c8 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xfb8>
   2f491: a9 00 10 00 00               	testl	$0x1000, %eax           # imm = 0x1000
   2f496: 0f 85 61 0a 00 00            	jne	0x2fefd <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xced>
   2f49c: a8 01                        	testb	$0x1, %al
   2f49e: 0f 84 9d 0a 00 00            	je	0x2ff41 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xd31>
   2f4a4: 49 8b 3f                     	movq	(%r15), %rdi
   2f4a7: 0f b7 87 ad 00 00 00         	movzwl	0xad(%rdi), %eax
   2f4ae: a9 00 0c 00 00               	testl	$0xc00, %eax            # imm = 0xC00
   2f4b3: 0f 85 23 0d 00 00            	jne	0x301dc <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xfcc>
   2f4b9: a9 00 10 00 00               	testl	$0x1000, %eax           # imm = 0x1000
   2f4be: 0f 85 9c 0a 00 00            	jne	0x2ff60 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xd50>
   2f4c4: a8 01                        	testb	$0x1, %al
   2f4c6: 0f 84 d8 0a 00 00            	je	0x2ffa4 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xd94>
   2f4cc: 48 8b 7d 00                  	movq	(%rbp), %rdi
   2f4d0: 0f b7 87 ad 00 00 00         	movzwl	0xad(%rdi), %eax
   2f4d7: a9 00 0c 00 00               	testl	$0xc00, %eax            # imm = 0xC00
   2f4dc: 0f 85 0e 0d 00 00            	jne	0x301f0 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xfe0>
   2f4e2: a9 00 10 00 00               	testl	$0x1000, %eax           # imm = 0x1000
   2f4e7: 0f 85 d6 0a 00 00            	jne	0x2ffc3 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xdb3>
   2f4ed: a8 01                        	testb	$0x1, %al
   2f4ef: 0f 84 12 0b 00 00            	je	0x30007 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xdf7>
   2f4f5: 49 8b 3e                     	movq	(%r14), %rdi
   2f4f8: f6 87 ae 00 00 00 08         	testb	$0x8, 0xae(%rdi)
   2f4ff: 0f 85 ff 0c 00 00            	jne	0x30204 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xff4>
   2f505: 48 8b 47 40                  	movq	0x40(%rdi), %rax
   2f509: 48 83 f8 04                  	cmpq	$0x4, %rax
   2f50d: 0f 85 37 0d 00 00            	jne	0x3024a <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x103a>
   2f513: 49 8b 3c 24                  	movq	(%r12), %rdi
   2f517: f6 87 ae 00 00 00 08         	testb	$0x8, 0xae(%rdi)
   2f51e: 0f 85 f2 0c 00 00            	jne	0x30216 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1006>
   2f524: 48 8b 47 40                  	movq	0x40(%rdi), %rax
   2f528: 48 83 f8 04                  	cmpq	$0x4, %rax
   2f52c: 0f 85 18 0d 00 00            	jne	0x3024a <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x103a>
   2f532: 49 8b 3f                     	movq	(%r15), %rdi
   2f535: f6 87 ae 00 00 00 08         	testb	$0x8, 0xae(%rdi)
   2f53c: 0f 85 e6 0c 00 00            	jne	0x30228 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1018>
   2f542: 48 8b 47 40                  	movq	0x40(%rdi), %rax
   2f546: 48 83 f8 04                  	cmpq	$0x4, %rax
   2f54a: 0f 85 fa 0c 00 00            	jne	0x3024a <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x103a>
   2f550: 48 8b 7d 00                  	movq	(%rbp), %rdi
   2f554: f6 87 ae 00 00 00 08         	testb	$0x8, 0xae(%rdi)
   2f55b: 0f 85 d9 0c 00 00            	jne	0x3023a <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x102a>
   2f561: 48 8b 47 40                  	movq	0x40(%rdi), %rax
   2f565: 48 83 f8 04                  	cmpq	$0x4, %rax
   2f569: 0f 85 db 0c 00 00            	jne	0x3024a <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x103a>
   2f56f: 49 8b 1e                     	movq	(%r14), %rbx
   2f572: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   2f579: 0f 85 ea 0c 00 00            	jne	0x30269 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1059>
   2f57f: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   2f583: 48 85 f6                     	testq	%rsi, %rsi
   2f586: 0f 8e b7 0f 00 00            	jle	0x30543 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1333>
   2f58c: 31 c0                        	xorl	%eax, %eax
   2f58e: 48 83 c3 48                  	addq	$0x48, %rbx
   2f592: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2f596: 0f 83 c2 0f 00 00            	jae	0x3055e <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x134e>
   2f59c: 4c 8b 2c c3                  	movq	(%rbx,%rax,8), %r13
   2f5a0: 49 8b 1c 24                  	movq	(%r12), %rbx
   2f5a4: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   2f5ab: 0f 85 d7 0c 00 00            	jne	0x30288 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1078>
   2f5b1: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   2f5b5: 48 85 f6                     	testq	%rsi, %rsi
   2f5b8: 0f 8e bd 0f 00 00            	jle	0x3057b <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x136b>
   2f5be: 31 c0                        	xorl	%eax, %eax
   2f5c0: 48 83 c3 48                  	addq	$0x48, %rbx
   2f5c4: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2f5c8: 0f 83 c8 0f 00 00            	jae	0x30596 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1386>
   2f5ce: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   2f5d2: 49 39 c5                     	cmpq	%rax, %r13
   2f5d5: 0f 85 2f 10 00 00            	jne	0x3060a <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x13fa>
   2f5db: 49 8b 1e                     	movq	(%r14), %rbx
   2f5de: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   2f5e5: 0f 85 b6 0c 00 00            	jne	0x302a1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1091>
   2f5eb: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   2f5ef: 48 85 f6                     	testq	%rsi, %rsi
   2f5f2: 0f 8e b0 0f 00 00            	jle	0x305a8 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1398>
   2f5f8: 31 c0                        	xorl	%eax, %eax
   2f5fa: 48 83 c3 48                  	addq	$0x48, %rbx
   2f5fe: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2f602: 0f 83 bb 0f 00 00            	jae	0x305c3 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x13b3>
   2f608: 4c 8b 2c c3                  	movq	(%rbx,%rax,8), %r13
   2f60c: 49 8b 1f                     	movq	(%r15), %rbx
   2f60f: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   2f616: 0f 85 a3 0c 00 00            	jne	0x302bf <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x10af>
   2f61c: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   2f620: 48 85 f6                     	testq	%rsi, %rsi
   2f623: 0f 8e b6 0f 00 00            	jle	0x305df <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x13cf>
   2f629: 31 c0                        	xorl	%eax, %eax
   2f62b: 48 83 c3 48                  	addq	$0x48, %rbx
   2f62f: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2f633: 0f 83 c1 0f 00 00            	jae	0x305fa <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x13ea>
   2f639: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   2f63d: 49 39 c5                     	cmpq	%rax, %r13
   2f640: 0f 85 c4 0f 00 00            	jne	0x3060a <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x13fa>
   2f646: 49 8b 1e                     	movq	(%r14), %rbx
   2f649: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   2f650: 0f 85 82 0c 00 00            	jne	0x302d8 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x10c8>
   2f656: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   2f65a: b8 02 00 00 00               	movl	$0x2, %eax
   2f65f: 48 83 fe 02                  	cmpq	$0x2, %rsi
   2f663: 0f 8e c0 0f 00 00            	jle	0x30629 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1419>
   2f669: 48 83 c3 48                  	addq	$0x48, %rbx
   2f66d: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2f671: 0f 83 d0 0f 00 00            	jae	0x30647 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1437>
   2f677: 4c 8b 2c c3                  	movq	(%rbx,%rax,8), %r13
   2f67b: 49 8b 1c 24                  	movq	(%r12), %rbx
   2f67f: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   2f686: 0f 85 6e 0c 00 00            	jne	0x302fa <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x10ea>
   2f68c: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   2f690: b8 02 00 00 00               	movl	$0x2, %eax
   2f695: 48 83 fe 02                  	cmpq	$0x2, %rsi
   2f699: 0f 8e c5 0f 00 00            	jle	0x30664 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1454>
   2f69f: 48 83 c3 48                  	addq	$0x48, %rbx
   2f6a3: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2f6a7: 0f 83 d5 0f 00 00            	jae	0x30682 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1472>
   2f6ad: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   2f6b1: 49 39 c5                     	cmpq	%rax, %r13
   2f6b4: 0f 85 42 10 00 00            	jne	0x306fc <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x14ec>
   2f6ba: 49 8b 1e                     	movq	(%r14), %rbx
   2f6bd: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   2f6c4: 0f 85 4c 0c 00 00            	jne	0x30316 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1106>
   2f6ca: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   2f6ce: b8 02 00 00 00               	movl	$0x2, %eax
   2f6d3: 48 83 fe 02                  	cmpq	$0x2, %rsi
   2f6d7: 0f 8e b7 0f 00 00            	jle	0x30694 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1484>
   2f6dd: 48 83 c3 48                  	addq	$0x48, %rbx
   2f6e1: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2f6e5: 0f 83 c7 0f 00 00            	jae	0x306b2 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x14a2>
   2f6eb: 4c 8b 2c c3                  	movq	(%rbx,%rax,8), %r13
   2f6ef: 49 8b 1f                     	movq	(%r15), %rbx
   2f6f2: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   2f6f9: 0f 85 38 0c 00 00            	jne	0x30337 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1127>
   2f6ff: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   2f703: b8 02 00 00 00               	movl	$0x2, %eax
   2f708: 48 83 fe 02                  	cmpq	$0x2, %rsi
   2f70c: 0f 8e bc 0f 00 00            	jle	0x306ce <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x14be>
   2f712: 48 83 c3 48                  	addq	$0x48, %rbx
   2f716: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2f71a: 0f 83 cc 0f 00 00            	jae	0x306ec <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x14dc>
   2f720: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   2f724: 49 39 c5                     	cmpq	%rax, %r13
   2f727: 0f 85 cf 0f 00 00            	jne	0x306fc <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x14ec>
   2f72d: 49 8b 1e                     	movq	(%r14), %rbx
   2f730: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   2f737: 0f 85 16 0c 00 00            	jne	0x30353 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1143>
   2f73d: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   2f741: b8 03 00 00 00               	movl	$0x3, %eax
   2f746: 48 83 fe 03                  	cmpq	$0x3, %rsi
   2f74a: 0f 8e cb 0f 00 00            	jle	0x3071b <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x150b>
   2f750: 48 83 c3 48                  	addq	$0x48, %rbx
   2f754: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2f758: 0f 83 db 0f 00 00            	jae	0x30739 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1529>
   2f75e: 4c 8b 2c c3                  	movq	(%rbx,%rax,8), %r13
   2f762: 49 8b 1c 24                  	movq	(%r12), %rbx
   2f766: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   2f76d: 0f 85 02 0c 00 00            	jne	0x30375 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1165>
   2f773: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   2f777: b8 03 00 00 00               	movl	$0x3, %eax
   2f77c: 48 83 fe 03                  	cmpq	$0x3, %rsi
   2f780: 0f 8e d0 0f 00 00            	jle	0x30756 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1546>
   2f786: 48 83 c3 48                  	addq	$0x48, %rbx
   2f78a: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2f78e: 0f 83 e0 0f 00 00            	jae	0x30774 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1564>
   2f794: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   2f798: 49 39 c5                     	cmpq	%rax, %r13
   2f79b: 0f 85 4d 10 00 00            	jne	0x307ee <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x15de>
   2f7a1: 49 8b 1e                     	movq	(%r14), %rbx
   2f7a4: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   2f7ab: 0f 85 e0 0b 00 00            	jne	0x30391 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1181>
   2f7b1: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   2f7b5: b8 03 00 00 00               	movl	$0x3, %eax
   2f7ba: 48 83 fe 03                  	cmpq	$0x3, %rsi
   2f7be: 0f 8e c2 0f 00 00            	jle	0x30786 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1576>
   2f7c4: 48 83 c3 48                  	addq	$0x48, %rbx
   2f7c8: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2f7cc: 0f 83 d2 0f 00 00            	jae	0x307a4 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1594>
   2f7d2: 4c 8b 2c c3                  	movq	(%rbx,%rax,8), %r13
   2f7d6: 49 8b 1f                     	movq	(%r15), %rbx
   2f7d9: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   2f7e0: 0f 85 cc 0b 00 00            	jne	0x303b2 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x11a2>
   2f7e6: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   2f7ea: b8 03 00 00 00               	movl	$0x3, %eax
   2f7ef: 48 83 fe 03                  	cmpq	$0x3, %rsi
   2f7f3: 0f 8e c7 0f 00 00            	jle	0x307c0 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x15b0>
   2f7f9: 48 83 c3 48                  	addq	$0x48, %rbx
   2f7fd: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2f801: 0f 83 d7 0f 00 00            	jae	0x307de <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x15ce>
   2f807: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   2f80b: 49 39 c5                     	cmpq	%rax, %r13
   2f80e: 0f 85 da 0f 00 00            	jne	0x307ee <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x15de>
   2f814: 49 8b 1e                     	movq	(%r14), %rbx
   2f817: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   2f81e: 0f 85 aa 0b 00 00            	jne	0x303ce <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x11be>
   2f824: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   2f828: b8 01 00 00 00               	movl	$0x1, %eax
   2f82d: 48 83 fe 01                  	cmpq	$0x1, %rsi
   2f831: 0f 8e d6 0f 00 00            	jle	0x3080d <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x15fd>
   2f837: 48 83 c3 48                  	addq	$0x48, %rbx
   2f83b: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2f83f: 0f 83 e6 0f 00 00            	jae	0x3082b <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x161b>
   2f845: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   2f849: 48 83 f8 08                  	cmpq	$0x8, %rax
   2f84d: 0f 85 93 0b 00 00            	jne	0x303e6 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x11d6>
   2f853: 49 8b 1c 24                  	movq	(%r12), %rbx
   2f857: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   2f85e: 0f 85 a1 0b 00 00            	jne	0x30405 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x11f5>
   2f864: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   2f868: b8 01 00 00 00               	movl	$0x1, %eax
   2f86d: 48 83 fe 01                  	cmpq	$0x1, %rsi
   2f871: 0f 8e ca 0f 00 00            	jle	0x30841 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1631>
   2f877: 48 83 c3 48                  	addq	$0x48, %rbx
   2f87b: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2f87f: 0f 83 da 0f 00 00            	jae	0x3085f <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x164f>
   2f885: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   2f889: 48 83 f8 01                  	cmpq	$0x1, %rax
   2f88d: 74 40                        	je	0x2f8cf <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x6bf>
   2f88f: 49 8b 1c 24                  	movq	(%r12), %rbx
   2f893: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   2f89a: 0f 85 79 11 00 00            	jne	0x30a19 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1809>
   2f8a0: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   2f8a4: b8 01 00 00 00               	movl	$0x1, %eax
   2f8a9: 48 83 fe 01                  	cmpq	$0x1, %rsi
   2f8ad: 0f 8e c0 11 00 00            	jle	0x30a73 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1863>
   2f8b3: 48 83 c3 48                  	addq	$0x48, %rbx
   2f8b7: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2f8bb: 0f 83 d0 11 00 00            	jae	0x30a91 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1881>
   2f8c1: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   2f8c5: 48 83 f8 08                  	cmpq	$0x8, %rax
   2f8c9: 0f 85 d3 11 00 00            	jne	0x30aa2 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1892>
   2f8cf: 49 8b 1f                     	movq	(%r15), %rbx
   2f8d2: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   2f8d9: 0f 85 43 0b 00 00            	jne	0x30422 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1212>
   2f8df: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   2f8e3: b8 01 00 00 00               	movl	$0x1, %eax
   2f8e8: 48 83 fe 01                  	cmpq	$0x1, %rsi
   2f8ec: 0f 8e 83 0f 00 00            	jle	0x30875 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1665>
   2f8f2: 48 83 c3 48                  	addq	$0x48, %rbx
   2f8f6: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2f8fa: 0f 83 93 0f 00 00            	jae	0x30893 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1683>
   2f900: 4c 8b 2c c3                  	movq	(%rbx,%rax,8), %r13
   2f904: 49 8b 1c 24                  	movq	(%r12), %rbx
   2f908: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   2f90f: 0f 85 2f 0b 00 00            	jne	0x30444 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1234>
   2f915: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   2f919: b8 01 00 00 00               	movl	$0x1, %eax
   2f91e: 48 83 fe 01                  	cmpq	$0x1, %rsi
   2f922: 0f 8e 88 0f 00 00            	jle	0x308b0 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x16a0>
   2f928: 48 83 c3 48                  	addq	$0x48, %rbx
   2f92c: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2f930: 0f 83 98 0f 00 00            	jae	0x308ce <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x16be>
   2f936: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   2f93a: 49 39 c5                     	cmpq	%rax, %r13
   2f93d: 0f 85 18 0b 00 00            	jne	0x3045b <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x124b>
   2f943: 49 8b 1e                     	movq	(%r14), %rbx
   2f946: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   2f94d: 0f 85 27 0b 00 00            	jne	0x3047a <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x126a>
   2f953: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   2f957: b8 03 00 00 00               	movl	$0x3, %eax
   2f95c: 48 83 fe 03                  	cmpq	$0x3, %rsi
   2f960: 0f 8e 7d 0f 00 00            	jle	0x308e3 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x16d3>
   2f966: 48 83 c3 48                  	addq	$0x48, %rbx
   2f96a: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2f96e: 0f 83 8d 0f 00 00            	jae	0x30901 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x16f1>
   2f974: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   2f978: 48 3d 80 00 00 00            	cmpq	$0x80, %rax
   2f97e: 0f 85 10 0b 00 00            	jne	0x30494 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1284>
   2f984: 48 8b 7d 00                  	movq	(%rbp), %rdi
   2f988: f6 87 ae 00 00 00 08         	testb	$0x8, 0xae(%rdi)
   2f98f: 0f 85 1e 0b 00 00            	jne	0x304b3 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x12a3>
   2f995: 48 8b 57 40                  	movq	0x40(%rdi), %rdx
   2f999: 48 83 c7 48                  	addq	$0x48, %rdi
   2f99d: 48 83 fa 06                  	cmpq	$0x6, %rdx
   2f9a1: 0f 83 72 0f 00 00            	jae	0x30919 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1709>
   2f9a7: 49 8b 36                     	movq	(%r14), %rsi
   2f9aa: f6 86 ae 00 00 00 08         	testb	$0x8, 0xae(%rsi)
   2f9b1: 0f 85 0a 0b 00 00            	jne	0x304c1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x12b1>
   2f9b7: 48 8b 46 40                  	movq	0x40(%rsi), %rax
   2f9bb: 48 83 c6 48                  	addq	$0x48, %rsi
   2f9bf: 48 83 f8 06                  	cmpq	$0x6, %rax
   2f9c3: 0f 83 82 0f 00 00            	jae	0x3094b <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x173b>
   2f9c9: 48 39 c2                     	cmpq	%rax, %rdx
   2f9cc: 0f 85 2d 11 00 00            	jne	0x30aff <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x18ef>
   2f9d2: 48 85 d2                     	testq	%rdx, %rdx
   2f9d5: 74 11                        	je	0x2f9e8 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x7d8>
   2f9d7: 48 c1 e2 03                  	shlq	$0x3, %rdx
   2f9db: e8 90 23 00 00               	callq	0x31d70 <bcmp@plt>
   2f9e0: 85 c0                        	testl	%eax, %eax
   2f9e2: 0f 85 17 11 00 00            	jne	0x30aff <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x18ef>
   2f9e8: 49 8b 3e                     	movq	(%r14), %rdi
   2f9eb: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   2f9f3: 0f 88 e8 0a 00 00            	js	0x304e1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x12d1>
   2f9f9: 80 bf ac 00 00 00 00         	cmpb	$0x0, 0xac(%rdi)
   2fa00: 0f 84 bb 10 00 00            	je	0x30ac1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x18b1>
   2fa06: 0f b7 b7 aa 00 00 00         	movzwl	0xaa(%rdi), %esi
   2fa0d: 48 8d bc 24 90 00 00 00      	leaq	0x90(%rsp), %rdi
   2fa15: e8 46 15 00 00               	callq	0x30f60 <_ZN3c104impl17InlineDeviceGuardINS0_16VirtualGuardImplEEC2ENS_6DeviceE@plt>
   2fa1a: c6 84 24 a8 00 00 00 01      	movb	$0x1, 0xa8(%rsp)
   2fa22: 4d 8b 2e                     	movq	(%r14), %r13
   2fa25: 41 f6 85 ae 00 00 00 08      	testb	$0x8, 0xae(%r13)
   2fa2d: 48 89 ac 24 b0 00 00 00      	movq	%rbp, 0xb0(%rsp)
   2fa35: 0f 85 b3 0a 00 00            	jne	0x304ee <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x12de>
   2fa3b: 49 8b 75 40                  	movq	0x40(%r13), %rsi
   2fa3f: 48 85 f6                     	testq	%rsi, %rsi
   2fa42: 0f 8e 35 0f 00 00            	jle	0x3097d <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x176d>
   2fa48: 31 c0                        	xorl	%eax, %eax
   2fa4a: 49 83 c5 48                  	addq	$0x48, %r13
   2fa4e: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2fa52: 0f 83 40 0f 00 00            	jae	0x30998 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1788>
   2fa58: 4d 8b 6c c5 00               	movq	(%r13,%rax,8), %r13
   2fa5d: 49 8b 2e                     	movq	(%r14), %rbp
   2fa60: f6 85 ae 00 00 00 08         	testb	$0x8, 0xae(%rbp)
   2fa67: 0f 85 a0 0a 00 00            	jne	0x3050d <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x12fd>
   2fa6d: 48 8b 75 40                  	movq	0x40(%rbp), %rsi
   2fa71: b8 02 00 00 00               	movl	$0x2, %eax
   2fa76: 48 83 fe 02                  	cmpq	$0x2, %rsi
   2fa7a: 0f 8e 36 0f 00 00            	jle	0x309b6 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x17a6>
   2fa80: 48 83 c5 48                  	addq	$0x48, %rbp
   2fa84: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2fa88: 0f 83 46 0f 00 00            	jae	0x309d4 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x17c4>
   2fa8e: 48 8b 6c c5 00               	movq	(%rbp,%rax,8), %rbp
   2fa93: 49 8b 1c 24                  	movq	(%r12), %rbx
   2fa97: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   2fa9e: 0f 85 8c 0a 00 00            	jne	0x30530 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1320>
   2faa4: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   2faa8: b8 01 00 00 00               	movl	$0x1, %eax
   2faad: 48 83 fe 01                  	cmpq	$0x1, %rsi
   2fab1: 0f 8e 3c 0f 00 00            	jle	0x309f3 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x17e3>
   2fab7: 48 83 c3 48                  	addq	$0x48, %rbx
   2fabb: 48 83 fe 06                  	cmpq	$0x6, %rsi
   2fabf: 0f 83 4c 0f 00 00            	jae	0x30a11 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1801>
   2fac5: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   2fac9: 4d 85 ed                     	testq	%r13, %r13
   2facc: 0f 84 8d 03 00 00            	je	0x2fe5f <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xc4f>
   2fad2: 48 85 ed                     	testq	%rbp, %rbp
   2fad5: 0f 84 84 03 00 00            	je	0x2fe5f <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xc4f>
   2fadb: 49 8b 0c 24                  	movq	(%r12), %rcx
   2fadf: 48 89 4c 24 28               	movq	%rcx, 0x28(%rsp)
   2fae4: 48 3b 0d 25 38 00 00         	cmpq	0x3825(%rip), %rcx      # 0x33310 <strncmp+0x33310>
   2faeb: 74 12                        	je	0x2faff <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x8ef>
   2faed: ba 01 00 00 00               	movl	$0x1, %edx
   2faf2: f0                           	lock
   2faf3: 0f c1 51 08                  	xaddl	%edx, 0x8(%rcx)
   2faf7: 85 d2                        	testl	%edx, %edx
   2faf9: 0f 84 db 10 00 00            	je	0x30bda <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x19ca>
   2faff: 49 8b 0f                     	movq	(%r15), %rcx
   2fb02: 48 89 4c 24 20               	movq	%rcx, 0x20(%rsp)
   2fb07: 48 3b 0d 02 38 00 00         	cmpq	0x3802(%rip), %rcx      # 0x33310 <strncmp+0x33310>
   2fb0e: 74 12                        	je	0x2fb22 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x912>
   2fb10: ba 01 00 00 00               	movl	$0x1, %edx
   2fb15: f0                           	lock
   2fb16: 0f c1 51 08                  	xaddl	%edx, 0x8(%rcx)
   2fb1a: 85 d2                        	testl	%edx, %edx
   2fb1c: 0f 84 de 10 00 00            	je	0x30c00 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x19f0>
   2fb22: 48 83 f8 01                  	cmpq	$0x1, %rax
   2fb26: 0f 85 a2 01 00 00            	jne	0x2fcce <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xabe>
   2fb2c: 48 b8 ff ff ff ff ff ff ff bf	movabsq	$-0x4000000000000001, %rax # imm = 0xBFFFFFFFFFFFFFFF
   2fb36: 4c 89 6c 24 40               	movq	%r13, 0x40(%rsp)
   2fb3b: 48 c7 44 24 48 08 00 00 00   	movq	$0x8, 0x48(%rsp)
   2fb44: 48 89 6c 24 50               	movq	%rbp, 0x50(%rsp)
   2fb49: 48 c7 44 24 58 80 00 00 00   	movq	$0x80, 0x58(%rsp)
   2fb52: 4c 89 ac 24 88 00 00 00      	movq	%r13, 0x88(%rsp)
   2fb5a: 49 39 c5                     	cmpq	%rax, %r13
   2fb5d: 0f 8e 37 10 00 00            	jle	0x30b9a <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x198a>
   2fb63: 48 89 ac 24 88 00 00 00      	movq	%rbp, 0x88(%rsp)
   2fb6b: 48 39 c5                     	cmpq	%rax, %rbp
   2fb6e: 0f 8e 26 10 00 00            	jle	0x30b9a <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x198a>
   2fb74: 48 8d 7c 24 38               	leaq	0x38(%rsp), %rdi
   2fb79: 48 8d 54 24 40               	leaq	0x40(%rsp), %rdx
   2fb7e: b9 04 00 00 00               	movl	$0x4, %ecx
   2fb83: 4c 89 e6                     	movq	%r12, %rsi
   2fb86: 45 31 c0                     	xorl	%r8d, %r8d
   2fb89: e8 f2 21 00 00               	callq	0x31d80 <_ZN2at4_ops6expand4callERKNS_6TensorEN3c108ArrayRefINS5_6SymIntEEEb@plt>
   2fb8e: 48 8b 44 24 38               	movq	0x38(%rsp), %rax
   2fb93: 4c 8b 25 76 37 00 00         	movq	0x3776(%rip), %r12      # 0x33310 <strncmp+0x33310>
   2fb9a: 4c 89 64 24 38               	movq	%r12, 0x38(%rsp)
   2fb9f: 48 8b 5c 24 28               	movq	0x28(%rsp), %rbx
   2fba4: 48 89 44 24 28               	movq	%rax, 0x28(%rsp)
   2fba9: 4c 39 e3                     	cmpq	%r12, %rbx
   2fbac: 74 67                        	je	0x2fc15 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xa05>
   2fbae: f0                           	lock
   2fbaf: ff 4b 08                     	decl	0x8(%rbx)
   2fbb2: 75 1a                        	jne	0x2fbce <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x9be>
   2fbb4: 8b 43 0c                     	movl	0xc(%rbx), %eax
   2fbb7: 83 f8 01                     	cmpl	$0x1, %eax
   2fbba: 0f 85 66 04 00 00            	jne	0x30026 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xe16>
   2fbc0: 48 85 db                     	testq	%rbx, %rbx
   2fbc3: 74 09                        	je	0x2fbce <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x9be>
   2fbc5: 48 8b 03                     	movq	(%rbx), %rax
   2fbc8: 48 89 df                     	movq	%rbx, %rdi
   2fbcb: ff 50 08                     	callq	*0x8(%rax)
   2fbce: 48 8b 44 24 38               	movq	0x38(%rsp), %rax
   2fbd3: 48 3b 05 36 37 00 00         	cmpq	0x3736(%rip), %rax      # 0x33310 <strncmp+0x33310>
   2fbda: 74 39                        	je	0x2fc15 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xa05>
   2fbdc: f0                           	lock
   2fbdd: ff 48 08                     	decl	0x8(%rax)
   2fbe0: 75 33                        	jne	0x2fc15 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xa05>
   2fbe2: 48 8b 44 24 38               	movq	0x38(%rsp), %rax
   2fbe7: 8b 40 0c                     	movl	0xc(%rax), %eax
   2fbea: 83 f8 01                     	cmpl	$0x1, %eax
   2fbed: 74 16                        	je	0x2fc05 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x9f5>
   2fbef: 48 8b 7c 24 38               	movq	0x38(%rsp), %rdi
   2fbf4: 48 8b 07                     	movq	(%rdi), %rax
   2fbf7: ff 50 10                     	callq	*0x10(%rax)
   2fbfa: 48 8b 44 24 38               	movq	0x38(%rsp), %rax
   2fbff: f0                           	lock
   2fc00: ff 48 0c                     	decl	0xc(%rax)
   2fc03: 75 10                        	jne	0x2fc15 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xa05>
   2fc05: 48 8b 7c 24 38               	movq	0x38(%rsp), %rdi
   2fc0a: 48 85 ff                     	testq	%rdi, %rdi
   2fc0d: 74 06                        	je	0x2fc15 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xa05>
   2fc0f: 48 8b 07                     	movq	(%rdi), %rax
   2fc12: ff 50 08                     	callq	*0x8(%rax)
   2fc15: 4c 89 6c 24 68               	movq	%r13, 0x68(%rsp)
   2fc1a: 48 c7 44 24 70 08 00 00 00   	movq	$0x8, 0x70(%rsp)
   2fc23: 48 89 6c 24 78               	movq	%rbp, 0x78(%rsp)
   2fc28: 48 c7 84 24 80 00 00 00 80 00 00 00  	movq	$0x80, 0x80(%rsp)
   2fc34: 48 8d 7c 24 40               	leaq	0x40(%rsp), %rdi
   2fc39: 48 8d 54 24 68               	leaq	0x68(%rsp), %rdx
   2fc3e: b9 04 00 00 00               	movl	$0x4, %ecx
   2fc43: 4c 89 fe                     	movq	%r15, %rsi
   2fc46: 45 31 c0                     	xorl	%r8d, %r8d
   2fc49: e8 32 21 00 00               	callq	0x31d80 <_ZN2at4_ops6expand4callERKNS_6TensorEN3c108ArrayRefINS5_6SymIntEEEb@plt>
   2fc4e: 48 8b 44 24 40               	movq	0x40(%rsp), %rax
   2fc53: 4c 89 64 24 40               	movq	%r12, 0x40(%rsp)
   2fc58: 48 8b 5c 24 20               	movq	0x20(%rsp), %rbx
   2fc5d: 48 89 44 24 20               	movq	%rax, 0x20(%rsp)
   2fc62: 4c 39 e3                     	cmpq	%r12, %rbx
   2fc65: 74 67                        	je	0x2fcce <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xabe>
   2fc67: f0                           	lock
   2fc68: ff 4b 08                     	decl	0x8(%rbx)
   2fc6b: 75 1a                        	jne	0x2fc87 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xa77>
   2fc6d: 8b 43 0c                     	movl	0xc(%rbx), %eax
   2fc70: 83 f8 01                     	cmpl	$0x1, %eax
   2fc73: 0f 85 c8 03 00 00            	jne	0x30041 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xe31>
   2fc79: 48 85 db                     	testq	%rbx, %rbx
   2fc7c: 74 09                        	je	0x2fc87 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xa77>
   2fc7e: 48 8b 03                     	movq	(%rbx), %rax
   2fc81: 48 89 df                     	movq	%rbx, %rdi
   2fc84: ff 50 08                     	callq	*0x8(%rax)
   2fc87: 48 8b 44 24 40               	movq	0x40(%rsp), %rax
   2fc8c: 48 3b 05 7d 36 00 00         	cmpq	0x367d(%rip), %rax      # 0x33310 <strncmp+0x33310>
   2fc93: 74 39                        	je	0x2fcce <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xabe>
   2fc95: f0                           	lock
   2fc96: ff 48 08                     	decl	0x8(%rax)
   2fc99: 75 33                        	jne	0x2fcce <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xabe>
   2fc9b: 48 8b 44 24 40               	movq	0x40(%rsp), %rax
   2fca0: 8b 40 0c                     	movl	0xc(%rax), %eax
   2fca3: 83 f8 01                     	cmpl	$0x1, %eax
   2fca6: 74 16                        	je	0x2fcbe <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xaae>
   2fca8: 48 8b 7c 24 40               	movq	0x40(%rsp), %rdi
   2fcad: 48 8b 07                     	movq	(%rdi), %rax
   2fcb0: ff 50 10                     	callq	*0x10(%rax)
   2fcb3: 48 8b 44 24 40               	movq	0x40(%rsp), %rax
   2fcb8: f0                           	lock
   2fcb9: ff 48 0c                     	decl	0xc(%rax)
   2fcbc: 75 10                        	jne	0x2fcce <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xabe>
   2fcbe: 48 8b 7c 24 40               	movq	0x40(%rsp), %rdi
   2fcc3: 48 85 ff                     	testq	%rdi, %rdi
   2fcc6: 74 06                        	je	0x2fcce <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xabe>
   2fcc8: 48 8b 07                     	movq	(%rdi), %rax
   2fccb: ff 50 08                     	callq	*0x8(%rax)
   2fcce: c6 44 24 48 00               	movb	$0x0, 0x48(%rsp)
   2fcd3: f3 0f 10 44 24 64            	movss	0x64(%rsp), %xmm0
   2fcd9: f3 0f 5a c0                  	cvtss2sd	%xmm0, %xmm0
   2fcdd: f2 0f 11 44 24 68            	movsd	%xmm0, 0x68(%rsp)
   2fce3: c6 44 24 70 01               	movb	$0x1, 0x70(%rsp)
   2fce8: 0f 10 44 24 68               	movups	0x68(%rsp), %xmm0
   2fced: 0f 11 04 24                  	movups	%xmm0, (%rsp)
   2fcf1: c7 44 24 10 00 00 00 00      	movl	$0x0, 0x10(%rsp)
   2fcf9: 48 8d 7c 24 30               	leaq	0x30(%rsp), %rdi
   2fcfe: 48 8d 54 24 28               	leaq	0x28(%rsp), %rdx
   2fd03: 48 8d 4c 24 20               	leaq	0x20(%rsp), %rcx
   2fd08: 4c 8d 44 24 40               	leaq	0x40(%rsp), %r8
   2fd0d: 0f 57 c0                     	xorps	%xmm0, %xmm0
   2fd10: 4c 89 f6                     	movq	%r14, %rsi
   2fd13: 41 b9 01 00 00 00            	movl	$0x1, %r9d
   2fd19: e8 72 20 00 00               	callq	0x31d90 <_ZN2at4_ops28scaled_dot_product_attention4callERKNS_6TensorES4_S4_RKSt8optionalIS2_EdbS5_IdEb@plt>
   2fd1e: 0f b6 44 24 48               	movzbl	0x48(%rsp), %eax
   2fd23: c6 44 24 48 00               	movb	$0x0, 0x48(%rsp)
   2fd28: 3c 01                        	cmpb	$0x1, %al
   2fd2a: 48 8b 9c 24 b0 00 00 00      	movq	0xb0(%rsp), %rbx
   2fd32: 75 47                        	jne	0x2fd7b <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xb6b>
   2fd34: 48 8b 44 24 40               	movq	0x40(%rsp), %rax
   2fd39: 48 3b 05 d0 35 00 00         	cmpq	0x35d0(%rip), %rax      # 0x33310 <strncmp+0x33310>
   2fd40: 74 39                        	je	0x2fd7b <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xb6b>
   2fd42: f0                           	lock
   2fd43: ff 48 08                     	decl	0x8(%rax)
   2fd46: 75 33                        	jne	0x2fd7b <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xb6b>
   2fd48: 48 8b 44 24 40               	movq	0x40(%rsp), %rax
   2fd4d: 8b 40 0c                     	movl	0xc(%rax), %eax
   2fd50: 83 f8 01                     	cmpl	$0x1, %eax
   2fd53: 74 16                        	je	0x2fd6b <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xb5b>
   2fd55: 48 8b 7c 24 40               	movq	0x40(%rsp), %rdi
   2fd5a: 48 8b 07                     	movq	(%rdi), %rax
   2fd5d: ff 50 10                     	callq	*0x10(%rax)
   2fd60: 48 8b 44 24 40               	movq	0x40(%rsp), %rax
   2fd65: f0                           	lock
   2fd66: ff 48 0c                     	decl	0xc(%rax)
   2fd69: 75 10                        	jne	0x2fd7b <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xb6b>
   2fd6b: 48 8b 7c 24 40               	movq	0x40(%rsp), %rdi
   2fd70: 48 85 ff                     	testq	%rdi, %rdi
   2fd73: 74 06                        	je	0x2fd7b <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xb6b>
   2fd75: 48 8b 07                     	movq	(%rdi), %rax
   2fd78: ff 50 08                     	callq	*0x8(%rax)
   2fd7b: 48 8d 74 24 30               	leaq	0x30(%rsp), %rsi
   2fd80: 48 89 df                     	movq	%rbx, %rdi
   2fd83: 31 d2                        	xorl	%edx, %edx
   2fd85: e8 16 20 00 00               	callq	0x31da0 <_ZN2at4_ops5copy_4callERNS_6TensorERKS2_b@plt>
   2fd8a: 48 8b 44 24 30               	movq	0x30(%rsp), %rax
   2fd8f: 48 3b 05 7a 35 00 00         	cmpq	0x357a(%rip), %rax      # 0x33310 <strncmp+0x33310>
   2fd96: 74 39                        	je	0x2fdd1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xbc1>
   2fd98: f0                           	lock
   2fd99: ff 48 08                     	decl	0x8(%rax)
   2fd9c: 75 33                        	jne	0x2fdd1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xbc1>
   2fd9e: 48 8b 44 24 30               	movq	0x30(%rsp), %rax
   2fda3: 8b 40 0c                     	movl	0xc(%rax), %eax
   2fda6: 83 f8 01                     	cmpl	$0x1, %eax
   2fda9: 74 16                        	je	0x2fdc1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xbb1>
   2fdab: 48 8b 7c 24 30               	movq	0x30(%rsp), %rdi
   2fdb0: 48 8b 07                     	movq	(%rdi), %rax
   2fdb3: ff 50 10                     	callq	*0x10(%rax)
   2fdb6: 48 8b 44 24 30               	movq	0x30(%rsp), %rax
   2fdbb: f0                           	lock
   2fdbc: ff 48 0c                     	decl	0xc(%rax)
   2fdbf: 75 10                        	jne	0x2fdd1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xbc1>
   2fdc1: 48 8b 7c 24 30               	movq	0x30(%rsp), %rdi
   2fdc6: 48 85 ff                     	testq	%rdi, %rdi
   2fdc9: 74 06                        	je	0x2fdd1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xbc1>
   2fdcb: 48 8b 07                     	movq	(%rdi), %rax
   2fdce: ff 50 08                     	callq	*0x8(%rax)
   2fdd1: 48 8b 44 24 20               	movq	0x20(%rsp), %rax
   2fdd6: 48 3b 05 33 35 00 00         	cmpq	0x3533(%rip), %rax      # 0x33310 <strncmp+0x33310>
   2fddd: 74 39                        	je	0x2fe18 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xc08>
   2fddf: f0                           	lock
   2fde0: ff 48 08                     	decl	0x8(%rax)
   2fde3: 75 33                        	jne	0x2fe18 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xc08>
   2fde5: 48 8b 44 24 20               	movq	0x20(%rsp), %rax
   2fdea: 8b 40 0c                     	movl	0xc(%rax), %eax
   2fded: 83 f8 01                     	cmpl	$0x1, %eax
   2fdf0: 74 16                        	je	0x2fe08 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xbf8>
   2fdf2: 48 8b 7c 24 20               	movq	0x20(%rsp), %rdi
   2fdf7: 48 8b 07                     	movq	(%rdi), %rax
   2fdfa: ff 50 10                     	callq	*0x10(%rax)
   2fdfd: 48 8b 44 24 20               	movq	0x20(%rsp), %rax
   2fe02: f0                           	lock
   2fe03: ff 48 0c                     	decl	0xc(%rax)
   2fe06: 75 10                        	jne	0x2fe18 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xc08>
   2fe08: 48 8b 7c 24 20               	movq	0x20(%rsp), %rdi
   2fe0d: 48 85 ff                     	testq	%rdi, %rdi
   2fe10: 74 06                        	je	0x2fe18 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xc08>
   2fe12: 48 8b 07                     	movq	(%rdi), %rax
   2fe15: ff 50 08                     	callq	*0x8(%rax)
   2fe18: 48 8b 44 24 28               	movq	0x28(%rsp), %rax
   2fe1d: 48 3b 05 ec 34 00 00         	cmpq	0x34ec(%rip), %rax      # 0x33310 <strncmp+0x33310>
   2fe24: 74 39                        	je	0x2fe5f <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xc4f>
   2fe26: f0                           	lock
   2fe27: ff 48 08                     	decl	0x8(%rax)
   2fe2a: 75 33                        	jne	0x2fe5f <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xc4f>
   2fe2c: 48 8b 44 24 28               	movq	0x28(%rsp), %rax
   2fe31: 8b 40 0c                     	movl	0xc(%rax), %eax
   2fe34: 83 f8 01                     	cmpl	$0x1, %eax
   2fe37: 74 16                        	je	0x2fe4f <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xc3f>
   2fe39: 48 8b 7c 24 28               	movq	0x28(%rsp), %rdi
   2fe3e: 48 8b 07                     	movq	(%rdi), %rax
   2fe41: ff 50 10                     	callq	*0x10(%rax)
   2fe44: 48 8b 44 24 28               	movq	0x28(%rsp), %rax
   2fe49: f0                           	lock
   2fe4a: ff 48 0c                     	decl	0xc(%rax)
   2fe4d: 75 10                        	jne	0x2fe5f <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xc4f>
   2fe4f: 48 8b 7c 24 28               	movq	0x28(%rsp), %rdi
   2fe54: 48 85 ff                     	testq	%rdi, %rdi
   2fe57: 74 06                        	je	0x2fe5f <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xc4f>
   2fe59: 48 8b 07                     	movq	(%rdi), %rax
   2fe5c: ff 50 08                     	callq	*0x8(%rax)
   2fe5f: 0f b6 84 24 a8 00 00 00      	movzbl	0xa8(%rsp), %eax
   2fe67: c6 84 24 a8 00 00 00 00      	movb	$0x0, 0xa8(%rsp)
   2fe6f: 3c 01                        	cmpb	$0x1, %al
   2fe71: 75 15                        	jne	0x2fe88 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xc78>
   2fe73: 48 8b bc 24 98 00 00 00      	movq	0x98(%rsp), %rdi
   2fe7b: 48 8b 07                     	movq	(%rdi), %rax
   2fe7e: 8b b4 24 a0 00 00 00         	movl	0xa0(%rsp), %esi
   2fe85: ff 50 20                     	callq	*0x20(%rax)
   2fe88: 48 81 c4 b8 00 00 00         	addq	$0xb8, %rsp
   2fe8f: 5b                           	popq	%rbx
   2fe90: 41 5c                        	popq	%r12
   2fe92: 41 5d                        	popq	%r13
   2fe94: 41 5e                        	popq	%r14
   2fe96: 41 5f                        	popq	%r15
   2fe98: 5d                           	popq	%rbp
   2fe99: c3                           	retq
   2fe9a: 48 8b 47 20                  	movq	0x20(%rdi), %rax
   2fe9e: 48 85 c0                     	testq	%rax, %rax
   2fea1: 0f 84 39 0c 00 00            	je	0x30ae0 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x18d0>
   2fea7: 48 8b 38                     	movq	(%rax), %rdi
   2feaa: 48 85 ff                     	testq	%rdi, %rdi
   2fead: 0f 84 2d 0c 00 00            	je	0x30ae0 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x18d0>
   2feb3: 8b 47 7c                     	movl	0x7c(%rdi), %eax
   2feb6: a8 02                        	testb	$0x2, %al
   2feb8: 0f 84 75 0b 00 00            	je	0x30a33 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1823>
   2febe: 48 81 c7 b0 00 00 00         	addq	$0xb0, %rdi
   2fec5: 48 8d 35 c5 cf fd ff         	leaq	-0x2303b(%rip), %rsi    # 0xce91 <strncmp+0xce91>
   2fecc: ba 42 03 00 00               	movl	$0x342, %edx            # imm = 0x342
   2fed1: e8 9a 10 00 00               	callq	0x30f70 <_ZNK3c107SymBool10guard_boolEPKcl@plt>
   2fed6: 84 c0                        	testb	%al, %al
   2fed8: 0f 85 9d f5 ff ff            	jne	0x2f47b <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x26b>
   2fede: 48 8d 3d 76 ef fd ff         	leaq	-0x2108a(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   2fee5: 48 8d 35 2a e4 fd ff         	leaq	-0x21bd6(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   2feec: 48 8d 0d 4a d2 fd ff         	leaq	-0x22db6(%rip), %rcx    # 0xd13d <strncmp+0xd13d>
   2fef3: ba 21 00 00 00               	movl	$0x21, %edx
   2fef8: e8 33 10 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   2fefd: 48 8b 47 20                  	movq	0x20(%rdi), %rax
   2ff01: 48 85 c0                     	testq	%rax, %rax
   2ff04: 0f 84 d6 0b 00 00            	je	0x30ae0 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x18d0>
   2ff0a: 48 8b 38                     	movq	(%rax), %rdi
   2ff0d: 48 85 ff                     	testq	%rdi, %rdi
   2ff10: 0f 84 ca 0b 00 00            	je	0x30ae0 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x18d0>
   2ff16: 8b 47 7c                     	movl	0x7c(%rdi), %eax
   2ff19: a8 02                        	testb	$0x2, %al
   2ff1b: 0f 84 22 0b 00 00            	je	0x30a43 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1833>
   2ff21: 48 81 c7 b0 00 00 00         	addq	$0xb0, %rdi
   2ff28: 48 8d 35 62 cf fd ff         	leaq	-0x2309e(%rip), %rsi    # 0xce91 <strncmp+0xce91>
   2ff2f: ba 42 03 00 00               	movl	$0x342, %edx            # imm = 0x342
   2ff34: e8 37 10 00 00               	callq	0x30f70 <_ZNK3c107SymBool10guard_boolEPKcl@plt>
   2ff39: 84 c0                        	testb	%al, %al
   2ff3b: 0f 85 63 f5 ff ff            	jne	0x2f4a4 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x294>
   2ff41: 48 8d 3d 13 ef fd ff         	leaq	-0x210ed(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   2ff48: 48 8d 35 c7 e3 fd ff         	leaq	-0x21c39(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   2ff4f: 48 8d 0d 2d e8 fd ff         	leaq	-0x217d3(%rip), %rcx    # 0xe783 <strncmp+0xe783>
   2ff56: ba 22 00 00 00               	movl	$0x22, %edx
   2ff5b: e8 d0 0f 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   2ff60: 48 8b 47 20                  	movq	0x20(%rdi), %rax
   2ff64: 48 85 c0                     	testq	%rax, %rax
   2ff67: 0f 84 73 0b 00 00            	je	0x30ae0 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x18d0>
   2ff6d: 48 8b 38                     	movq	(%rax), %rdi
   2ff70: 48 85 ff                     	testq	%rdi, %rdi
   2ff73: 0f 84 67 0b 00 00            	je	0x30ae0 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x18d0>
   2ff79: 8b 47 7c                     	movl	0x7c(%rdi), %eax
   2ff7c: a8 02                        	testb	$0x2, %al
   2ff7e: 0f 84 cf 0a 00 00            	je	0x30a53 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1843>
   2ff84: 48 81 c7 b0 00 00 00         	addq	$0xb0, %rdi
   2ff8b: 48 8d 35 ff ce fd ff         	leaq	-0x23101(%rip), %rsi    # 0xce91 <strncmp+0xce91>
   2ff92: ba 42 03 00 00               	movl	$0x342, %edx            # imm = 0x342
   2ff97: e8 d4 0f 00 00               	callq	0x30f70 <_ZNK3c107SymBool10guard_boolEPKcl@plt>
   2ff9c: 84 c0                        	testb	%al, %al
   2ff9e: 0f 85 28 f5 ff ff            	jne	0x2f4cc <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x2bc>
   2ffa4: 48 8d 3d b0 ee fd ff         	leaq	-0x21150(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   2ffab: 48 8d 35 64 e3 fd ff         	leaq	-0x21c9c(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   2ffb2: 48 8d 0d 20 c6 fd ff         	leaq	-0x239e0(%rip), %rcx    # 0xc5d9 <strncmp+0xc5d9>
   2ffb9: ba 23 00 00 00               	movl	$0x23, %edx
   2ffbe: e8 6d 0f 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   2ffc3: 48 8b 47 20                  	movq	0x20(%rdi), %rax
   2ffc7: 48 85 c0                     	testq	%rax, %rax
   2ffca: 0f 84 10 0b 00 00            	je	0x30ae0 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x18d0>
   2ffd0: 48 8b 38                     	movq	(%rax), %rdi
   2ffd3: 48 85 ff                     	testq	%rdi, %rdi
   2ffd6: 0f 84 04 0b 00 00            	je	0x30ae0 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x18d0>
   2ffdc: 8b 47 7c                     	movl	0x7c(%rdi), %eax
   2ffdf: a8 02                        	testb	$0x2, %al
   2ffe1: 0f 84 7c 0a 00 00            	je	0x30a63 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1853>
   2ffe7: 48 81 c7 b0 00 00 00         	addq	$0xb0, %rdi
   2ffee: 48 8d 35 9c ce fd ff         	leaq	-0x23164(%rip), %rsi    # 0xce91 <strncmp+0xce91>
   2fff5: ba 42 03 00 00               	movl	$0x342, %edx            # imm = 0x342
   2fffa: e8 71 0f 00 00               	callq	0x30f70 <_ZNK3c107SymBool10guard_boolEPKcl@plt>
   2ffff: 84 c0                        	testb	%al, %al
   30001: 0f 85 ee f4 ff ff            	jne	0x2f4f5 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x2e5>
   30007: 48 8d 3d 4d ee fd ff         	leaq	-0x211b3(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   3000e: 48 8d 35 01 e3 fd ff         	leaq	-0x21cff(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   30015: 48 8d 0d 98 c9 fd ff         	leaq	-0x23668(%rip), %rcx    # 0xc9b4 <strncmp+0xc9b4>
   3001c: ba 24 00 00 00               	movl	$0x24, %edx
   30021: e8 0a 0f 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   30026: 48 8b 03                     	movq	(%rbx), %rax
   30029: 48 89 df                     	movq	%rbx, %rdi
   3002c: ff 50 10                     	callq	*0x10(%rax)
   3002f: 48 8d 43 0c                  	leaq	0xc(%rbx), %rax
   30033: f0                           	lock
   30034: ff 08                        	decl	(%rax)
   30036: 0f 84 89 fb ff ff            	je	0x2fbc5 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x9b5>
   3003c: e9 8d fb ff ff               	jmp	0x2fbce <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x9be>
   30041: 48 8b 03                     	movq	(%rbx), %rax
   30044: 48 89 df                     	movq	%rbx, %rdi
   30047: ff 50 10                     	callq	*0x10(%rax)
   3004a: 48 8d 43 0c                  	leaq	0xc(%rbx), %rax
   3004e: f0                           	lock
   3004f: ff 08                        	decl	(%rax)
   30051: 0f 84 27 fc ff ff            	je	0x2fc7e <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xa6e>
   30057: e9 2b fc ff ff               	jmp	0x2fc87 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xa77>
   3005c: 89 c7                        	movl	%eax, %edi
   3005e: e8 ad 0f 00 00               	callq	0x31010 <_ZN6caffe28TypeMeta26error_unsupported_typemetaES0_@plt>
   30063: 48 8b 07                     	movq	(%rdi), %rax
   30066: ff 50 68                     	callq	*0x68(%rax)
   30069: 3c 01                        	cmpb	$0x1, %al
   3006b: 0f 84 ed f1 ff ff            	je	0x2f25e <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x4e>
   30071: 48 8d 3d e3 ed fd ff         	leaq	-0x2121d(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   30078: 48 8d 35 97 e2 fd ff         	leaq	-0x21d69(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   3007f: 48 8d 0d f3 cd fd ff         	leaq	-0x2320d(%rip), %rcx    # 0xce79 <strncmp+0xce79>
   30086: ba 13 00 00 00               	movl	$0x13, %edx
   3008b: e8 a0 0e 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   30090: 48 8b 07                     	movq	(%rdi), %rax
   30093: ff 50 68                     	callq	*0x68(%rax)
   30096: 3c 01                        	cmpb	$0x1, %al
   30098: 0f 84 ec f1 ff ff            	je	0x2f28a <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x7a>
   3009e: 48 8d 3d b6 ed fd ff         	leaq	-0x2124a(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   300a5: 48 8d 35 6a e2 fd ff         	leaq	-0x21d96(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   300ac: 48 8d 0d db d1 fd ff         	leaq	-0x22e25(%rip), %rcx    # 0xd28e <strncmp+0xd28e>
   300b3: ba 14 00 00 00               	movl	$0x14, %edx
   300b8: e8 73 0e 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   300bd: 48 8b 07                     	movq	(%rdi), %rax
   300c0: ff 50 68                     	callq	*0x68(%rax)
   300c3: 3c 01                        	cmpb	$0x1, %al
   300c5: 0f 84 ea f1 ff ff            	je	0x2f2b5 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xa5>
   300cb: 48 8d 3d 89 ed fd ff         	leaq	-0x21277(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   300d2: 48 8d 35 3d e2 fd ff         	leaq	-0x21dc3(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   300d9: 48 8d 0d c0 e7 fd ff         	leaq	-0x21840(%rip), %rcx    # 0xe8a0 <strncmp+0xe8a0>
   300e0: ba 15 00 00 00               	movl	$0x15, %edx
   300e5: e8 46 0e 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   300ea: 48 8b 07                     	movq	(%rdi), %rax
   300ed: ff 50 68                     	callq	*0x68(%rax)
   300f0: 3c 01                        	cmpb	$0x1, %al
   300f2: 0f 84 e9 f1 ff ff            	je	0x2f2e1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xd1>
   300f8: 48 8d 3d 5c ed fd ff         	leaq	-0x212a4(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   300ff: 48 8d 35 10 e2 fd ff         	leaq	-0x21df0(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   30106: 48 8d 0d 3a dc fd ff         	leaq	-0x223c6(%rip), %rcx    # 0xdd47 <strncmp+0xdd47>
   3010d: ba 16 00 00 00               	movl	$0x16, %edx
   30112: e8 19 0e 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   30117: 48 8b 07                     	movq	(%rdi), %rax
   3011a: ff 50 68                     	callq	*0x68(%rax)
   3011d: 89 c3                        	movl	%eax, %ebx
   3011f: 49 8b 3c 24                  	movq	(%r12), %rdi
   30123: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   3012b: 0f 89 e7 f1 ff ff            	jns	0x2f318 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x108>
   30131: 48 8b 07                     	movq	(%rdi), %rax
   30134: ff 50 68                     	callq	*0x68(%rax)
   30137: 66 39 c3                     	cmpw	%ax, %bx
   3013a: 0f 84 f5 f1 ff ff            	je	0x2f335 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x125>
   30140: eb 53                        	jmp	0x30195 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xf85>
   30142: 48 8b 07                     	movq	(%rdi), %rax
   30145: ff 50 68                     	callq	*0x68(%rax)
   30148: 89 c3                        	movl	%eax, %ebx
   3014a: 49 8b 3f                     	movq	(%r15), %rdi
   3014d: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   30155: 0f 89 10 f2 ff ff            	jns	0x2f36b <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x15b>
   3015b: 48 8b 07                     	movq	(%rdi), %rax
   3015e: ff 50 68                     	callq	*0x68(%rax)
   30161: 66 39 c3                     	cmpw	%ax, %bx
   30164: 0f 84 1e f2 ff ff            	je	0x2f388 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x178>
   3016a: eb 29                        	jmp	0x30195 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xf85>
   3016c: 48 8b 07                     	movq	(%rdi), %rax
   3016f: ff 50 68                     	callq	*0x68(%rax)
   30172: 89 c3                        	movl	%eax, %ebx
   30174: 48 8b 7d 00                  	movq	(%rbp), %rdi
   30178: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   30180: 0f 89 39 f2 ff ff            	jns	0x2f3bf <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1af>
   30186: 48 8b 07                     	movq	(%rdi), %rax
   30189: ff 50 68                     	callq	*0x68(%rax)
   3018c: 66 39 c3                     	cmpw	%ax, %bx
   3018f: 0f 84 47 f2 ff ff            	je	0x2f3dc <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1cc>
   30195: 48 8d 3d bf ec fd ff         	leaq	-0x21341(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   3019c: 48 8d 35 73 e1 fd ff         	leaq	-0x21e8d(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   301a3: 48 8d 0d f5 d0 fd ff         	leaq	-0x22f0b(%rip), %rcx    # 0xd29f <strncmp+0xd29f>
   301aa: ba 1a 00 00 00               	movl	$0x1a, %edx
   301af: e8 7c 0d 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   301b4: 31 f6                        	xorl	%esi, %esi
   301b6: e8 65 0e 00 00               	callq	0x31020 <_ZNK3c1010TensorImpl20is_contiguous_customENS_12MemoryFormatE@plt>
   301bb: 84 c0                        	testb	%al, %al
   301bd: 0f 85 b8 f2 ff ff            	jne	0x2f47b <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x26b>
   301c3: e9 16 fd ff ff               	jmp	0x2fede <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xcce>
   301c8: 31 f6                        	xorl	%esi, %esi
   301ca: e8 51 0e 00 00               	callq	0x31020 <_ZNK3c1010TensorImpl20is_contiguous_customENS_12MemoryFormatE@plt>
   301cf: 84 c0                        	testb	%al, %al
   301d1: 0f 85 cd f2 ff ff            	jne	0x2f4a4 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x294>
   301d7: e9 65 fd ff ff               	jmp	0x2ff41 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xd31>
   301dc: 31 f6                        	xorl	%esi, %esi
   301de: e8 3d 0e 00 00               	callq	0x31020 <_ZNK3c1010TensorImpl20is_contiguous_customENS_12MemoryFormatE@plt>
   301e3: 84 c0                        	testb	%al, %al
   301e5: 0f 85 e1 f2 ff ff            	jne	0x2f4cc <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x2bc>
   301eb: e9 b4 fd ff ff               	jmp	0x2ffa4 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xd94>
   301f0: 31 f6                        	xorl	%esi, %esi
   301f2: e8 29 0e 00 00               	callq	0x31020 <_ZNK3c1010TensorImpl20is_contiguous_customENS_12MemoryFormatE@plt>
   301f7: 84 c0                        	testb	%al, %al
   301f9: 0f 85 f6 f2 ff ff            	jne	0x2f4f5 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x2e5>
   301ff: e9 03 fe ff ff               	jmp	0x30007 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xdf7>
   30204: 48 8b 07                     	movq	(%rdi), %rax
   30207: ff 50 60                     	callq	*0x60(%rax)
   3020a: 48 83 f8 04                  	cmpq	$0x4, %rax
   3020e: 0f 84 ff f2 ff ff            	je	0x2f513 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x303>
   30214: eb 34                        	jmp	0x3024a <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x103a>
   30216: 48 8b 07                     	movq	(%rdi), %rax
   30219: ff 50 60                     	callq	*0x60(%rax)
   3021c: 48 83 f8 04                  	cmpq	$0x4, %rax
   30220: 0f 84 0c f3 ff ff            	je	0x2f532 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x322>
   30226: eb 22                        	jmp	0x3024a <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x103a>
   30228: 48 8b 07                     	movq	(%rdi), %rax
   3022b: ff 50 60                     	callq	*0x60(%rax)
   3022e: 48 83 f8 04                  	cmpq	$0x4, %rax
   30232: 0f 84 18 f3 ff ff            	je	0x2f550 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x340>
   30238: eb 10                        	jmp	0x3024a <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x103a>
   3023a: 48 8b 07                     	movq	(%rdi), %rax
   3023d: ff 50 60                     	callq	*0x60(%rax)
   30240: 48 83 f8 04                  	cmpq	$0x4, %rax
   30244: 0f 84 25 f3 ff ff            	je	0x2f56f <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x35f>
   3024a: 48 8d 3d 0a ec fd ff         	leaq	-0x213f6(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   30251: 48 8d 35 be e0 fd ff         	leaq	-0x21f42(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   30258: 48 8d 0d 82 d3 fd ff         	leaq	-0x22c7e(%rip), %rcx    # 0xd5e1 <strncmp+0xd5e1>
   3025f: ba 28 00 00 00               	movl	$0x28, %edx
   30264: e8 c7 0c 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   30269: 48 8b 03                     	movq	(%rbx), %rax
   3026c: 48 89 df                     	movq	%rbx, %rdi
   3026f: 31 f6                        	xorl	%esi, %esi
   30271: ff 50 30                     	callq	*0x30(%rax)
   30274: 49 89 c5                     	movq	%rax, %r13
   30277: 49 8b 1c 24                  	movq	(%r12), %rbx
   3027b: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   30282: 0f 84 29 f3 ff ff            	je	0x2f5b1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x3a1>
   30288: 48 8b 03                     	movq	(%rbx), %rax
   3028b: 48 89 df                     	movq	%rbx, %rdi
   3028e: 31 f6                        	xorl	%esi, %esi
   30290: ff 50 30                     	callq	*0x30(%rax)
   30293: 49 39 c5                     	cmpq	%rax, %r13
   30296: 0f 84 3f f3 ff ff            	je	0x2f5db <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x3cb>
   3029c: e9 69 03 00 00               	jmp	0x3060a <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x13fa>
   302a1: 48 8b 03                     	movq	(%rbx), %rax
   302a4: 48 89 df                     	movq	%rbx, %rdi
   302a7: 31 f6                        	xorl	%esi, %esi
   302a9: ff 50 30                     	callq	*0x30(%rax)
   302ac: 49 89 c5                     	movq	%rax, %r13
   302af: 49 8b 1f                     	movq	(%r15), %rbx
   302b2: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   302b9: 0f 84 5d f3 ff ff            	je	0x2f61c <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x40c>
   302bf: 48 8b 03                     	movq	(%rbx), %rax
   302c2: 48 89 df                     	movq	%rbx, %rdi
   302c5: 31 f6                        	xorl	%esi, %esi
   302c7: ff 50 30                     	callq	*0x30(%rax)
   302ca: 49 39 c5                     	cmpq	%rax, %r13
   302cd: 0f 84 73 f3 ff ff            	je	0x2f646 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x436>
   302d3: e9 32 03 00 00               	jmp	0x3060a <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x13fa>
   302d8: 48 8b 03                     	movq	(%rbx), %rax
   302db: be 02 00 00 00               	movl	$0x2, %esi
   302e0: 48 89 df                     	movq	%rbx, %rdi
   302e3: ff 50 30                     	callq	*0x30(%rax)
   302e6: 49 89 c5                     	movq	%rax, %r13
   302e9: 49 8b 1c 24                  	movq	(%r12), %rbx
   302ed: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   302f4: 0f 84 92 f3 ff ff            	je	0x2f68c <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x47c>
   302fa: 48 8b 03                     	movq	(%rbx), %rax
   302fd: be 02 00 00 00               	movl	$0x2, %esi
   30302: 48 89 df                     	movq	%rbx, %rdi
   30305: ff 50 30                     	callq	*0x30(%rax)
   30308: 49 39 c5                     	cmpq	%rax, %r13
   3030b: 0f 84 a9 f3 ff ff            	je	0x2f6ba <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x4aa>
   30311: e9 e6 03 00 00               	jmp	0x306fc <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x14ec>
   30316: 48 8b 03                     	movq	(%rbx), %rax
   30319: be 02 00 00 00               	movl	$0x2, %esi
   3031e: 48 89 df                     	movq	%rbx, %rdi
   30321: ff 50 30                     	callq	*0x30(%rax)
   30324: 49 89 c5                     	movq	%rax, %r13
   30327: 49 8b 1f                     	movq	(%r15), %rbx
   3032a: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   30331: 0f 84 c8 f3 ff ff            	je	0x2f6ff <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x4ef>
   30337: 48 8b 03                     	movq	(%rbx), %rax
   3033a: be 02 00 00 00               	movl	$0x2, %esi
   3033f: 48 89 df                     	movq	%rbx, %rdi
   30342: ff 50 30                     	callq	*0x30(%rax)
   30345: 49 39 c5                     	cmpq	%rax, %r13
   30348: 0f 84 df f3 ff ff            	je	0x2f72d <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x51d>
   3034e: e9 a9 03 00 00               	jmp	0x306fc <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x14ec>
   30353: 48 8b 03                     	movq	(%rbx), %rax
   30356: be 03 00 00 00               	movl	$0x3, %esi
   3035b: 48 89 df                     	movq	%rbx, %rdi
   3035e: ff 50 30                     	callq	*0x30(%rax)
   30361: 49 89 c5                     	movq	%rax, %r13
   30364: 49 8b 1c 24                  	movq	(%r12), %rbx
   30368: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   3036f: 0f 84 fe f3 ff ff            	je	0x2f773 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x563>
   30375: 48 8b 03                     	movq	(%rbx), %rax
   30378: be 03 00 00 00               	movl	$0x3, %esi
   3037d: 48 89 df                     	movq	%rbx, %rdi
   30380: ff 50 30                     	callq	*0x30(%rax)
   30383: 49 39 c5                     	cmpq	%rax, %r13
   30386: 0f 84 15 f4 ff ff            	je	0x2f7a1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x591>
   3038c: e9 5d 04 00 00               	jmp	0x307ee <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x15de>
   30391: 48 8b 03                     	movq	(%rbx), %rax
   30394: be 03 00 00 00               	movl	$0x3, %esi
   30399: 48 89 df                     	movq	%rbx, %rdi
   3039c: ff 50 30                     	callq	*0x30(%rax)
   3039f: 49 89 c5                     	movq	%rax, %r13
   303a2: 49 8b 1f                     	movq	(%r15), %rbx
   303a5: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   303ac: 0f 84 34 f4 ff ff            	je	0x2f7e6 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x5d6>
   303b2: 48 8b 03                     	movq	(%rbx), %rax
   303b5: be 03 00 00 00               	movl	$0x3, %esi
   303ba: 48 89 df                     	movq	%rbx, %rdi
   303bd: ff 50 30                     	callq	*0x30(%rax)
   303c0: 49 39 c5                     	cmpq	%rax, %r13
   303c3: 0f 84 4b f4 ff ff            	je	0x2f814 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x604>
   303c9: e9 20 04 00 00               	jmp	0x307ee <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x15de>
   303ce: 48 8b 03                     	movq	(%rbx), %rax
   303d1: be 01 00 00 00               	movl	$0x1, %esi
   303d6: 48 89 df                     	movq	%rbx, %rdi
   303d9: ff 50 30                     	callq	*0x30(%rax)
   303dc: 48 83 f8 08                  	cmpq	$0x8, %rax
   303e0: 0f 84 6d f4 ff ff            	je	0x2f853 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x643>
   303e6: 48 8d 3d 6e ea fd ff         	leaq	-0x21592(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   303ed: 48 8d 35 22 df fd ff         	leaq	-0x220de(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   303f4: 48 8d 0d 56 e8 fd ff         	leaq	-0x217aa(%rip), %rcx    # 0xec51 <strncmp+0xec51>
   303fb: ba 2c 00 00 00               	movl	$0x2c, %edx
   30400: e8 2b 0b 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   30405: 48 8b 03                     	movq	(%rbx), %rax
   30408: be 01 00 00 00               	movl	$0x1, %esi
   3040d: 48 89 df                     	movq	%rbx, %rdi
   30410: ff 50 30                     	callq	*0x30(%rax)
   30413: 48 83 f8 01                  	cmpq	$0x1, %rax
   30417: 0f 85 72 f4 ff ff            	jne	0x2f88f <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x67f>
   3041d: e9 ad f4 ff ff               	jmp	0x2f8cf <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x6bf>
   30422: 48 8b 03                     	movq	(%rbx), %rax
   30425: be 01 00 00 00               	movl	$0x1, %esi
   3042a: 48 89 df                     	movq	%rbx, %rdi
   3042d: ff 50 30                     	callq	*0x30(%rax)
   30430: 49 89 c5                     	movq	%rax, %r13
   30433: 49 8b 1c 24                  	movq	(%r12), %rbx
   30437: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   3043e: 0f 84 d1 f4 ff ff            	je	0x2f915 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x705>
   30444: 48 8b 03                     	movq	(%rbx), %rax
   30447: be 01 00 00 00               	movl	$0x1, %esi
   3044c: 48 89 df                     	movq	%rbx, %rdi
   3044f: ff 50 30                     	callq	*0x30(%rax)
   30452: 49 39 c5                     	cmpq	%rax, %r13
   30455: 0f 84 e8 f4 ff ff            	je	0x2f943 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x733>
   3045b: 48 8d 3d f9 e9 fd ff         	leaq	-0x21607(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   30462: 48 8d 35 ad de fd ff         	leaq	-0x22153(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   30469: 48 8d 0d ff c7 fd ff         	leaq	-0x23801(%rip), %rcx    # 0xcc6f <strncmp+0xcc6f>
   30470: ba 2e 00 00 00               	movl	$0x2e, %edx
   30475: e8 b6 0a 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   3047a: 48 8b 03                     	movq	(%rbx), %rax
   3047d: be 03 00 00 00               	movl	$0x3, %esi
   30482: 48 89 df                     	movq	%rbx, %rdi
   30485: ff 50 30                     	callq	*0x30(%rax)
   30488: 48 3d 80 00 00 00            	cmpq	$0x80, %rax
   3048e: 0f 84 f0 f4 ff ff            	je	0x2f984 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x774>
   30494: 48 8d 3d c0 e9 fd ff         	leaq	-0x21640(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   3049b: 48 8d 35 74 de fd ff         	leaq	-0x2218c(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   304a2: 48 8d 0d 16 c6 fd ff         	leaq	-0x239ea(%rip), %rcx    # 0xcabf <strncmp+0xcabf>
   304a9: ba 2f 00 00 00               	movl	$0x2f, %edx
   304ae: e8 7d 0a 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   304b3: 48 8b 07                     	movq	(%rdi), %rax
   304b6: ff 50 40                     	callq	*0x40(%rax)
   304b9: 48 89 c7                     	movq	%rax, %rdi
   304bc: e9 e6 f4 ff ff               	jmp	0x2f9a7 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x797>
   304c1: 48 8b 06                     	movq	(%rsi), %rax
   304c4: 49 89 fd                     	movq	%rdi, %r13
   304c7: 48 89 f7                     	movq	%rsi, %rdi
   304ca: 48 89 d3                     	movq	%rdx, %rbx
   304cd: ff 50 40                     	callq	*0x40(%rax)
   304d0: 4c 89 ef                     	movq	%r13, %rdi
   304d3: 48 89 c6                     	movq	%rax, %rsi
   304d6: 48 89 d0                     	movq	%rdx, %rax
   304d9: 48 89 da                     	movq	%rbx, %rdx
   304dc: e9 e8 f4 ff ff               	jmp	0x2f9c9 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x7b9>
   304e1: 48 8b 07                     	movq	(%rdi), %rax
   304e4: ff 50 68                     	callq	*0x68(%rax)
   304e7: 89 c6                        	movl	%eax, %esi
   304e9: e9 1f f5 ff ff               	jmp	0x2fa0d <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x7fd>
   304ee: 49 8b 45 00                  	movq	(%r13), %rax
   304f2: 4c 89 ef                     	movq	%r13, %rdi
   304f5: 31 f6                        	xorl	%esi, %esi
   304f7: ff 50 30                     	callq	*0x30(%rax)
   304fa: 49 89 c5                     	movq	%rax, %r13
   304fd: 49 8b 2e                     	movq	(%r14), %rbp
   30500: f6 85 ae 00 00 00 08         	testb	$0x8, 0xae(%rbp)
   30507: 0f 84 60 f5 ff ff            	je	0x2fa6d <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x85d>
   3050d: 48 8b 45 00                  	movq	(%rbp), %rax
   30511: be 02 00 00 00               	movl	$0x2, %esi
   30516: 48 89 ef                     	movq	%rbp, %rdi
   30519: ff 50 30                     	callq	*0x30(%rax)
   3051c: 48 89 c5                     	movq	%rax, %rbp
   3051f: 49 8b 1c 24                  	movq	(%r12), %rbx
   30523: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   3052a: 0f 84 74 f5 ff ff            	je	0x2faa4 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x894>
   30530: 48 8b 03                     	movq	(%rbx), %rax
   30533: be 01 00 00 00               	movl	$0x1, %esi
   30538: 48 89 df                     	movq	%rbx, %rdi
   3053b: ff 50 30                     	callq	*0x30(%rax)
   3053e: e9 86 f5 ff ff               	jmp	0x2fac9 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x8b9>
   30543: 31 ff                        	xorl	%edi, %edi
   30545: 31 d2                        	xorl	%edx, %edx
   30547: e8 f4 09 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   3054c: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   30550: 48 83 c3 48                  	addq	$0x48, %rbx
   30554: 48 83 fe 06                  	cmpq	$0x6, %rsi
   30558: 0f 82 3e f0 ff ff            	jb	0x2f59c <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x38c>
   3055e: 48 8b 1b                     	movq	(%rbx), %rbx
   30561: 4c 8b 2c c3                  	movq	(%rbx,%rax,8), %r13
   30565: 49 8b 1c 24                  	movq	(%r12), %rbx
   30569: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   30570: 0f 84 3b f0 ff ff            	je	0x2f5b1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x3a1>
   30576: e9 0d fd ff ff               	jmp	0x30288 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1078>
   3057b: 31 ff                        	xorl	%edi, %edi
   3057d: 31 d2                        	xorl	%edx, %edx
   3057f: e8 bc 09 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   30584: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   30588: 48 83 c3 48                  	addq	$0x48, %rbx
   3058c: 48 83 fe 06                  	cmpq	$0x6, %rsi
   30590: 0f 82 38 f0 ff ff            	jb	0x2f5ce <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x3be>
   30596: 48 8b 1b                     	movq	(%rbx), %rbx
   30599: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   3059d: 49 39 c5                     	cmpq	%rax, %r13
   305a0: 0f 84 35 f0 ff ff            	je	0x2f5db <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x3cb>
   305a6: eb 62                        	jmp	0x3060a <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x13fa>
   305a8: 31 ff                        	xorl	%edi, %edi
   305aa: 31 d2                        	xorl	%edx, %edx
   305ac: e8 8f 09 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   305b1: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   305b5: 48 83 c3 48                  	addq	$0x48, %rbx
   305b9: 48 83 fe 06                  	cmpq	$0x6, %rsi
   305bd: 0f 82 45 f0 ff ff            	jb	0x2f608 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x3f8>
   305c3: 48 8b 1b                     	movq	(%rbx), %rbx
   305c6: 4c 8b 2c c3                  	movq	(%rbx,%rax,8), %r13
   305ca: 49 8b 1f                     	movq	(%r15), %rbx
   305cd: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   305d4: 0f 84 42 f0 ff ff            	je	0x2f61c <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x40c>
   305da: e9 e0 fc ff ff               	jmp	0x302bf <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x10af>
   305df: 31 ff                        	xorl	%edi, %edi
   305e1: 31 d2                        	xorl	%edx, %edx
   305e3: e8 58 09 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   305e8: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   305ec: 48 83 c3 48                  	addq	$0x48, %rbx
   305f0: 48 83 fe 06                  	cmpq	$0x6, %rsi
   305f4: 0f 82 3f f0 ff ff            	jb	0x2f639 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x429>
   305fa: 48 8b 1b                     	movq	(%rbx), %rbx
   305fd: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   30601: 49 39 c5                     	cmpq	%rax, %r13
   30604: 0f 84 3c f0 ff ff            	je	0x2f646 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x436>
   3060a: 48 8d 3d 4a e8 fd ff         	leaq	-0x217b6(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   30611: 48 8d 35 fe dc fd ff         	leaq	-0x22302(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   30618: 48 8d 0d 34 c6 fd ff         	leaq	-0x239cc(%rip), %rcx    # 0xcc53 <strncmp+0xcc53>
   3061f: ba 29 00 00 00               	movl	$0x29, %edx
   30624: e8 07 09 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   30629: bf 02 00 00 00               	movl	$0x2, %edi
   3062e: 31 d2                        	xorl	%edx, %edx
   30630: e8 0b 09 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   30635: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   30639: 48 83 c3 48                  	addq	$0x48, %rbx
   3063d: 48 83 fe 06                  	cmpq	$0x6, %rsi
   30641: 0f 82 30 f0 ff ff            	jb	0x2f677 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x467>
   30647: 48 8b 1b                     	movq	(%rbx), %rbx
   3064a: 4c 8b 2c c3                  	movq	(%rbx,%rax,8), %r13
   3064e: 49 8b 1c 24                  	movq	(%r12), %rbx
   30652: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   30659: 0f 84 2d f0 ff ff            	je	0x2f68c <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x47c>
   3065f: e9 96 fc ff ff               	jmp	0x302fa <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x10ea>
   30664: bf 02 00 00 00               	movl	$0x2, %edi
   30669: 31 d2                        	xorl	%edx, %edx
   3066b: e8 d0 08 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   30670: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   30674: 48 83 c3 48                  	addq	$0x48, %rbx
   30678: 48 83 fe 06                  	cmpq	$0x6, %rsi
   3067c: 0f 82 2b f0 ff ff            	jb	0x2f6ad <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x49d>
   30682: 48 8b 1b                     	movq	(%rbx), %rbx
   30685: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   30689: 49 39 c5                     	cmpq	%rax, %r13
   3068c: 0f 84 28 f0 ff ff            	je	0x2f6ba <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x4aa>
   30692: eb 68                        	jmp	0x306fc <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x14ec>
   30694: bf 02 00 00 00               	movl	$0x2, %edi
   30699: 31 d2                        	xorl	%edx, %edx
   3069b: e8 a0 08 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   306a0: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   306a4: 48 83 c3 48                  	addq	$0x48, %rbx
   306a8: 48 83 fe 06                  	cmpq	$0x6, %rsi
   306ac: 0f 82 39 f0 ff ff            	jb	0x2f6eb <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x4db>
   306b2: 48 8b 1b                     	movq	(%rbx), %rbx
   306b5: 4c 8b 2c c3                  	movq	(%rbx,%rax,8), %r13
   306b9: 49 8b 1f                     	movq	(%r15), %rbx
   306bc: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   306c3: 0f 84 36 f0 ff ff            	je	0x2f6ff <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x4ef>
   306c9: e9 69 fc ff ff               	jmp	0x30337 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1127>
   306ce: bf 02 00 00 00               	movl	$0x2, %edi
   306d3: 31 d2                        	xorl	%edx, %edx
   306d5: e8 66 08 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   306da: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   306de: 48 83 c3 48                  	addq	$0x48, %rbx
   306e2: 48 83 fe 06                  	cmpq	$0x6, %rsi
   306e6: 0f 82 34 f0 ff ff            	jb	0x2f720 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x510>
   306ec: 48 8b 1b                     	movq	(%rbx), %rbx
   306ef: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   306f3: 49 39 c5                     	cmpq	%rax, %r13
   306f6: 0f 84 31 f0 ff ff            	je	0x2f72d <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x51d>
   306fc: 48 8d 3d 58 e7 fd ff         	leaq	-0x218a8(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   30703: 48 8d 35 0c dc fd ff         	leaq	-0x223f4(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   3070a: 48 8d 0d e6 d9 fd ff         	leaq	-0x2261a(%rip), %rcx    # 0xe0f7 <strncmp+0xe0f7>
   30711: ba 2a 00 00 00               	movl	$0x2a, %edx
   30716: e8 15 08 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   3071b: bf 03 00 00 00               	movl	$0x3, %edi
   30720: 31 d2                        	xorl	%edx, %edx
   30722: e8 19 08 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   30727: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   3072b: 48 83 c3 48                  	addq	$0x48, %rbx
   3072f: 48 83 fe 06                  	cmpq	$0x6, %rsi
   30733: 0f 82 25 f0 ff ff            	jb	0x2f75e <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x54e>
   30739: 48 8b 1b                     	movq	(%rbx), %rbx
   3073c: 4c 8b 2c c3                  	movq	(%rbx,%rax,8), %r13
   30740: 49 8b 1c 24                  	movq	(%r12), %rbx
   30744: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   3074b: 0f 84 22 f0 ff ff            	je	0x2f773 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x563>
   30751: e9 1f fc ff ff               	jmp	0x30375 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1165>
   30756: bf 03 00 00 00               	movl	$0x3, %edi
   3075b: 31 d2                        	xorl	%edx, %edx
   3075d: e8 de 07 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   30762: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   30766: 48 83 c3 48                  	addq	$0x48, %rbx
   3076a: 48 83 fe 06                  	cmpq	$0x6, %rsi
   3076e: 0f 82 20 f0 ff ff            	jb	0x2f794 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x584>
   30774: 48 8b 1b                     	movq	(%rbx), %rbx
   30777: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   3077b: 49 39 c5                     	cmpq	%rax, %r13
   3077e: 0f 84 1d f0 ff ff            	je	0x2f7a1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x591>
   30784: eb 68                        	jmp	0x307ee <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x15de>
   30786: bf 03 00 00 00               	movl	$0x3, %edi
   3078b: 31 d2                        	xorl	%edx, %edx
   3078d: e8 ae 07 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   30792: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   30796: 48 83 c3 48                  	addq	$0x48, %rbx
   3079a: 48 83 fe 06                  	cmpq	$0x6, %rsi
   3079e: 0f 82 2e f0 ff ff            	jb	0x2f7d2 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x5c2>
   307a4: 48 8b 1b                     	movq	(%rbx), %rbx
   307a7: 4c 8b 2c c3                  	movq	(%rbx,%rax,8), %r13
   307ab: 49 8b 1f                     	movq	(%r15), %rbx
   307ae: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   307b5: 0f 84 2b f0 ff ff            	je	0x2f7e6 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x5d6>
   307bb: e9 f2 fb ff ff               	jmp	0x303b2 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x11a2>
   307c0: bf 03 00 00 00               	movl	$0x3, %edi
   307c5: 31 d2                        	xorl	%edx, %edx
   307c7: e8 74 07 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   307cc: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   307d0: 48 83 c3 48                  	addq	$0x48, %rbx
   307d4: 48 83 fe 06                  	cmpq	$0x6, %rsi
   307d8: 0f 82 29 f0 ff ff            	jb	0x2f807 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x5f7>
   307de: 48 8b 1b                     	movq	(%rbx), %rbx
   307e1: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   307e5: 49 39 c5                     	cmpq	%rax, %r13
   307e8: 0f 84 26 f0 ff ff            	je	0x2f814 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x604>
   307ee: 48 8d 3d 66 e6 fd ff         	leaq	-0x2199a(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   307f5: 48 8d 35 1a db fd ff         	leaq	-0x224e6(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   307fc: 48 8d 0d c1 ca fd ff         	leaq	-0x2353f(%rip), %rcx    # 0xd2c4 <strncmp+0xd2c4>
   30803: ba 2b 00 00 00               	movl	$0x2b, %edx
   30808: e8 23 07 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   3080d: bf 01 00 00 00               	movl	$0x1, %edi
   30812: 31 d2                        	xorl	%edx, %edx
   30814: e8 27 07 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   30819: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   3081d: 48 83 c3 48                  	addq	$0x48, %rbx
   30821: 48 83 fe 06                  	cmpq	$0x6, %rsi
   30825: 0f 82 1a f0 ff ff            	jb	0x2f845 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x635>
   3082b: 48 8b 1b                     	movq	(%rbx), %rbx
   3082e: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   30832: 48 83 f8 08                  	cmpq	$0x8, %rax
   30836: 0f 84 17 f0 ff ff            	je	0x2f853 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x643>
   3083c: e9 a5 fb ff ff               	jmp	0x303e6 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x11d6>
   30841: bf 01 00 00 00               	movl	$0x1, %edi
   30846: 31 d2                        	xorl	%edx, %edx
   30848: e8 f3 06 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   3084d: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   30851: 48 83 c3 48                  	addq	$0x48, %rbx
   30855: 48 83 fe 06                  	cmpq	$0x6, %rsi
   30859: 0f 82 26 f0 ff ff            	jb	0x2f885 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x675>
   3085f: 48 8b 1b                     	movq	(%rbx), %rbx
   30862: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   30866: 48 83 f8 01                  	cmpq	$0x1, %rax
   3086a: 0f 85 1f f0 ff ff            	jne	0x2f88f <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x67f>
   30870: e9 5a f0 ff ff               	jmp	0x2f8cf <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x6bf>
   30875: bf 01 00 00 00               	movl	$0x1, %edi
   3087a: 31 d2                        	xorl	%edx, %edx
   3087c: e8 bf 06 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   30881: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   30885: 48 83 c3 48                  	addq	$0x48, %rbx
   30889: 48 83 fe 06                  	cmpq	$0x6, %rsi
   3088d: 0f 82 6d f0 ff ff            	jb	0x2f900 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x6f0>
   30893: 48 8b 1b                     	movq	(%rbx), %rbx
   30896: 4c 8b 2c c3                  	movq	(%rbx,%rax,8), %r13
   3089a: 49 8b 1c 24                  	movq	(%r12), %rbx
   3089e: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   308a5: 0f 84 6a f0 ff ff            	je	0x2f915 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x705>
   308ab: e9 94 fb ff ff               	jmp	0x30444 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1234>
   308b0: bf 01 00 00 00               	movl	$0x1, %edi
   308b5: 31 d2                        	xorl	%edx, %edx
   308b7: e8 84 06 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   308bc: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   308c0: 48 83 c3 48                  	addq	$0x48, %rbx
   308c4: 48 83 fe 06                  	cmpq	$0x6, %rsi
   308c8: 0f 82 68 f0 ff ff            	jb	0x2f936 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x726>
   308ce: 48 8b 1b                     	movq	(%rbx), %rbx
   308d1: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   308d5: 49 39 c5                     	cmpq	%rax, %r13
   308d8: 0f 84 65 f0 ff ff            	je	0x2f943 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x733>
   308de: e9 78 fb ff ff               	jmp	0x3045b <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x124b>
   308e3: bf 03 00 00 00               	movl	$0x3, %edi
   308e8: 31 d2                        	xorl	%edx, %edx
   308ea: e8 51 06 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   308ef: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   308f3: 48 83 c3 48                  	addq	$0x48, %rbx
   308f7: 48 83 fe 06                  	cmpq	$0x6, %rsi
   308fb: 0f 82 73 f0 ff ff            	jb	0x2f974 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x764>
   30901: 48 8b 1b                     	movq	(%rbx), %rbx
   30904: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   30908: 48 3d 80 00 00 00            	cmpq	$0x80, %rax
   3090e: 0f 84 70 f0 ff ff            	je	0x2f984 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x774>
   30914: e9 7b fb ff ff               	jmp	0x30494 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1284>
   30919: 48 8b 3f                     	movq	(%rdi), %rdi
   3091c: 48 85 ff                     	testq	%rdi, %rdi
   3091f: 0f 85 82 f0 ff ff            	jne	0x2f9a7 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x797>
   30925: 48 8d 3d aa ca fd ff         	leaq	-0x23556(%rip), %rdi    # 0xd3d6 <strncmp+0xd3d6>
   3092c: 48 8d 35 0e b2 fd ff         	leaq	-0x24df2(%rip), %rsi    # 0xbb41 <strncmp+0xbb41>
   30933: 48 8d 0d 84 d8 fd ff         	leaq	-0x2277c(%rip), %rcx    # 0xe1be <strncmp+0xe1be>
   3093a: 4c 8d 05 d9 e1 fd ff         	leaq	-0x21e27(%rip), %r8     # 0xeb1a <strncmp+0xeb1a>
   30941: ba 3f 00 00 00               	movl	$0x3f, %edx
   30946: e8 35 06 00 00               	callq	0x30f80 <_ZN3c106detail23torchInternalAssertFailEPKcS2_jS2_S2_@plt>
   3094b: 48 8b 36                     	movq	(%rsi), %rsi
   3094e: 48 85 f6                     	testq	%rsi, %rsi
   30951: 0f 85 72 f0 ff ff            	jne	0x2f9c9 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x7b9>
   30957: 48 8d 3d 78 ca fd ff         	leaq	-0x23588(%rip), %rdi    # 0xd3d6 <strncmp+0xd3d6>
   3095e: 48 8d 35 dc b1 fd ff         	leaq	-0x24e24(%rip), %rsi    # 0xbb41 <strncmp+0xbb41>
   30965: 48 8d 0d 52 d8 fd ff         	leaq	-0x227ae(%rip), %rcx    # 0xe1be <strncmp+0xe1be>
   3096c: 4c 8d 05 a7 e1 fd ff         	leaq	-0x21e59(%rip), %r8     # 0xeb1a <strncmp+0xeb1a>
   30973: ba 3f 00 00 00               	movl	$0x3f, %edx
   30978: e8 03 06 00 00               	callq	0x30f80 <_ZN3c106detail23torchInternalAssertFailEPKcS2_jS2_S2_@plt>
   3097d: 31 ff                        	xorl	%edi, %edi
   3097f: 31 d2                        	xorl	%edx, %edx
   30981: e8 ba 05 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   30986: 49 8b 75 40                  	movq	0x40(%r13), %rsi
   3098a: 49 83 c5 48                  	addq	$0x48, %r13
   3098e: 48 83 fe 06                  	cmpq	$0x6, %rsi
   30992: 0f 82 c0 f0 ff ff            	jb	0x2fa58 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x848>
   30998: 4d 8b 6d 00                  	movq	(%r13), %r13
   3099c: 4d 8b 6c c5 00               	movq	(%r13,%rax,8), %r13
   309a1: 49 8b 2e                     	movq	(%r14), %rbp
   309a4: f6 85 ae 00 00 00 08         	testb	$0x8, 0xae(%rbp)
   309ab: 0f 84 bc f0 ff ff            	je	0x2fa6d <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x85d>
   309b1: e9 57 fb ff ff               	jmp	0x3050d <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x12fd>
   309b6: bf 02 00 00 00               	movl	$0x2, %edi
   309bb: 31 d2                        	xorl	%edx, %edx
   309bd: e8 7e 05 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   309c2: 48 8b 75 40                  	movq	0x40(%rbp), %rsi
   309c6: 48 83 c5 48                  	addq	$0x48, %rbp
   309ca: 48 83 fe 06                  	cmpq	$0x6, %rsi
   309ce: 0f 82 ba f0 ff ff            	jb	0x2fa8e <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x87e>
   309d4: 48 8b 6d 00                  	movq	(%rbp), %rbp
   309d8: 48 8b 6c c5 00               	movq	(%rbp,%rax,8), %rbp
   309dd: 49 8b 1c 24                  	movq	(%r12), %rbx
   309e1: f6 83 ae 00 00 00 08         	testb	$0x8, 0xae(%rbx)
   309e8: 0f 84 b6 f0 ff ff            	je	0x2faa4 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x894>
   309ee: e9 3d fb ff ff               	jmp	0x30530 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1320>
   309f3: bf 01 00 00 00               	movl	$0x1, %edi
   309f8: 31 d2                        	xorl	%edx, %edx
   309fa: e8 41 05 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   309ff: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   30a03: 48 83 c3 48                  	addq	$0x48, %rbx
   30a07: 48 83 fe 06                  	cmpq	$0x6, %rsi
   30a0b: 0f 82 b4 f0 ff ff            	jb	0x2fac5 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x8b5>
   30a11: 48 8b 1b                     	movq	(%rbx), %rbx
   30a14: e9 ac f0 ff ff               	jmp	0x2fac5 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x8b5>
   30a19: 48 8b 03                     	movq	(%rbx), %rax
   30a1c: be 01 00 00 00               	movl	$0x1, %esi
   30a21: 48 89 df                     	movq	%rbx, %rdi
   30a24: ff 50 30                     	callq	*0x30(%rax)
   30a27: 48 83 f8 08                  	cmpq	$0x8, %rax
   30a2b: 0f 84 9e ee ff ff            	je	0x2f8cf <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x6bf>
   30a31: eb 6f                        	jmp	0x30aa2 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1892>
   30a33: 48 89 fb                     	movq	%rdi, %rbx
   30a36: e8 f5 05 00 00               	callq	0x31030 <_ZNK3c1017SymbolicShapeMeta18init_is_contiguousEv@plt>
   30a3b: 48 89 df                     	movq	%rbx, %rdi
   30a3e: e9 7b f4 ff ff               	jmp	0x2febe <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xcae>
   30a43: 48 89 fb                     	movq	%rdi, %rbx
   30a46: e8 e5 05 00 00               	callq	0x31030 <_ZNK3c1017SymbolicShapeMeta18init_is_contiguousEv@plt>
   30a4b: 48 89 df                     	movq	%rbx, %rdi
   30a4e: e9 ce f4 ff ff               	jmp	0x2ff21 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xd11>
   30a53: 48 89 fb                     	movq	%rdi, %rbx
   30a56: e8 d5 05 00 00               	callq	0x31030 <_ZNK3c1017SymbolicShapeMeta18init_is_contiguousEv@plt>
   30a5b: 48 89 df                     	movq	%rbx, %rdi
   30a5e: e9 21 f5 ff ff               	jmp	0x2ff84 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xd74>
   30a63: 48 89 fb                     	movq	%rdi, %rbx
   30a66: e8 c5 05 00 00               	callq	0x31030 <_ZNK3c1017SymbolicShapeMeta18init_is_contiguousEv@plt>
   30a6b: 48 89 df                     	movq	%rbx, %rdi
   30a6e: e9 74 f5 ff ff               	jmp	0x2ffe7 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0xdd7>
   30a73: bf 01 00 00 00               	movl	$0x1, %edi
   30a78: 31 d2                        	xorl	%edx, %edx
   30a7a: e8 c1 04 00 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   30a7f: 48 8b 73 40                  	movq	0x40(%rbx), %rsi
   30a83: 48 83 c3 48                  	addq	$0x48, %rbx
   30a87: 48 83 fe 06                  	cmpq	$0x6, %rsi
   30a8b: 0f 82 30 ee ff ff            	jb	0x2f8c1 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x6b1>
   30a91: 48 8b 1b                     	movq	(%rbx), %rbx
   30a94: 48 8b 04 c3                  	movq	(%rbx,%rax,8), %rax
   30a98: 48 83 f8 08                  	cmpq	$0x8, %rax
   30a9c: 0f 84 2d ee ff ff            	je	0x2f8cf <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x6bf>
   30aa2: 48 8d 3d b2 e3 fd ff         	leaq	-0x21c4e(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   30aa9: 48 8d 35 66 d8 fd ff         	leaq	-0x2279a(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   30ab0: 48 8d 0d 37 bb fd ff         	leaq	-0x244c9(%rip), %rcx    # 0xc5ee <strncmp+0xc5ee>
   30ab7: ba 2d 00 00 00               	movl	$0x2d, %edx
   30abc: e8 6f 04 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   30ac1: 48 8d 3d ef c1 fd ff         	leaq	-0x23e11(%rip), %rdi    # 0xccb7 <strncmp+0xccb7>
   30ac8: 48 8d 35 c2 c3 fd ff         	leaq	-0x23c3e(%rip), %rsi    # 0xce91 <strncmp+0xce91>
   30acf: 48 8d 0d f0 c1 fd ff         	leaq	-0x23e10(%rip), %rcx    # 0xccc6 <strncmp+0xccc6>
   30ad6: ba 11 05 00 00               	movl	$0x511, %edx            # imm = 0x511
   30adb: e8 50 04 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   30ae0: 48 8d 3d cf d8 fd ff         	leaq	-0x22731(%rip), %rdi    # 0xe3b6 <strncmp+0xe3b6>
   30ae7: 48 8d 35 a3 c3 fd ff         	leaq	-0x23c5d(%rip), %rsi    # 0xce91 <strncmp+0xce91>
   30aee: 48 8d 0d 66 df fd ff         	leaq	-0x2209a(%rip), %rcx    # 0xea5b <strncmp+0xea5b>
   30af5: ba e5 06 00 00               	movl	$0x6e5, %edx            # imm = 0x6E5
   30afa: e8 41 05 00 00               	callq	0x31040 <_ZN3c106detail23torchInternalAssertFailEPKcS2_jS2_NS0_22CompileTimeEmptyStringE@plt>
   30aff: 48 8d 3d 55 e3 fd ff         	leaq	-0x21cab(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   30b06: 48 8d 35 09 d8 fd ff         	leaq	-0x227f7(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   30b0d: 48 8d 0d 80 d0 fd ff         	leaq	-0x22f80(%rip), %rcx    # 0xdb94 <strncmp+0xdb94>
   30b14: ba 30 00 00 00               	movl	$0x30, %edx
   30b19: e8 12 04 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   30b1e: 48 8d 3d 36 e3 fd ff         	leaq	-0x21cca(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   30b25: 48 8d 35 ea d7 fd ff         	leaq	-0x22816(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   30b2c: 48 8d 0d 13 cf fd ff         	leaq	-0x230ed(%rip), %rcx    # 0xda46 <strncmp+0xda46>
   30b33: ba 1c 00 00 00               	movl	$0x1c, %edx
   30b38: e8 f3 03 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   30b3d: 48 8d 3d 17 e3 fd ff         	leaq	-0x21ce9(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   30b44: 48 8d 35 cb d7 fd ff         	leaq	-0x22835(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   30b4b: 48 8d 0d 2f d0 fd ff         	leaq	-0x22fd1(%rip), %rcx    # 0xdb81 <strncmp+0xdb81>
   30b52: ba 1d 00 00 00               	movl	$0x1d, %edx
   30b57: e8 d4 03 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   30b5c: 48 8d 3d f8 e2 fd ff         	leaq	-0x21d08(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   30b63: 48 8d 35 ac d7 fd ff         	leaq	-0x22854(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   30b6a: 48 8d 0d 6a cb fd ff         	leaq	-0x23496(%rip), %rcx    # 0xd6db <strncmp+0xd6db>
   30b71: ba 1e 00 00 00               	movl	$0x1e, %edx
   30b76: e8 b5 03 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   30b7b: 48 8d 3d d9 e2 fd ff         	leaq	-0x21d27(%rip), %rdi    # 0xee5b <strncmp+0xee5b>
   30b82: 48 8d 35 8d d7 fd ff         	leaq	-0x22873(%rip), %rsi    # 0xe316 <strncmp+0xe316>
   30b89: 48 8d 0d a1 dd fd ff         	leaq	-0x2225f(%rip), %rcx    # 0xe931 <strncmp+0xe931>
   30b90: ba 1f 00 00 00               	movl	$0x1f, %edx
   30b95: e8 96 03 00 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   30b9a: 48 8d 05 64 ca fd ff         	leaq	-0x2359c(%rip), %rax    # 0xd605 <strncmp+0xd605>
   30ba1: 48 89 44 24 30               	movq	%rax, 0x30(%rsp)
   30ba6: 48 8d 7c 24 68               	leaq	0x68(%rsp), %rdi
   30bab: 48 8d 74 24 30               	leaq	0x30(%rsp), %rsi
   30bb0: 48 8d 94 24 88 00 00 00      	leaq	0x88(%rsp), %rdx
   30bb8: e8 a3 0a 00 00               	callq	0x31660 <_ZN3c106detail12_str_wrapperIJPKcRKlEE4callB5cxx11ERKS3_S5_@plt>
   30bbd: 48 8d 3d a8 c8 fd ff         	leaq	-0x23758(%rip), %rdi    # 0xd46c <strncmp+0xd46c>
   30bc4: 48 8d 35 11 bb fd ff         	leaq	-0x244ef(%rip), %rsi    # 0xc6dc <strncmp+0xc6dc>
   30bcb: 48 8d 4c 24 68               	leaq	0x68(%rsp), %rcx
   30bd0: ba 53 00 00 00               	movl	$0x53, %edx
   30bd5: e8 86 09 00 00               	callq	0x31560 <_ZN3c106detail14torchCheckFailEPKcS2_jRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE@plt>
   30bda: 48 8d 3d df d1 fd ff         	leaq	-0x22e21(%rip), %rdi    # 0xddc0 <strncmp+0xddc0>
   30be1: 48 8d 35 6e d0 fd ff         	leaq	-0x22f92(%rip), %rsi    # 0xdc56 <strncmp+0xdc56>
   30be8: 48 8d 0d 57 b3 fd ff         	leaq	-0x24ca9(%rip), %rcx    # 0xbf46 <strncmp+0xbf46>
   30bef: 4c 8d 05 96 cb fd ff         	leaq	-0x2346a(%rip), %r8     # 0xd78c <strncmp+0xd78c>
   30bf6: ba 14 01 00 00               	movl	$0x114, %edx            # imm = 0x114
   30bfb: e8 80 03 00 00               	callq	0x30f80 <_ZN3c106detail23torchInternalAssertFailEPKcS2_jS2_S2_@plt>
   30c00: 48 8d 3d b9 d1 fd ff         	leaq	-0x22e47(%rip), %rdi    # 0xddc0 <strncmp+0xddc0>
   30c07: 48 8d 35 48 d0 fd ff         	leaq	-0x22fb8(%rip), %rsi    # 0xdc56 <strncmp+0xdc56>
   30c0e: 48 8d 0d 31 b3 fd ff         	leaq	-0x24ccf(%rip), %rcx    # 0xbf46 <strncmp+0xbf46>
   30c15: 4c 8d 05 70 cb fd ff         	leaq	-0x23490(%rip), %r8     # 0xd78c <strncmp+0xd78c>
   30c1c: ba 14 01 00 00               	movl	$0x114, %edx            # imm = 0x114
   30c21: e8 5a 03 00 00               	callq	0x30f80 <_ZN3c106detail23torchInternalAssertFailEPKcS2_jS2_S2_@plt>
   30c26: 48 89 c7                     	movq	%rax, %rdi
   30c29: e8 a2 5a fe ff               	callq	0x166d0 <__clang_call_terminate>
   30c2e: 48 89 c7                     	movq	%rax, %rdi
   30c31: e8 9a 5a fe ff               	callq	0x166d0 <__clang_call_terminate>
   30c36: 48 89 c3                     	movq	%rax, %rbx
   30c39: e9 b2 00 00 00               	jmp	0x30cf0 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1ae0>
   30c3e: 48 89 c3                     	movq	%rax, %rbx
   30c41: e9 aa 00 00 00               	jmp	0x30cf0 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1ae0>
   30c46: 48 89 c3                     	movq	%rax, %rbx
   30c49: e9 a2 00 00 00               	jmp	0x30cf0 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1ae0>
   30c4e: 48 89 c7                     	movq	%rax, %rdi
   30c51: e8 7a 5a fe ff               	callq	0x166d0 <__clang_call_terminate>
   30c56: 48 89 c7                     	movq	%rax, %rdi
   30c59: e8 72 5a fe ff               	callq	0x166d0 <__clang_call_terminate>
   30c5e: 48 89 c7                     	movq	%rax, %rdi
   30c61: e8 6a 5a fe ff               	callq	0x166d0 <__clang_call_terminate>
   30c66: 48 89 c7                     	movq	%rax, %rdi
   30c69: e8 62 5a fe ff               	callq	0x166d0 <__clang_call_terminate>
   30c6e: 48 89 c7                     	movq	%rax, %rdi
   30c71: e8 5a 5a fe ff               	callq	0x166d0 <__clang_call_terminate>
   30c76: 48 89 c7                     	movq	%rax, %rdi
   30c79: e8 52 5a fe ff               	callq	0x166d0 <__clang_call_terminate>
   30c7e: 48 89 c7                     	movq	%rax, %rdi
   30c81: e8 4a 5a fe ff               	callq	0x166d0 <__clang_call_terminate>
   30c86: 48 89 c7                     	movq	%rax, %rdi
   30c89: e8 42 5a fe ff               	callq	0x166d0 <__clang_call_terminate>
   30c8e: eb 49                        	jmp	0x30cd9 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1ac9>
   30c90: 48 89 c3                     	movq	%rax, %rbx
   30c93: 48 8b 7c 24 68               	movq	0x68(%rsp), %rdi
   30c98: 48 8d 44 24 78               	leaq	0x78(%rsp), %rax
   30c9d: 48 39 c7                     	cmpq	%rax, %rdi
   30ca0: 74 3a                        	je	0x30cdc <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1acc>
   30ca2: 48 8b 74 24 78               	movq	0x78(%rsp), %rsi
   30ca7: 48 ff c6                     	incq	%rsi
   30caa: e8 21 02 00 00               	callq	0x30ed0 <_ZdlPvm@plt>
   30caf: eb 2b                        	jmp	0x30cdc <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1acc>
   30cb1: 48 89 c3                     	movq	%rax, %rbx
   30cb4: eb 30                        	jmp	0x30ce6 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1ad6>
   30cb6: 48 89 c3                     	movq	%rax, %rbx
   30cb9: eb 35                        	jmp	0x30cf0 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1ae0>
   30cbb: 48 89 c3                     	movq	%rax, %rbx
   30cbe: 48 8d 7c 24 30               	leaq	0x30(%rsp), %rdi
   30cc3: e8 88 03 00 00               	callq	0x31050 <_ZN2at10TensorBaseD2Ev@plt>
   30cc8: eb 12                        	jmp	0x30cdc <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1acc>
   30cca: 48 89 c3                     	movq	%rax, %rbx
   30ccd: 48 8d 7c 24 40               	leaq	0x40(%rsp), %rdi
   30cd2: e8 d9 10 00 00               	callq	0x31db0 <_ZNSt14_Optional_baseIN2at6TensorELb0ELb0EED2Ev@plt>
   30cd7: eb 03                        	jmp	0x30cdc <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1acc>
   30cd9: 48 89 c3                     	movq	%rax, %rbx
   30cdc: 48 8d 7c 24 20               	leaq	0x20(%rsp), %rdi
   30ce1: e8 6a 03 00 00               	callq	0x31050 <_ZN2at10TensorBaseD2Ev@plt>
   30ce6: 48 8d 7c 24 28               	leaq	0x28(%rsp), %rdi
   30ceb: e8 60 03 00 00               	callq	0x31050 <_ZN2at10TensorBaseD2Ev@plt>
   30cf0: 0f b6 84 24 a8 00 00 00      	movzbl	0xa8(%rsp), %eax
   30cf8: c6 84 24 a8 00 00 00 00      	movb	$0x0, 0xa8(%rsp)
   30d00: 3c 01                        	cmpb	$0x1, %al
   30d02: 75 15                        	jne	0x30d19 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_+0x1b09>
   30d04: 48 8b bc 24 98 00 00 00      	movq	0x98(%rsp), %rdi
   30d0c: 48 8b 07                     	movq	(%rdi), %rax
   30d0f: 8b b4 24 a0 00 00 00         	movl	0xa0(%rsp), %esi
   30d16: ff 50 20                     	callq	*0x20(%rax)
   30d19: 48 89 df                     	movq	%rbx, %rdi
   30d1c: e8 cf 01 00 00               	callq	0x30ef0 <_Unwind_Resume@plt>
   30d21: cc                           	int3
   30d22: cc                           	int3
   30d23: cc                           	int3
   30d24: cc                           	int3
   30d25: cc                           	int3
   30d26: cc                           	int3
   30d27: cc                           	int3
   30d28: cc                           	int3
   30d29: cc                           	int3
   30d2a: cc                           	int3
   30d2b: cc                           	int3
   30d2c: cc                           	int3
   30d2d: cc                           	int3
   30d2e: cc                           	int3
   30d2f: cc                           	int3
