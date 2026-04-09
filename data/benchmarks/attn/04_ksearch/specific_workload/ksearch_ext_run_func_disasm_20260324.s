
/root/.cache/torch_extensions/py312_cpu/ksearch_dense_qkv_prefill_causal_h8_kv1or8_d128_ext_src_v1/ksearch_dense_qkv_prefill_causal_h8_kv1or8_d128_ext_src_v1.so:	file format elf64-x86-64

Disassembly of section .text:

0000000000014460 <_Z3runN2at6TensorES0_S0_d>:
   14460: 55                           	pushq	%rbp
   14461: 41 57                        	pushq	%r15
   14463: 41 56                        	pushq	%r14
   14465: 41 55                        	pushq	%r13
   14467: 41 54                        	pushq	%r12
   14469: 53                           	pushq	%rbx
   1446a: 48 81 ec d8 00 00 00         	subq	$0xd8, %rsp
   14471: f2 0f 11 84 24 a8 00 00 00   	movsd	%xmm0, 0xa8(%rsp)
   1447a: 49 89 cc                     	movq	%rcx, %r12
   1447d: 48 89 d5                     	movq	%rdx, %rbp
   14480: 49 89 f5                     	movq	%rsi, %r13
   14483: 49 89 ff                     	movq	%rdi, %r15
   14486: e8 85 ca 01 00               	callq	0x30f10 <_ZN3c108GradMode10is_enabledEv@plt>
   1448b: 88 44 24 15                  	movb	%al, 0x15(%rsp)
   1448f: 31 ff                        	xorl	%edi, %edi
   14491: e8 8a ca 01 00               	callq	0x30f20 <_ZN3c108GradMode11set_enabledEb@plt>
   14496: 49 8b 7d 00                  	movq	(%r13), %rdi
   1449a: f6 87 ae 00 00 00 08         	testb	$0x8, 0xae(%rdi)
   144a1: 0f 85 d6 04 00 00            	jne	0x1497d <_Z3runN2at6TensorES0_S0_d+0x51d>
   144a7: 48 8b 47 40                  	movq	0x40(%rdi), %rax
   144ab: 48 83 f8 04                  	cmpq	$0x4, %rax
   144af: 0f 85 ab 12 00 00            	jne	0x15760 <_Z3runN2at6TensorES0_S0_d+0x1300>
   144b5: 48 8b 7d 00                  	movq	(%rbp), %rdi
   144b9: f6 87 ae 00 00 00 08         	testb	$0x8, 0xae(%rdi)
   144c0: 0f 85 c2 04 00 00            	jne	0x14988 <_Z3runN2at6TensorES0_S0_d+0x528>
   144c6: 48 8b 47 40                  	movq	0x40(%rdi), %rax
   144ca: 48 83 f8 04                  	cmpq	$0x4, %rax
   144ce: 0f 85 9a 12 00 00            	jne	0x1576e <_Z3runN2at6TensorES0_S0_d+0x130e>
   144d4: 49 8b 3c 24                  	movq	(%r12), %rdi
   144d8: f6 87 ae 00 00 00 08         	testb	$0x8, 0xae(%rdi)
   144df: 0f 85 ae 04 00 00            	jne	0x14993 <_Z3runN2at6TensorES0_S0_d+0x533>
   144e5: 48 8b 47 40                  	movq	0x40(%rdi), %rax
   144e9: 48 83 f8 04                  	cmpq	$0x4, %rax
   144ed: 0f 85 89 12 00 00            	jne	0x1577c <_Z3runN2at6TensorES0_S0_d+0x131c>
   144f3: 4d 8b 75 00                  	movq	(%r13), %r14
   144f7: 41 f6 86 ae 00 00 00 08      	testb	$0x8, 0xae(%r14)
   144ff: 0f 85 99 04 00 00            	jne	0x1499e <_Z3runN2at6TensorES0_S0_d+0x53e>
   14505: 49 8b 76 40                  	movq	0x40(%r14), %rsi
   14509: 48 85 f6                     	testq	%rsi, %rsi
   1450c: 0f 8e c4 05 00 00            	jle	0x14ad6 <_Z3runN2at6TensorES0_S0_d+0x676>
   14512: 31 c0                        	xorl	%eax, %eax
   14514: 49 83 c6 48                  	addq	$0x48, %r14
   14518: 48 83 fe 06                  	cmpq	$0x6, %rsi
   1451c: 0f 83 cf 05 00 00            	jae	0x14af1 <_Z3runN2at6TensorES0_S0_d+0x691>
   14522: 49 8b 1c c6                  	movq	(%r14,%rax,8), %rbx
   14526: 4d 8b 75 00                  	movq	(%r13), %r14
   1452a: 41 f6 86 ae 00 00 00 08      	testb	$0x8, 0xae(%r14)
   14532: 0f 85 86 04 00 00            	jne	0x149be <_Z3runN2at6TensorES0_S0_d+0x55e>
   14538: 49 8b 76 40                  	movq	0x40(%r14), %rsi
   1453c: b8 01 00 00 00               	movl	$0x1, %eax
   14541: 48 83 fe 01                  	cmpq	$0x1, %rsi
   14545: 0f 8e c4 05 00 00            	jle	0x14b0f <_Z3runN2at6TensorES0_S0_d+0x6af>
   1454b: 49 83 c6 48                  	addq	$0x48, %r14
   1454f: 48 83 fe 06                  	cmpq	$0x6, %rsi
   14553: 0f 83 d4 05 00 00            	jae	0x14b2d <_Z3runN2at6TensorES0_S0_d+0x6cd>
   14559: 49 8b 04 c6                  	movq	(%r14,%rax,8), %rax
   1455d: 48 89 84 24 c0 00 00 00      	movq	%rax, 0xc0(%rsp)
   14565: 4d 8b 75 00                  	movq	(%r13), %r14
   14569: 41 f6 86 ae 00 00 00 08      	testb	$0x8, 0xae(%r14)
   14571: 0f 85 5a 04 00 00            	jne	0x149d1 <_Z3runN2at6TensorES0_S0_d+0x571>
   14577: 49 8b 76 40                  	movq	0x40(%r14), %rsi
   1457b: b8 02 00 00 00               	movl	$0x2, %eax
   14580: 48 83 fe 02                  	cmpq	$0x2, %rsi
   14584: 0f 8e ab 05 00 00            	jle	0x14b35 <_Z3runN2at6TensorES0_S0_d+0x6d5>
   1458a: 49 83 c6 48                  	addq	$0x48, %r14
   1458e: 48 83 fe 06                  	cmpq	$0x6, %rsi
   14592: 0f 83 bb 05 00 00            	jae	0x14b53 <_Z3runN2at6TensorES0_S0_d+0x6f3>
   14598: 49 8b 04 c6                  	movq	(%r14,%rax,8), %rax
   1459c: 48 89 84 24 80 00 00 00      	movq	%rax, 0x80(%rsp)
   145a4: 4d 8b 75 00                  	movq	(%r13), %r14
   145a8: 41 f6 86 ae 00 00 00 08      	testb	$0x8, 0xae(%r14)
   145b0: 0f 85 2e 04 00 00            	jne	0x149e4 <_Z3runN2at6TensorES0_S0_d+0x584>
   145b6: 49 8b 76 40                  	movq	0x40(%r14), %rsi
   145ba: b8 03 00 00 00               	movl	$0x3, %eax
   145bf: 48 83 fe 03                  	cmpq	$0x3, %rsi
   145c3: 0f 8e 92 05 00 00            	jle	0x14b5b <_Z3runN2at6TensorES0_S0_d+0x6fb>
   145c9: 49 83 c6 48                  	addq	$0x48, %r14
   145cd: 48 83 fe 06                  	cmpq	$0x6, %rsi
   145d1: 0f 83 a2 05 00 00            	jae	0x14b79 <_Z3runN2at6TensorES0_S0_d+0x719>
   145d7: 49 8b 04 c6                  	movq	(%r14,%rax,8), %rax
   145db: 48 89 44 24 78               	movq	%rax, 0x78(%rsp)
   145e0: 4c 8b 75 00                  	movq	(%rbp), %r14
   145e4: 41 f6 86 ae 00 00 00 08      	testb	$0x8, 0xae(%r14)
   145ec: 0f 85 05 04 00 00            	jne	0x149f7 <_Z3runN2at6TensorES0_S0_d+0x597>
   145f2: 49 8b 76 40                  	movq	0x40(%r14), %rsi
   145f6: 48 85 f6                     	testq	%rsi, %rsi
   145f9: 0f 8e 82 05 00 00            	jle	0x14b81 <_Z3runN2at6TensorES0_S0_d+0x721>
   145ff: 31 c0                        	xorl	%eax, %eax
   14601: 49 83 c6 48                  	addq	$0x48, %r14
   14605: 48 83 fe 06                  	cmpq	$0x6, %rsi
   14609: 0f 83 8d 05 00 00            	jae	0x14b9c <_Z3runN2at6TensorES0_S0_d+0x73c>
   1460f: 49 8b 04 c6                  	movq	(%r14,%rax,8), %rax
   14613: 48 89 84 24 d0 00 00 00      	movq	%rax, 0xd0(%rsp)
   1461b: 4c 8b 75 00                  	movq	(%rbp), %r14
   1461f: 41 f6 86 ae 00 00 00 08      	testb	$0x8, 0xae(%r14)
   14627: 4c 89 bc 24 b8 00 00 00      	movq	%r15, 0xb8(%rsp)
   1462f: 0f 85 d2 03 00 00            	jne	0x14a07 <_Z3runN2at6TensorES0_S0_d+0x5a7>
   14635: 49 8b 76 40                  	movq	0x40(%r14), %rsi
   14639: b8 01 00 00 00               	movl	$0x1, %eax
   1463e: 48 83 fe 01                  	cmpq	$0x1, %rsi
   14642: 0f 8e 5c 05 00 00            	jle	0x14ba4 <_Z3runN2at6TensorES0_S0_d+0x744>
   14648: 49 83 c6 48                  	addq	$0x48, %r14
   1464c: 48 83 fe 06                  	cmpq	$0x6, %rsi
   14650: 0f 83 6c 05 00 00            	jae	0x14bc2 <_Z3runN2at6TensorES0_S0_d+0x762>
   14656: 49 8b 04 c6                  	movq	(%r14,%rax,8), %rax
   1465a: 48 89 84 24 b0 00 00 00      	movq	%rax, 0xb0(%rsp)
   14662: 4c 8b 7d 00                  	movq	(%rbp), %r15
   14666: 41 f6 87 ae 00 00 00 08      	testb	$0x8, 0xae(%r15)
   1466e: 0f 85 a6 03 00 00            	jne	0x14a1a <_Z3runN2at6TensorES0_S0_d+0x5ba>
   14674: 49 8b 77 40                  	movq	0x40(%r15), %rsi
   14678: b8 02 00 00 00               	movl	$0x2, %eax
   1467d: 48 83 fe 02                  	cmpq	$0x2, %rsi
   14681: 0f 8e 43 05 00 00            	jle	0x14bca <_Z3runN2at6TensorES0_S0_d+0x76a>
   14687: 49 83 c7 48                  	addq	$0x48, %r15
   1468b: 48 83 fe 06                  	cmpq	$0x6, %rsi
   1468f: 0f 83 53 05 00 00            	jae	0x14be8 <_Z3runN2at6TensorES0_S0_d+0x788>
   14695: 49 8b 04 c7                  	movq	(%r15,%rax,8), %rax
   14699: 48 89 84 24 c8 00 00 00      	movq	%rax, 0xc8(%rsp)
   146a1: 4c 8b 7d 00                  	movq	(%rbp), %r15
   146a5: 41 f6 87 ae 00 00 00 08      	testb	$0x8, 0xae(%r15)
   146ad: 0f 85 7a 03 00 00            	jne	0x14a2d <_Z3runN2at6TensorES0_S0_d+0x5cd>
   146b3: 49 8b 77 40                  	movq	0x40(%r15), %rsi
   146b7: b8 03 00 00 00               	movl	$0x3, %eax
   146bc: 48 83 fe 03                  	cmpq	$0x3, %rsi
   146c0: 0f 8e 2a 05 00 00            	jle	0x14bf0 <_Z3runN2at6TensorES0_S0_d+0x790>
   146c6: 49 83 c7 48                  	addq	$0x48, %r15
   146ca: 48 83 fe 06                  	cmpq	$0x6, %rsi
   146ce: 0f 83 3a 05 00 00            	jae	0x14c0e <_Z3runN2at6TensorES0_S0_d+0x7ae>
   146d4: 49 8b 04 c7                  	movq	(%r15,%rax,8), %rax
   146d8: 48 89 44 24 70               	movq	%rax, 0x70(%rsp)
   146dd: 4d 8b 3c 24                  	movq	(%r12), %r15
   146e1: 41 f6 87 ae 00 00 00 08      	testb	$0x8, 0xae(%r15)
   146e9: 4c 89 6c 24 40               	movq	%r13, 0x40(%rsp)
   146ee: 0f 85 4c 03 00 00            	jne	0x14a40 <_Z3runN2at6TensorES0_S0_d+0x5e0>
   146f4: 49 8b 77 40                  	movq	0x40(%r15), %rsi
   146f8: 48 85 f6                     	testq	%rsi, %rsi
   146fb: 0f 8e 15 05 00 00            	jle	0x14c16 <_Z3runN2at6TensorES0_S0_d+0x7b6>
   14701: 31 c0                        	xorl	%eax, %eax
   14703: 49 83 c7 48                  	addq	$0x48, %r15
   14707: 49 89 de                     	movq	%rbx, %r14
   1470a: 4c 89 e3                     	movq	%r12, %rbx
   1470d: 48 83 fe 06                  	cmpq	$0x6, %rsi
   14711: 0f 83 11 05 00 00            	jae	0x14c28 <_Z3runN2at6TensorES0_S0_d+0x7c8>
   14717: 4d 8b 2c c7                  	movq	(%r15,%rax,8), %r13
   1471b: 4c 8b 3b                     	movq	(%rbx), %r15
   1471e: 41 f6 87 ae 00 00 00 08      	testb	$0x8, 0xae(%r15)
   14726: 48 89 6c 24 48               	movq	%rbp, 0x48(%rsp)
   1472b: 0f 85 39 03 00 00            	jne	0x14a6a <_Z3runN2at6TensorES0_S0_d+0x60a>
   14731: 49 8b 77 40                  	movq	0x40(%r15), %rsi
   14735: b8 01 00 00 00               	movl	$0x1, %eax
   1473a: 48 83 fe 01                  	cmpq	$0x1, %rsi
   1473e: 0f 8e ec 04 00 00            	jle	0x14c30 <_Z3runN2at6TensorES0_S0_d+0x7d0>
   14744: 49 83 c7 48                  	addq	$0x48, %r15
   14748: 48 83 fe 06                  	cmpq	$0x6, %rsi
   1474c: 0f 83 fc 04 00 00            	jae	0x14c4e <_Z3runN2at6TensorES0_S0_d+0x7ee>
   14752: 49 8b 2c c7                  	movq	(%r15,%rax,8), %rbp
   14756: 4c 8b 3b                     	movq	(%rbx), %r15
   14759: 41 f6 87 ae 00 00 00 08      	testb	$0x8, 0xae(%r15)
   14761: 0f 85 25 03 00 00            	jne	0x14a8c <_Z3runN2at6TensorES0_S0_d+0x62c>
   14767: 49 8b 77 40                  	movq	0x40(%r15), %rsi
   1476b: b8 02 00 00 00               	movl	$0x2, %eax
   14770: 48 83 fe 02                  	cmpq	$0x2, %rsi
   14774: 0f 8e f1 04 00 00            	jle	0x14c6b <_Z3runN2at6TensorES0_S0_d+0x80b>
   1477a: 49 83 c7 48                  	addq	$0x48, %r15
   1477e: 48 83 fe 06                  	cmpq	$0x6, %rsi
   14782: 0f 83 01 05 00 00            	jae	0x14c89 <_Z3runN2at6TensorES0_S0_d+0x829>
   14788: 4d 8b 3c c7                  	movq	(%r15,%rax,8), %r15
   1478c: 4c 8b 23                     	movq	(%rbx), %r12
   1478f: 41 f6 84 24 ae 00 00 00 08   	testb	$0x8, 0xae(%r12)
   14798: 0f 85 11 03 00 00            	jne	0x14aaf <_Z3runN2at6TensorES0_S0_d+0x64f>
   1479e: 49 8b 74 24 40               	movq	0x40(%r12), %rsi
   147a3: b8 03 00 00 00               	movl	$0x3, %eax
   147a8: 48 83 fe 03                  	cmpq	$0x3, %rsi
   147ac: 0f 8e f5 04 00 00            	jle	0x14ca7 <_Z3runN2at6TensorES0_S0_d+0x847>
   147b2: 49 83 c4 48                  	addq	$0x48, %r12
   147b6: 48 83 fe 06                  	cmpq	$0x6, %rsi
   147ba: 0f 83 06 05 00 00            	jae	0x14cc6 <_Z3runN2at6TensorES0_S0_d+0x866>
   147c0: 49 8b 04 c4                  	movq	(%r12,%rax,8), %rax
   147c4: 48 8d 0d cd 9f ff ff         	leaq	-0x6033(%rip), %rcx     # 0xe798 <strncmp+0xe798>
   147cb: ba 13 00 00 00               	movl	$0x13, %edx
   147d0: 4c 3b b4 24 d0 00 00 00      	cmpq	0xd0(%rsp), %r14
   147d8: 0f 85 f3 0f 00 00            	jne	0x157d1 <_Z3runN2at6TensorES0_S0_d+0x1371>
   147de: 4d 39 ee                     	cmpq	%r13, %r14
   147e1: 48 8b 7c 24 78               	movq	0x78(%rsp), %rdi
   147e6: 0f 85 e5 0f 00 00            	jne	0x157d1 <_Z3runN2at6TensorES0_S0_d+0x1371>
   147ec: 48 8d 0d 22 99 ff ff         	leaq	-0x66de(%rip), %rcx     # 0xe115 <strncmp+0xe115>
   147f3: ba 14 00 00 00               	movl	$0x14, %edx
   147f8: 48 8b b4 24 80 00 00 00      	movq	0x80(%rsp), %rsi
   14800: 48 3b b4 24 c8 00 00 00      	cmpq	0xc8(%rsp), %rsi
   14808: 0f 85 c3 0f 00 00            	jne	0x157d1 <_Z3runN2at6TensorES0_S0_d+0x1371>
   1480e: 4c 39 fe                     	cmpq	%r15, %rsi
   14811: 0f 85 ba 0f 00 00            	jne	0x157d1 <_Z3runN2at6TensorES0_S0_d+0x1371>
   14817: 48 8d 0d 5a 91 ff ff         	leaq	-0x6ea6(%rip), %rcx     # 0xd978 <strncmp+0xd978>
   1481e: ba 15 00 00 00               	movl	$0x15, %edx
   14823: 48 3b 7c 24 70               	cmpq	0x70(%rsp), %rdi
   14828: 0f 85 a3 0f 00 00            	jne	0x157d1 <_Z3runN2at6TensorES0_S0_d+0x1371>
   1482e: 48 39 c7                     	cmpq	%rax, %rdi
   14831: 0f 85 9a 0f 00 00            	jne	0x157d1 <_Z3runN2at6TensorES0_S0_d+0x1371>
   14837: 48 83 bc 24 c0 00 00 00 08   	cmpq	$0x8, 0xc0(%rsp)
   14840: 0f 85 55 0f 00 00            	jne	0x1579b <_Z3runN2at6TensorES0_S0_d+0x133b>
   14846: 49 89 dc                     	movq	%rbx, %r12
   14849: 4d 89 f5                     	movq	%r14, %r13
   1484c: 48 8b 84 24 b0 00 00 00      	movq	0xb0(%rsp), %rax
   14854: 48 83 f8 08                  	cmpq	$0x8, %rax
   14858: 4c 8b bc 24 b8 00 00 00      	movq	0xb8(%rsp), %r15
   14860: 74 0a                        	je	0x1486c <_Z3runN2at6TensorES0_S0_d+0x40c>
   14862: 48 83 f8 01                  	cmpq	$0x1, %rax
   14866: 0f 85 59 0f 00 00            	jne	0x157c5 <_Z3runN2at6TensorES0_S0_d+0x1365>
   1486c: 48 39 c5                     	cmpq	%rax, %rbp
   1486f: 0f 85 34 0f 00 00            	jne	0x157a9 <_Z3runN2at6TensorES0_S0_d+0x1349>
   14875: 48 81 ff 80 00 00 00         	cmpq	$0x80, %rdi
   1487c: 48 8b 5c 24 48               	movq	0x48(%rsp), %rbx
   14881: 0f 85 30 0f 00 00            	jne	0x157b7 <_Z3runN2at6TensorES0_S0_d+0x1357>
   14887: 48 8b 44 24 40               	movq	0x40(%rsp), %rax
   1488c: 48 8b 38                     	movq	(%rax), %rdi
   1488f: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   14897: 0f 88 26 02 00 00            	js	0x14ac3 <_Z3runN2at6TensorES0_S0_d+0x663>
   1489d: 80 bf ac 00 00 00 01         	cmpb	$0x1, 0xac(%rdi)
   148a4: 75 09                        	jne	0x148af <_Z3runN2at6TensorES0_S0_d+0x44f>
   148a6: 80 bf aa 00 00 00 01         	cmpb	$0x1, 0xaa(%rdi)
   148ad: 74 4f                        	je	0x148fe <_Z3runN2at6TensorES0_S0_d+0x49e>
   148af: 48 8b 3b                     	movq	(%rbx), %rdi
   148b2: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   148ba: 0f 88 0f 04 00 00            	js	0x14ccf <_Z3runN2at6TensorES0_S0_d+0x86f>
   148c0: 80 bf ac 00 00 00 01         	cmpb	$0x1, 0xac(%rdi)
   148c7: 75 09                        	jne	0x148d2 <_Z3runN2at6TensorES0_S0_d+0x472>
   148c9: 80 bf aa 00 00 00 01         	cmpb	$0x1, 0xaa(%rdi)
   148d0: 74 2c                        	je	0x148fe <_Z3runN2at6TensorES0_S0_d+0x49e>
   148d2: 49 8b 3c 24                  	movq	(%r12), %rdi
   148d6: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   148de: 0f 88 2d 04 00 00            	js	0x14d11 <_Z3runN2at6TensorES0_S0_d+0x8b1>
   148e4: 80 bf ac 00 00 00 00         	cmpb	$0x0, 0xac(%rdi)
   148eb: 0f 84 2e 04 00 00            	je	0x14d1f <_Z3runN2at6TensorES0_S0_d+0x8bf>
   148f1: 80 bf aa 00 00 00 01         	cmpb	$0x1, 0xaa(%rdi)
   148f8: 0f 85 21 04 00 00            	jne	0x14d1f <_Z3runN2at6TensorES0_S0_d+0x8bf>
   148fe: 48 8b 44 24 40               	movq	0x40(%rsp), %rax
   14903: 48 8b 38                     	movq	(%rax), %rdi
   14906: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   1490e: 0f 88 ce 03 00 00            	js	0x14ce2 <_Z3runN2at6TensorES0_S0_d+0x882>
   14914: 80 bf ac 00 00 00 01         	cmpb	$0x1, 0xac(%rdi)
   1491b: 75 09                        	jne	0x14926 <_Z3runN2at6TensorES0_S0_d+0x4c6>
   1491d: 80 bf aa 00 00 00 01         	cmpb	$0x1, 0xaa(%rdi)
   14924: 74 2b                        	je	0x14951 <_Z3runN2at6TensorES0_S0_d+0x4f1>
   14926: 48 8b 3b                     	movq	(%rbx), %rdi
   14929: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   14931: 0f 88 02 04 00 00            	js	0x14d39 <_Z3runN2at6TensorES0_S0_d+0x8d9>
   14937: 80 bf ac 00 00 00 01         	cmpb	$0x1, 0xac(%rdi)
   1493e: 0f 85 1b 04 00 00            	jne	0x14d5f <_Z3runN2at6TensorES0_S0_d+0x8ff>
   14944: 80 bf aa 00 00 00 01         	cmpb	$0x1, 0xaa(%rdi)
   1494b: 0f 85 0e 04 00 00            	jne	0x14d5f <_Z3runN2at6TensorES0_S0_d+0x8ff>
   14951: 80 bf ac 00 00 00 01         	cmpb	$0x1, 0xac(%rdi)
   14958: 0f 84 20 04 00 00            	je	0x14d7e <_Z3runN2at6TensorES0_S0_d+0x91e>
   1495e: 48 8d 3d 52 83 ff ff         	leaq	-0x7cae(%rip), %rdi     # 0xccb7 <strncmp+0xccb7>
   14965: 48 8d 35 25 85 ff ff         	leaq	-0x7adb(%rip), %rsi     # 0xce91 <strncmp+0xce91>
   1496c: 48 8d 0d 53 83 ff ff         	leaq	-0x7cad(%rip), %rcx     # 0xccc6 <strncmp+0xccc6>
   14973: ba 11 05 00 00               	movl	$0x511, %edx            # imm = 0x511
   14978: e8 b3 c5 01 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   1497d: 48 8b 07                     	movq	(%rdi), %rax
   14980: ff 50 60                     	callq	*0x60(%rax)
   14983: e9 23 fb ff ff               	jmp	0x144ab <_Z3runN2at6TensorES0_S0_d+0x4b>
   14988: 48 8b 07                     	movq	(%rdi), %rax
   1498b: ff 50 60                     	callq	*0x60(%rax)
   1498e: e9 37 fb ff ff               	jmp	0x144ca <_Z3runN2at6TensorES0_S0_d+0x6a>
   14993: 48 8b 07                     	movq	(%rdi), %rax
   14996: ff 50 60                     	callq	*0x60(%rax)
   14999: e9 4b fb ff ff               	jmp	0x144e9 <_Z3runN2at6TensorES0_S0_d+0x89>
   1499e: 49 8b 06                     	movq	(%r14), %rax
   149a1: 4c 89 f7                     	movq	%r14, %rdi
   149a4: 31 f6                        	xorl	%esi, %esi
   149a6: ff 50 30                     	callq	*0x30(%rax)
   149a9: 48 89 c3                     	movq	%rax, %rbx
   149ac: 4d 8b 75 00                  	movq	(%r13), %r14
   149b0: 41 f6 86 ae 00 00 00 08      	testb	$0x8, 0xae(%r14)
   149b8: 0f 84 7a fb ff ff            	je	0x14538 <_Z3runN2at6TensorES0_S0_d+0xd8>
   149be: 49 8b 06                     	movq	(%r14), %rax
   149c1: be 01 00 00 00               	movl	$0x1, %esi
   149c6: 4c 89 f7                     	movq	%r14, %rdi
   149c9: ff 50 30                     	callq	*0x30(%rax)
   149cc: e9 8c fb ff ff               	jmp	0x1455d <_Z3runN2at6TensorES0_S0_d+0xfd>
   149d1: 49 8b 06                     	movq	(%r14), %rax
   149d4: be 02 00 00 00               	movl	$0x2, %esi
   149d9: 4c 89 f7                     	movq	%r14, %rdi
   149dc: ff 50 30                     	callq	*0x30(%rax)
   149df: e9 b8 fb ff ff               	jmp	0x1459c <_Z3runN2at6TensorES0_S0_d+0x13c>
   149e4: 49 8b 06                     	movq	(%r14), %rax
   149e7: be 03 00 00 00               	movl	$0x3, %esi
   149ec: 4c 89 f7                     	movq	%r14, %rdi
   149ef: ff 50 30                     	callq	*0x30(%rax)
   149f2: e9 e4 fb ff ff               	jmp	0x145db <_Z3runN2at6TensorES0_S0_d+0x17b>
   149f7: 49 8b 06                     	movq	(%r14), %rax
   149fa: 4c 89 f7                     	movq	%r14, %rdi
   149fd: 31 f6                        	xorl	%esi, %esi
   149ff: ff 50 30                     	callq	*0x30(%rax)
   14a02: e9 0c fc ff ff               	jmp	0x14613 <_Z3runN2at6TensorES0_S0_d+0x1b3>
   14a07: 49 8b 06                     	movq	(%r14), %rax
   14a0a: be 01 00 00 00               	movl	$0x1, %esi
   14a0f: 4c 89 f7                     	movq	%r14, %rdi
   14a12: ff 50 30                     	callq	*0x30(%rax)
   14a15: e9 40 fc ff ff               	jmp	0x1465a <_Z3runN2at6TensorES0_S0_d+0x1fa>
   14a1a: 49 8b 07                     	movq	(%r15), %rax
   14a1d: be 02 00 00 00               	movl	$0x2, %esi
   14a22: 4c 89 ff                     	movq	%r15, %rdi
   14a25: ff 50 30                     	callq	*0x30(%rax)
   14a28: e9 6c fc ff ff               	jmp	0x14699 <_Z3runN2at6TensorES0_S0_d+0x239>
   14a2d: 49 8b 07                     	movq	(%r15), %rax
   14a30: be 03 00 00 00               	movl	$0x3, %esi
   14a35: 4c 89 ff                     	movq	%r15, %rdi
   14a38: ff 50 30                     	callq	*0x30(%rax)
   14a3b: e9 98 fc ff ff               	jmp	0x146d8 <_Z3runN2at6TensorES0_S0_d+0x278>
   14a40: 49 89 de                     	movq	%rbx, %r14
   14a43: 4c 89 e3                     	movq	%r12, %rbx
   14a46: 49 8b 07                     	movq	(%r15), %rax
   14a49: 4c 89 ff                     	movq	%r15, %rdi
   14a4c: 31 f6                        	xorl	%esi, %esi
   14a4e: ff 50 30                     	callq	*0x30(%rax)
   14a51: 49 89 c5                     	movq	%rax, %r13
   14a54: 4c 8b 3b                     	movq	(%rbx), %r15
   14a57: 41 f6 87 ae 00 00 00 08      	testb	$0x8, 0xae(%r15)
   14a5f: 48 89 6c 24 48               	movq	%rbp, 0x48(%rsp)
   14a64: 0f 84 c7 fc ff ff            	je	0x14731 <_Z3runN2at6TensorES0_S0_d+0x2d1>
   14a6a: 49 8b 07                     	movq	(%r15), %rax
   14a6d: be 01 00 00 00               	movl	$0x1, %esi
   14a72: 4c 89 ff                     	movq	%r15, %rdi
   14a75: ff 50 30                     	callq	*0x30(%rax)
   14a78: 48 89 c5                     	movq	%rax, %rbp
   14a7b: 4c 8b 3b                     	movq	(%rbx), %r15
   14a7e: 41 f6 87 ae 00 00 00 08      	testb	$0x8, 0xae(%r15)
   14a86: 0f 84 db fc ff ff            	je	0x14767 <_Z3runN2at6TensorES0_S0_d+0x307>
   14a8c: 49 8b 07                     	movq	(%r15), %rax
   14a8f: be 02 00 00 00               	movl	$0x2, %esi
   14a94: 4c 89 ff                     	movq	%r15, %rdi
   14a97: ff 50 30                     	callq	*0x30(%rax)
   14a9a: 49 89 c7                     	movq	%rax, %r15
   14a9d: 4c 8b 23                     	movq	(%rbx), %r12
   14aa0: 41 f6 84 24 ae 00 00 00 08   	testb	$0x8, 0xae(%r12)
   14aa9: 0f 84 ef fc ff ff            	je	0x1479e <_Z3runN2at6TensorES0_S0_d+0x33e>
   14aaf: 49 8b 04 24                  	movq	(%r12), %rax
   14ab3: be 03 00 00 00               	movl	$0x3, %esi
   14ab8: 4c 89 e7                     	movq	%r12, %rdi
   14abb: ff 50 30                     	callq	*0x30(%rax)
   14abe: e9 01 fd ff ff               	jmp	0x147c4 <_Z3runN2at6TensorES0_S0_d+0x364>
   14ac3: 48 8b 07                     	movq	(%rdi), %rax
   14ac6: ff 50 68                     	callq	*0x68(%rax)
   14ac9: 3c 01                        	cmpb	$0x1, %al
   14acb: 0f 85 de fd ff ff            	jne	0x148af <_Z3runN2at6TensorES0_S0_d+0x44f>
   14ad1: e9 28 fe ff ff               	jmp	0x148fe <_Z3runN2at6TensorES0_S0_d+0x49e>
   14ad6: 31 ff                        	xorl	%edi, %edi
   14ad8: 31 d2                        	xorl	%edx, %edx
   14ada: e8 61 c4 01 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   14adf: 49 8b 76 40                  	movq	0x40(%r14), %rsi
   14ae3: 49 83 c6 48                  	addq	$0x48, %r14
   14ae7: 48 83 fe 06                  	cmpq	$0x6, %rsi
   14aeb: 0f 82 31 fa ff ff            	jb	0x14522 <_Z3runN2at6TensorES0_S0_d+0xc2>
   14af1: 4d 8b 36                     	movq	(%r14), %r14
   14af4: 49 8b 1c c6                  	movq	(%r14,%rax,8), %rbx
   14af8: 4d 8b 75 00                  	movq	(%r13), %r14
   14afc: 41 f6 86 ae 00 00 00 08      	testb	$0x8, 0xae(%r14)
   14b04: 0f 84 2e fa ff ff            	je	0x14538 <_Z3runN2at6TensorES0_S0_d+0xd8>
   14b0a: e9 af fe ff ff               	jmp	0x149be <_Z3runN2at6TensorES0_S0_d+0x55e>
   14b0f: bf 01 00 00 00               	movl	$0x1, %edi
   14b14: 31 d2                        	xorl	%edx, %edx
   14b16: e8 25 c4 01 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   14b1b: 49 8b 76 40                  	movq	0x40(%r14), %rsi
   14b1f: 49 83 c6 48                  	addq	$0x48, %r14
   14b23: 48 83 fe 06                  	cmpq	$0x6, %rsi
   14b27: 0f 82 2c fa ff ff            	jb	0x14559 <_Z3runN2at6TensorES0_S0_d+0xf9>
   14b2d: 4d 8b 36                     	movq	(%r14), %r14
   14b30: e9 24 fa ff ff               	jmp	0x14559 <_Z3runN2at6TensorES0_S0_d+0xf9>
   14b35: bf 02 00 00 00               	movl	$0x2, %edi
   14b3a: 31 d2                        	xorl	%edx, %edx
   14b3c: e8 ff c3 01 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   14b41: 49 8b 76 40                  	movq	0x40(%r14), %rsi
   14b45: 49 83 c6 48                  	addq	$0x48, %r14
   14b49: 48 83 fe 06                  	cmpq	$0x6, %rsi
   14b4d: 0f 82 45 fa ff ff            	jb	0x14598 <_Z3runN2at6TensorES0_S0_d+0x138>
   14b53: 4d 8b 36                     	movq	(%r14), %r14
   14b56: e9 3d fa ff ff               	jmp	0x14598 <_Z3runN2at6TensorES0_S0_d+0x138>
   14b5b: bf 03 00 00 00               	movl	$0x3, %edi
   14b60: 31 d2                        	xorl	%edx, %edx
   14b62: e8 d9 c3 01 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   14b67: 49 8b 76 40                  	movq	0x40(%r14), %rsi
   14b6b: 49 83 c6 48                  	addq	$0x48, %r14
   14b6f: 48 83 fe 06                  	cmpq	$0x6, %rsi
   14b73: 0f 82 5e fa ff ff            	jb	0x145d7 <_Z3runN2at6TensorES0_S0_d+0x177>
   14b79: 4d 8b 36                     	movq	(%r14), %r14
   14b7c: e9 56 fa ff ff               	jmp	0x145d7 <_Z3runN2at6TensorES0_S0_d+0x177>
   14b81: 31 ff                        	xorl	%edi, %edi
   14b83: 31 d2                        	xorl	%edx, %edx
   14b85: e8 b6 c3 01 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   14b8a: 49 8b 76 40                  	movq	0x40(%r14), %rsi
   14b8e: 49 83 c6 48                  	addq	$0x48, %r14
   14b92: 48 83 fe 06                  	cmpq	$0x6, %rsi
   14b96: 0f 82 73 fa ff ff            	jb	0x1460f <_Z3runN2at6TensorES0_S0_d+0x1af>
   14b9c: 4d 8b 36                     	movq	(%r14), %r14
   14b9f: e9 6b fa ff ff               	jmp	0x1460f <_Z3runN2at6TensorES0_S0_d+0x1af>
   14ba4: bf 01 00 00 00               	movl	$0x1, %edi
   14ba9: 31 d2                        	xorl	%edx, %edx
   14bab: e8 90 c3 01 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   14bb0: 49 8b 76 40                  	movq	0x40(%r14), %rsi
   14bb4: 49 83 c6 48                  	addq	$0x48, %r14
   14bb8: 48 83 fe 06                  	cmpq	$0x6, %rsi
   14bbc: 0f 82 94 fa ff ff            	jb	0x14656 <_Z3runN2at6TensorES0_S0_d+0x1f6>
   14bc2: 4d 8b 36                     	movq	(%r14), %r14
   14bc5: e9 8c fa ff ff               	jmp	0x14656 <_Z3runN2at6TensorES0_S0_d+0x1f6>
   14bca: bf 02 00 00 00               	movl	$0x2, %edi
   14bcf: 31 d2                        	xorl	%edx, %edx
   14bd1: e8 6a c3 01 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   14bd6: 49 8b 77 40                  	movq	0x40(%r15), %rsi
   14bda: 49 83 c7 48                  	addq	$0x48, %r15
   14bde: 48 83 fe 06                  	cmpq	$0x6, %rsi
   14be2: 0f 82 ad fa ff ff            	jb	0x14695 <_Z3runN2at6TensorES0_S0_d+0x235>
   14be8: 4d 8b 3f                     	movq	(%r15), %r15
   14beb: e9 a5 fa ff ff               	jmp	0x14695 <_Z3runN2at6TensorES0_S0_d+0x235>
   14bf0: bf 03 00 00 00               	movl	$0x3, %edi
   14bf5: 31 d2                        	xorl	%edx, %edx
   14bf7: e8 44 c3 01 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   14bfc: 49 8b 77 40                  	movq	0x40(%r15), %rsi
   14c00: 49 83 c7 48                  	addq	$0x48, %r15
   14c04: 48 83 fe 06                  	cmpq	$0x6, %rsi
   14c08: 0f 82 c6 fa ff ff            	jb	0x146d4 <_Z3runN2at6TensorES0_S0_d+0x274>
   14c0e: 4d 8b 3f                     	movq	(%r15), %r15
   14c11: e9 be fa ff ff               	jmp	0x146d4 <_Z3runN2at6TensorES0_S0_d+0x274>
   14c16: 31 ff                        	xorl	%edi, %edi
   14c18: 31 d2                        	xorl	%edx, %edx
   14c1a: e8 21 c3 01 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   14c1f: 49 8b 77 40                  	movq	0x40(%r15), %rsi
   14c23: e9 db fa ff ff               	jmp	0x14703 <_Z3runN2at6TensorES0_S0_d+0x2a3>
   14c28: 4d 8b 3f                     	movq	(%r15), %r15
   14c2b: e9 e7 fa ff ff               	jmp	0x14717 <_Z3runN2at6TensorES0_S0_d+0x2b7>
   14c30: bf 01 00 00 00               	movl	$0x1, %edi
   14c35: 31 d2                        	xorl	%edx, %edx
   14c37: e8 04 c3 01 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   14c3c: 49 8b 77 40                  	movq	0x40(%r15), %rsi
   14c40: 49 83 c7 48                  	addq	$0x48, %r15
   14c44: 48 83 fe 06                  	cmpq	$0x6, %rsi
   14c48: 0f 82 04 fb ff ff            	jb	0x14752 <_Z3runN2at6TensorES0_S0_d+0x2f2>
   14c4e: 4d 8b 3f                     	movq	(%r15), %r15
   14c51: 49 8b 2c c7                  	movq	(%r15,%rax,8), %rbp
   14c55: 4c 8b 3b                     	movq	(%rbx), %r15
   14c58: 41 f6 87 ae 00 00 00 08      	testb	$0x8, 0xae(%r15)
   14c60: 0f 84 01 fb ff ff            	je	0x14767 <_Z3runN2at6TensorES0_S0_d+0x307>
   14c66: e9 21 fe ff ff               	jmp	0x14a8c <_Z3runN2at6TensorES0_S0_d+0x62c>
   14c6b: bf 02 00 00 00               	movl	$0x2, %edi
   14c70: 31 d2                        	xorl	%edx, %edx
   14c72: e8 c9 c2 01 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   14c77: 49 8b 77 40                  	movq	0x40(%r15), %rsi
   14c7b: 49 83 c7 48                  	addq	$0x48, %r15
   14c7f: 48 83 fe 06                  	cmpq	$0x6, %rsi
   14c83: 0f 82 ff fa ff ff            	jb	0x14788 <_Z3runN2at6TensorES0_S0_d+0x328>
   14c89: 4d 8b 3f                     	movq	(%r15), %r15
   14c8c: 4d 8b 3c c7                  	movq	(%r15,%rax,8), %r15
   14c90: 4c 8b 23                     	movq	(%rbx), %r12
   14c93: 41 f6 84 24 ae 00 00 00 08   	testb	$0x8, 0xae(%r12)
   14c9c: 0f 84 fc fa ff ff            	je	0x1479e <_Z3runN2at6TensorES0_S0_d+0x33e>
   14ca2: e9 08 fe ff ff               	jmp	0x14aaf <_Z3runN2at6TensorES0_S0_d+0x64f>
   14ca7: bf 03 00 00 00               	movl	$0x3, %edi
   14cac: 31 d2                        	xorl	%edx, %edx
   14cae: e8 8d c2 01 00               	callq	0x30f40 <_ZN3c106detail19maybe_wrap_dim_slowIlEET_S2_S2_b@plt>
   14cb3: 49 8b 74 24 40               	movq	0x40(%r12), %rsi
   14cb8: 49 83 c4 48                  	addq	$0x48, %r12
   14cbc: 48 83 fe 06                  	cmpq	$0x6, %rsi
   14cc0: 0f 82 fa fa ff ff            	jb	0x147c0 <_Z3runN2at6TensorES0_S0_d+0x360>
   14cc6: 4d 8b 24 24                  	movq	(%r12), %r12
   14cca: e9 f1 fa ff ff               	jmp	0x147c0 <_Z3runN2at6TensorES0_S0_d+0x360>
   14ccf: 48 8b 07                     	movq	(%rdi), %rax
   14cd2: ff 50 68                     	callq	*0x68(%rax)
   14cd5: 3c 01                        	cmpb	$0x1, %al
   14cd7: 0f 85 f5 fb ff ff            	jne	0x148d2 <_Z3runN2at6TensorES0_S0_d+0x472>
   14cdd: e9 1c fc ff ff               	jmp	0x148fe <_Z3runN2at6TensorES0_S0_d+0x49e>
   14ce2: 48 8b 07                     	movq	(%rdi), %rax
   14ce5: ff 50 68                     	callq	*0x68(%rax)
   14ce8: 3c 01                        	cmpb	$0x1, %al
   14cea: 0f 85 36 fc ff ff            	jne	0x14926 <_Z3runN2at6TensorES0_S0_d+0x4c6>
   14cf0: 48 8b 44 24 40               	movq	0x40(%rsp), %rax
   14cf5: 48 8b 38                     	movq	(%rax), %rdi
   14cf8: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   14d00: 0f 89 4b fc ff ff            	jns	0x14951 <_Z3runN2at6TensorES0_S0_d+0x4f1>
   14d06: 48 8b 07                     	movq	(%rdi), %rax
   14d09: ff 50 68                     	callq	*0x68(%rax)
   14d0c: e9 df 09 00 00               	jmp	0x156f0 <_Z3runN2at6TensorES0_S0_d+0x1290>
   14d11: 48 8b 07                     	movq	(%rdi), %rax
   14d14: ff 50 68                     	callq	*0x68(%rax)
   14d17: 3c 01                        	cmpb	$0x1, %al
   14d19: 0f 84 df fb ff ff            	je	0x148fe <_Z3runN2at6TensorES0_S0_d+0x49e>
   14d1f: 66 c7 44 24 16 01 00         	movw	$0x1, 0x16(%rsp)
   14d26: 48 8d 7c 24 16               	leaq	0x16(%rsp), %rdi
   14d2b: e8 20 c2 01 00               	callq	0x30f50 <_ZN3c106Device8validateEv@plt>
   14d30: 0f b7 74 24 16               	movzwl	0x16(%rsp), %esi
   14d35: 31 ed                        	xorl	%ebp, %ebp
   14d37: eb 54                        	jmp	0x14d8d <_Z3runN2at6TensorES0_S0_d+0x92d>
   14d39: 48 8b 07                     	movq	(%rdi), %rax
   14d3c: ff 50 68                     	callq	*0x68(%rax)
   14d3f: 3c 01                        	cmpb	$0x1, %al
   14d41: 75 1c                        	jne	0x14d5f <_Z3runN2at6TensorES0_S0_d+0x8ff>
   14d43: 48 8b 3b                     	movq	(%rbx), %rdi
   14d46: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   14d4e: 0f 89 fd fb ff ff            	jns	0x14951 <_Z3runN2at6TensorES0_S0_d+0x4f1>
   14d54: 48 8b 07                     	movq	(%rdi), %rax
   14d57: ff 50 68                     	callq	*0x68(%rax)
   14d5a: e9 91 09 00 00               	jmp	0x156f0 <_Z3runN2at6TensorES0_S0_d+0x1290>
   14d5f: 49 8b 3c 24                  	movq	(%r12), %rdi
   14d63: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   14d6b: 0f 88 79 09 00 00            	js	0x156ea <_Z3runN2at6TensorES0_S0_d+0x128a>
   14d71: 80 bf ac 00 00 00 00         	cmpb	$0x0, 0xac(%rdi)
   14d78: 0f 84 e0 fb ff ff            	je	0x1495e <_Z3runN2at6TensorES0_S0_d+0x4fe>
   14d7e: 0f b7 b7 aa 00 00 00         	movzwl	0xaa(%rdi), %esi
   14d85: 66 89 74 24 16               	movw	%si, 0x16(%rsp)
   14d8a: 40 b5 01                     	movb	$0x1, %bpl
   14d8d: 48 8d bc 24 88 00 00 00      	leaq	0x88(%rsp), %rdi
   14d95: e8 c6 c1 01 00               	callq	0x30f60 <_ZN3c104impl17InlineDeviceGuardINS0_16VirtualGuardImplEEC2ENS_6DeviceE@plt>
   14d9a: c6 84 24 a0 00 00 00 01      	movb	$0x1, 0xa0(%rsp)
   14da2: 48 8b 44 24 40               	movq	0x40(%rsp), %rax
   14da7: 48 8b 38                     	movq	(%rax), %rdi
   14daa: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   14db2: 0f 88 1c 01 00 00            	js	0x14ed4 <_Z3runN2at6TensorES0_S0_d+0xa74>
   14db8: 80 bf ac 00 00 00 01         	cmpb	$0x1, 0xac(%rdi)
   14dbf: 0f 85 3a 01 00 00            	jne	0x14eff <_Z3runN2at6TensorES0_S0_d+0xa9f>
   14dc5: 80 bf aa 00 00 00 01         	cmpb	$0x1, 0xaa(%rdi)
   14dcc: 0f 85 2d 01 00 00            	jne	0x14eff <_Z3runN2at6TensorES0_S0_d+0xa9f>
   14dd2: 80 bf ac 00 00 00 00         	cmpb	$0x0, 0xac(%rdi)
   14dd9: 0f 84 05 0a 00 00            	je	0x157e4 <_Z3runN2at6TensorES0_S0_d+0x1384>
   14ddf: 0f b7 87 aa 00 00 00         	movzwl	0xaa(%rdi), %eax
   14de6: 38 44 24 16                  	cmpb	%al, 0x16(%rsp)
   14dea: 0f 85 0f 01 00 00            	jne	0x14eff <_Z3runN2at6TensorES0_S0_d+0xa9f>
   14df0: c1 e8 08                     	shrl	$0x8, %eax
   14df3: 38 44 24 17                  	cmpb	%al, 0x17(%rsp)
   14df7: 0f 85 02 01 00 00            	jne	0x14eff <_Z3runN2at6TensorES0_S0_d+0xa9f>
   14dfd: 48 8b 44 24 40               	movq	0x40(%rsp), %rax
   14e02: 48 8b 38                     	movq	(%rax), %rdi
   14e05: 0f b7 87 a8 00 00 00         	movzwl	0xa8(%rdi), %eax
   14e0c: 66 83 f8 2f                  	cmpw	$0x2f, %ax
   14e10: 0f 83 e1 08 00 00            	jae	0x156f7 <_Z3runN2at6TensorES0_S0_d+0x1297>
   14e16: 66 83 f8 0f                  	cmpw	$0xf, %ax
   14e1a: 0f 85 df 00 00 00            	jne	0x14eff <_Z3runN2at6TensorES0_S0_d+0xa9f>
   14e20: 0f b7 87 ad 00 00 00         	movzwl	0xad(%rdi), %eax
   14e27: a9 00 0c 00 00               	testl	$0xc00, %eax            # imm = 0xC00
   14e2c: 0f 85 da 08 00 00            	jne	0x1570c <_Z3runN2at6TensorES0_S0_d+0x12ac>
   14e32: a9 00 10 00 00               	testl	$0x1000, %eax           # imm = 0x1000
   14e37: 75 09                        	jne	0x14e42 <_Z3runN2at6TensorES0_S0_d+0x9e2>
   14e39: a8 01                        	testb	$0x1, %al
   14e3b: 75 4d                        	jne	0x14e8a <_Z3runN2at6TensorES0_S0_d+0xa2a>
   14e3d: e9 bd 00 00 00               	jmp	0x14eff <_Z3runN2at6TensorES0_S0_d+0xa9f>
   14e42: 48 8b 47 20                  	movq	0x20(%rdi), %rax
   14e46: 48 85 c0                     	testq	%rax, %rax
   14e49: 0f 84 f2 09 00 00            	je	0x15841 <_Z3runN2at6TensorES0_S0_d+0x13e1>
   14e4f: 48 8b 38                     	movq	(%rax), %rdi
   14e52: 48 85 ff                     	testq	%rdi, %rdi
   14e55: 0f 84 e6 09 00 00            	je	0x15841 <_Z3runN2at6TensorES0_S0_d+0x13e1>
   14e5b: 8b 47 7c                     	movl	0x7c(%rdi), %eax
   14e5e: a8 02                        	testb	$0x2, %al
   14e60: 0f 84 ca 08 00 00            	je	0x15730 <_Z3runN2at6TensorES0_S0_d+0x12d0>
   14e66: 48 81 c7 b0 00 00 00         	addq	$0xb0, %rdi
   14e6d: 48 8d 35 1d 80 ff ff         	leaq	-0x7fe3(%rip), %rsi     # 0xce91 <strncmp+0xce91>
   14e74: ba 42 03 00 00               	movl	$0x342, %edx            # imm = 0x342
   14e79: e8 f2 c0 01 00               	callq	0x30f70 <_ZNK3c107SymBool10guard_boolEPKcl@plt>
   14e7e: 84 c0                        	testb	%al, %al
   14e80: 74 7d                        	je	0x14eff <_Z3runN2at6TensorES0_S0_d+0xa9f>
   14e82: 48 8b 44 24 40               	movq	0x40(%rsp), %rax
   14e87: 48 8b 38                     	movq	(%rax), %rdi
   14e8a: 48 89 7c 24 28               	movq	%rdi, 0x28(%rsp)
   14e8f: 48 3b 3d 7a e4 01 00         	cmpq	0x1e47a(%rip), %rdi     # 0x33310 <strncmp+0x33310>
   14e96: 0f 84 54 01 00 00            	je	0x14ff0 <_Z3runN2at6TensorES0_S0_d+0xb90>
   14e9c: b8 01 00 00 00               	movl	$0x1, %eax
   14ea1: f0                           	lock
   14ea2: 0f c1 47 08                  	xaddl	%eax, 0x8(%rdi)
   14ea6: 85 c0                        	testl	%eax, %eax
   14ea8: 0f 85 42 01 00 00            	jne	0x14ff0 <_Z3runN2at6TensorES0_S0_d+0xb90>
   14eae: 48 8d 3d 0b 8f ff ff         	leaq	-0x70f5(%rip), %rdi     # 0xddc0 <strncmp+0xddc0>
   14eb5: 48 8d 35 9a 8d ff ff         	leaq	-0x7266(%rip), %rsi     # 0xdc56 <strncmp+0xdc56>
   14ebc: 48 8d 0d 83 70 ff ff         	leaq	-0x8f7d(%rip), %rcx     # 0xbf46 <strncmp+0xbf46>
   14ec3: 4c 8d 05 c2 88 ff ff         	leaq	-0x773e(%rip), %r8      # 0xd78c <strncmp+0xd78c>
   14eca: ba 14 01 00 00               	movl	$0x114, %edx            # imm = 0x114
   14ecf: e8 ac c0 01 00               	callq	0x30f80 <_ZN3c106detail23torchInternalAssertFailEPKcS2_jS2_S2_@plt>
   14ed4: 48 8b 07                     	movq	(%rdi), %rax
   14ed7: ff 50 68                     	callq	*0x68(%rax)
   14eda: 3c 01                        	cmpb	$0x1, %al
   14edc: 75 21                        	jne	0x14eff <_Z3runN2at6TensorES0_S0_d+0xa9f>
   14ede: 48 8b 44 24 40               	movq	0x40(%rsp), %rax
   14ee3: 48 8b 38                     	movq	(%rax), %rdi
   14ee6: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   14eee: 0f 89 de fe ff ff            	jns	0x14dd2 <_Z3runN2at6TensorES0_S0_d+0x972>
   14ef4: 48 8b 07                     	movq	(%rdi), %rax
   14ef7: ff 50 68                     	callq	*0x68(%rax)
   14efa: e9 e7 fe ff ff               	jmp	0x14de6 <_Z3runN2at6TensorES0_S0_d+0x986>
   14eff: 0f b7 54 24 16               	movzwl	0x16(%rsp), %edx
   14f04: c7 04 24 00 00 00 00         	movl	$0x0, (%rsp)
   14f0b: 48 8d 7c 24 18               	leaq	0x18(%rsp), %rdi
   14f10: 48 8b 74 24 40               	movq	0x40(%rsp), %rsi
   14f15: b9 0f 00 00 00               	movl	$0xf, %ecx
   14f1a: 45 31 c0                     	xorl	%r8d, %r8d
   14f1d: 45 31 c9                     	xorl	%r9d, %r9d
   14f20: e8 6b c0 01 00               	callq	0x30f90 <_ZN2at4_ops9to_device4callERKNS_6TensorEN3c106DeviceENS5_10ScalarTypeEbbSt8optionalINS5_12MemoryFormatEE@plt>
   14f25: 48 8d 7c 24 18               	leaq	0x18(%rsp), %rdi
   14f2a: 31 f6                        	xorl	%esi, %esi
   14f2c: e8 6f c0 01 00               	callq	0x30fa0 <_ZNK2at10TensorBase22is_contiguous_or_falseEN3c1012MemoryFormatE@plt>
   14f31: 84 c0                        	testb	%al, %al
   14f33: 74 4b                        	je	0x14f80 <_Z3runN2at6TensorES0_S0_d+0xb20>
   14f35: 48 8b 44 24 18               	movq	0x18(%rsp), %rax
   14f3a: 48 89 44 24 50               	movq	%rax, 0x50(%rsp)
   14f3f: 48 3b 05 ca e3 01 00         	cmpq	0x1e3ca(%rip), %rax     # 0x33310 <strncmp+0x33310>
   14f46: 0f 84 98 00 00 00            	je	0x14fe4 <_Z3runN2at6TensorES0_S0_d+0xb84>
   14f4c: b9 01 00 00 00               	movl	$0x1, %ecx
   14f51: f0                           	lock
   14f52: 0f c1 48 08                  	xaddl	%ecx, 0x8(%rax)
   14f56: 85 c9                        	testl	%ecx, %ecx
   14f58: 75 37                        	jne	0x14f91 <_Z3runN2at6TensorES0_S0_d+0xb31>
   14f5a: 48 8d 3d 5f 8e ff ff         	leaq	-0x71a1(%rip), %rdi     # 0xddc0 <strncmp+0xddc0>
   14f61: 48 8d 35 ee 8c ff ff         	leaq	-0x7312(%rip), %rsi     # 0xdc56 <strncmp+0xdc56>
   14f68: 48 8d 0d d7 6f ff ff         	leaq	-0x9029(%rip), %rcx     # 0xbf46 <strncmp+0xbf46>
   14f6f: 4c 8d 05 16 88 ff ff         	leaq	-0x77ea(%rip), %r8      # 0xd78c <strncmp+0xd78c>
   14f76: ba 14 01 00 00               	movl	$0x114, %edx            # imm = 0x114
   14f7b: e8 00 c0 01 00               	callq	0x30f80 <_ZN3c106detail23torchInternalAssertFailEPKcS2_jS2_S2_@plt>
   14f80: 48 8d 7c 24 50               	leaq	0x50(%rsp), %rdi
   14f85: 48 8d 74 24 18               	leaq	0x18(%rsp), %rsi
   14f8a: 31 d2                        	xorl	%edx, %edx
   14f8c: e8 1f c0 01 00               	callq	0x30fb0 <_ZNK2at10TensorBase21__dispatch_contiguousEN3c1012MemoryFormatE@plt>
   14f91: 48 8b 44 24 18               	movq	0x18(%rsp), %rax
   14f96: 48 8b 4c 24 50               	movq	0x50(%rsp), %rcx
   14f9b: 48 89 4c 24 28               	movq	%rcx, 0x28(%rsp)
   14fa0: 48 3b 05 69 e3 01 00         	cmpq	0x1e369(%rip), %rax     # 0x33310 <strncmp+0x33310>
   14fa7: 74 47                        	je	0x14ff0 <_Z3runN2at6TensorES0_S0_d+0xb90>
   14fa9: f0                           	lock
   14faa: ff 48 08                     	decl	0x8(%rax)
   14fad: 75 41                        	jne	0x14ff0 <_Z3runN2at6TensorES0_S0_d+0xb90>
   14faf: 48 8b 44 24 18               	movq	0x18(%rsp), %rax
   14fb4: 8b 40 0c                     	movl	0xc(%rax), %eax
   14fb7: 83 f8 01                     	cmpl	$0x1, %eax
   14fba: 74 16                        	je	0x14fd2 <_Z3runN2at6TensorES0_S0_d+0xb72>
   14fbc: 48 8b 7c 24 18               	movq	0x18(%rsp), %rdi
   14fc1: 48 8b 07                     	movq	(%rdi), %rax
   14fc4: ff 50 10                     	callq	*0x10(%rax)
   14fc7: 48 8b 44 24 18               	movq	0x18(%rsp), %rax
   14fcc: f0                           	lock
   14fcd: ff 48 0c                     	decl	0xc(%rax)
   14fd0: 75 1e                        	jne	0x14ff0 <_Z3runN2at6TensorES0_S0_d+0xb90>
   14fd2: 48 8b 7c 24 18               	movq	0x18(%rsp), %rdi
   14fd7: 48 85 ff                     	testq	%rdi, %rdi
   14fda: 74 14                        	je	0x14ff0 <_Z3runN2at6TensorES0_S0_d+0xb90>
   14fdc: 48 8b 07                     	movq	(%rdi), %rax
   14fdf: ff 50 08                     	callq	*0x8(%rax)
   14fe2: eb 0c                        	jmp	0x14ff0 <_Z3runN2at6TensorES0_S0_d+0xb90>
   14fe4: 48 8b 05 25 e3 01 00         	movq	0x1e325(%rip), %rax     # 0x33310 <strncmp+0x33310>
   14feb: 48 89 44 24 28               	movq	%rax, 0x28(%rsp)
   14ff0: 48 8b 44 24 48               	movq	0x48(%rsp), %rax
   14ff5: 48 8b 38                     	movq	(%rax), %rdi
   14ff8: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   15000: 0f 88 1c 01 00 00            	js	0x15122 <_Z3runN2at6TensorES0_S0_d+0xcc2>
   15006: 80 bf ac 00 00 00 01         	cmpb	$0x1, 0xac(%rdi)
   1500d: 0f 85 3a 01 00 00            	jne	0x1514d <_Z3runN2at6TensorES0_S0_d+0xced>
   15013: 80 bf aa 00 00 00 01         	cmpb	$0x1, 0xaa(%rdi)
   1501a: 0f 85 2d 01 00 00            	jne	0x1514d <_Z3runN2at6TensorES0_S0_d+0xced>
   15020: 80 bf ac 00 00 00 00         	cmpb	$0x0, 0xac(%rdi)
   15027: 0f 84 d6 07 00 00            	je	0x15803 <_Z3runN2at6TensorES0_S0_d+0x13a3>
   1502d: 0f b7 87 aa 00 00 00         	movzwl	0xaa(%rdi), %eax
   15034: 38 44 24 16                  	cmpb	%al, 0x16(%rsp)
   15038: 0f 85 0f 01 00 00            	jne	0x1514d <_Z3runN2at6TensorES0_S0_d+0xced>
   1503e: c1 e8 08                     	shrl	$0x8, %eax
   15041: 38 44 24 17                  	cmpb	%al, 0x17(%rsp)
   15045: 0f 85 02 01 00 00            	jne	0x1514d <_Z3runN2at6TensorES0_S0_d+0xced>
   1504b: 48 8b 44 24 48               	movq	0x48(%rsp), %rax
   15050: 48 8b 38                     	movq	(%rax), %rdi
   15053: 0f b7 87 a8 00 00 00         	movzwl	0xa8(%rdi), %eax
   1505a: 66 83 f8 2f                  	cmpw	$0x2f, %ax
   1505e: 0f 83 9a 06 00 00            	jae	0x156fe <_Z3runN2at6TensorES0_S0_d+0x129e>
   15064: 66 83 f8 0f                  	cmpw	$0xf, %ax
   15068: 0f 85 df 00 00 00            	jne	0x1514d <_Z3runN2at6TensorES0_S0_d+0xced>
   1506e: 0f b7 87 ad 00 00 00         	movzwl	0xad(%rdi), %eax
   15075: a9 00 0c 00 00               	testl	$0xc00, %eax            # imm = 0xC00
   1507a: 0f 85 98 06 00 00            	jne	0x15718 <_Z3runN2at6TensorES0_S0_d+0x12b8>
   15080: a9 00 10 00 00               	testl	$0x1000, %eax           # imm = 0x1000
   15085: 75 09                        	jne	0x15090 <_Z3runN2at6TensorES0_S0_d+0xc30>
   15087: a8 01                        	testb	$0x1, %al
   15089: 75 4d                        	jne	0x150d8 <_Z3runN2at6TensorES0_S0_d+0xc78>
   1508b: e9 bd 00 00 00               	jmp	0x1514d <_Z3runN2at6TensorES0_S0_d+0xced>
   15090: 48 8b 47 20                  	movq	0x20(%rdi), %rax
   15094: 48 85 c0                     	testq	%rax, %rax
   15097: 0f 84 c3 07 00 00            	je	0x15860 <_Z3runN2at6TensorES0_S0_d+0x1400>
   1509d: 48 8b 38                     	movq	(%rax), %rdi
   150a0: 48 85 ff                     	testq	%rdi, %rdi
   150a3: 0f 84 b7 07 00 00            	je	0x15860 <_Z3runN2at6TensorES0_S0_d+0x1400>
   150a9: 8b 47 7c                     	movl	0x7c(%rdi), %eax
   150ac: a8 02                        	testb	$0x2, %al
   150ae: 0f 84 8c 06 00 00            	je	0x15740 <_Z3runN2at6TensorES0_S0_d+0x12e0>
   150b4: 48 81 c7 b0 00 00 00         	addq	$0xb0, %rdi
   150bb: 48 8d 35 cf 7d ff ff         	leaq	-0x8231(%rip), %rsi     # 0xce91 <strncmp+0xce91>
   150c2: ba 42 03 00 00               	movl	$0x342, %edx            # imm = 0x342
   150c7: e8 a4 be 01 00               	callq	0x30f70 <_ZNK3c107SymBool10guard_boolEPKcl@plt>
   150cc: 84 c0                        	testb	%al, %al
   150ce: 74 7d                        	je	0x1514d <_Z3runN2at6TensorES0_S0_d+0xced>
   150d0: 48 8b 44 24 48               	movq	0x48(%rsp), %rax
   150d5: 48 8b 38                     	movq	(%rax), %rdi
   150d8: 48 89 7c 24 38               	movq	%rdi, 0x38(%rsp)
   150dd: 48 3b 3d 2c e2 01 00         	cmpq	0x1e22c(%rip), %rdi     # 0x33310 <strncmp+0x33310>
   150e4: 0f 84 54 01 00 00            	je	0x1523e <_Z3runN2at6TensorES0_S0_d+0xdde>
   150ea: b8 01 00 00 00               	movl	$0x1, %eax
   150ef: f0                           	lock
   150f0: 0f c1 47 08                  	xaddl	%eax, 0x8(%rdi)
   150f4: 85 c0                        	testl	%eax, %eax
   150f6: 0f 85 42 01 00 00            	jne	0x1523e <_Z3runN2at6TensorES0_S0_d+0xdde>
   150fc: 48 8d 3d bd 8c ff ff         	leaq	-0x7343(%rip), %rdi     # 0xddc0 <strncmp+0xddc0>
   15103: 48 8d 35 4c 8b ff ff         	leaq	-0x74b4(%rip), %rsi     # 0xdc56 <strncmp+0xdc56>
   1510a: 48 8d 0d 35 6e ff ff         	leaq	-0x91cb(%rip), %rcx     # 0xbf46 <strncmp+0xbf46>
   15111: 4c 8d 05 74 86 ff ff         	leaq	-0x798c(%rip), %r8      # 0xd78c <strncmp+0xd78c>
   15118: ba 14 01 00 00               	movl	$0x114, %edx            # imm = 0x114
   1511d: e8 5e be 01 00               	callq	0x30f80 <_ZN3c106detail23torchInternalAssertFailEPKcS2_jS2_S2_@plt>
   15122: 48 8b 07                     	movq	(%rdi), %rax
   15125: ff 50 68                     	callq	*0x68(%rax)
   15128: 3c 01                        	cmpb	$0x1, %al
   1512a: 75 21                        	jne	0x1514d <_Z3runN2at6TensorES0_S0_d+0xced>
   1512c: 48 8b 44 24 48               	movq	0x48(%rsp), %rax
   15131: 48 8b 38                     	movq	(%rax), %rdi
   15134: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   1513c: 0f 89 de fe ff ff            	jns	0x15020 <_Z3runN2at6TensorES0_S0_d+0xbc0>
   15142: 48 8b 07                     	movq	(%rdi), %rax
   15145: ff 50 68                     	callq	*0x68(%rax)
   15148: e9 e7 fe ff ff               	jmp	0x15034 <_Z3runN2at6TensorES0_S0_d+0xbd4>
   1514d: 0f b7 54 24 16               	movzwl	0x16(%rsp), %edx
   15152: c7 04 24 00 00 00 00         	movl	$0x0, (%rsp)
   15159: 48 8d 7c 24 18               	leaq	0x18(%rsp), %rdi
   1515e: 48 8b 74 24 48               	movq	0x48(%rsp), %rsi
   15163: b9 0f 00 00 00               	movl	$0xf, %ecx
   15168: 45 31 c0                     	xorl	%r8d, %r8d
   1516b: 45 31 c9                     	xorl	%r9d, %r9d
   1516e: e8 1d be 01 00               	callq	0x30f90 <_ZN2at4_ops9to_device4callERKNS_6TensorEN3c106DeviceENS5_10ScalarTypeEbbSt8optionalINS5_12MemoryFormatEE@plt>
   15173: 48 8d 7c 24 18               	leaq	0x18(%rsp), %rdi
   15178: 31 f6                        	xorl	%esi, %esi
   1517a: e8 21 be 01 00               	callq	0x30fa0 <_ZNK2at10TensorBase22is_contiguous_or_falseEN3c1012MemoryFormatE@plt>
   1517f: 84 c0                        	testb	%al, %al
   15181: 74 4b                        	je	0x151ce <_Z3runN2at6TensorES0_S0_d+0xd6e>
   15183: 48 8b 44 24 18               	movq	0x18(%rsp), %rax
   15188: 48 89 44 24 50               	movq	%rax, 0x50(%rsp)
   1518d: 48 3b 05 7c e1 01 00         	cmpq	0x1e17c(%rip), %rax     # 0x33310 <strncmp+0x33310>
   15194: 0f 84 98 00 00 00            	je	0x15232 <_Z3runN2at6TensorES0_S0_d+0xdd2>
   1519a: b9 01 00 00 00               	movl	$0x1, %ecx
   1519f: f0                           	lock
   151a0: 0f c1 48 08                  	xaddl	%ecx, 0x8(%rax)
   151a4: 85 c9                        	testl	%ecx, %ecx
   151a6: 75 37                        	jne	0x151df <_Z3runN2at6TensorES0_S0_d+0xd7f>
   151a8: 48 8d 3d 11 8c ff ff         	leaq	-0x73ef(%rip), %rdi     # 0xddc0 <strncmp+0xddc0>
   151af: 48 8d 35 a0 8a ff ff         	leaq	-0x7560(%rip), %rsi     # 0xdc56 <strncmp+0xdc56>
   151b6: 48 8d 0d 89 6d ff ff         	leaq	-0x9277(%rip), %rcx     # 0xbf46 <strncmp+0xbf46>
   151bd: 4c 8d 05 c8 85 ff ff         	leaq	-0x7a38(%rip), %r8      # 0xd78c <strncmp+0xd78c>
   151c4: ba 14 01 00 00               	movl	$0x114, %edx            # imm = 0x114
   151c9: e8 b2 bd 01 00               	callq	0x30f80 <_ZN3c106detail23torchInternalAssertFailEPKcS2_jS2_S2_@plt>
   151ce: 48 8d 7c 24 50               	leaq	0x50(%rsp), %rdi
   151d3: 48 8d 74 24 18               	leaq	0x18(%rsp), %rsi
   151d8: 31 d2                        	xorl	%edx, %edx
   151da: e8 d1 bd 01 00               	callq	0x30fb0 <_ZNK2at10TensorBase21__dispatch_contiguousEN3c1012MemoryFormatE@plt>
   151df: 48 8b 44 24 18               	movq	0x18(%rsp), %rax
   151e4: 48 8b 4c 24 50               	movq	0x50(%rsp), %rcx
   151e9: 48 89 4c 24 38               	movq	%rcx, 0x38(%rsp)
   151ee: 48 3b 05 1b e1 01 00         	cmpq	0x1e11b(%rip), %rax     # 0x33310 <strncmp+0x33310>
   151f5: 74 47                        	je	0x1523e <_Z3runN2at6TensorES0_S0_d+0xdde>
   151f7: f0                           	lock
   151f8: ff 48 08                     	decl	0x8(%rax)
   151fb: 75 41                        	jne	0x1523e <_Z3runN2at6TensorES0_S0_d+0xdde>
   151fd: 48 8b 44 24 18               	movq	0x18(%rsp), %rax
   15202: 8b 40 0c                     	movl	0xc(%rax), %eax
   15205: 83 f8 01                     	cmpl	$0x1, %eax
   15208: 74 16                        	je	0x15220 <_Z3runN2at6TensorES0_S0_d+0xdc0>
   1520a: 48 8b 7c 24 18               	movq	0x18(%rsp), %rdi
   1520f: 48 8b 07                     	movq	(%rdi), %rax
   15212: ff 50 10                     	callq	*0x10(%rax)
   15215: 48 8b 44 24 18               	movq	0x18(%rsp), %rax
   1521a: f0                           	lock
   1521b: ff 48 0c                     	decl	0xc(%rax)
   1521e: 75 1e                        	jne	0x1523e <_Z3runN2at6TensorES0_S0_d+0xdde>
   15220: 48 8b 7c 24 18               	movq	0x18(%rsp), %rdi
   15225: 48 85 ff                     	testq	%rdi, %rdi
   15228: 74 14                        	je	0x1523e <_Z3runN2at6TensorES0_S0_d+0xdde>
   1522a: 48 8b 07                     	movq	(%rdi), %rax
   1522d: ff 50 08                     	callq	*0x8(%rax)
   15230: eb 0c                        	jmp	0x1523e <_Z3runN2at6TensorES0_S0_d+0xdde>
   15232: 48 8b 05 d7 e0 01 00         	movq	0x1e0d7(%rip), %rax     # 0x33310 <strncmp+0x33310>
   15239: 48 89 44 24 38               	movq	%rax, 0x38(%rsp)
   1523e: 49 8b 3c 24                  	movq	(%r12), %rdi
   15242: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   1524a: 0f 88 14 01 00 00            	js	0x15364 <_Z3runN2at6TensorES0_S0_d+0xf04>
   15250: 80 bf ac 00 00 00 01         	cmpb	$0x1, 0xac(%rdi)
   15257: 0f 85 2e 01 00 00            	jne	0x1538b <_Z3runN2at6TensorES0_S0_d+0xf2b>
   1525d: 80 bf aa 00 00 00 01         	cmpb	$0x1, 0xaa(%rdi)
   15264: 0f 85 21 01 00 00            	jne	0x1538b <_Z3runN2at6TensorES0_S0_d+0xf2b>
   1526a: 80 bf ac 00 00 00 00         	cmpb	$0x0, 0xac(%rdi)
   15271: 0f 84 ab 05 00 00            	je	0x15822 <_Z3runN2at6TensorES0_S0_d+0x13c2>
   15277: 0f b7 87 aa 00 00 00         	movzwl	0xaa(%rdi), %eax
   1527e: 38 44 24 16                  	cmpb	%al, 0x16(%rsp)
   15282: 0f 85 03 01 00 00            	jne	0x1538b <_Z3runN2at6TensorES0_S0_d+0xf2b>
   15288: c1 e8 08                     	shrl	$0x8, %eax
   1528b: 38 44 24 17                  	cmpb	%al, 0x17(%rsp)
   1528f: 0f 85 f6 00 00 00            	jne	0x1538b <_Z3runN2at6TensorES0_S0_d+0xf2b>
   15295: 49 8b 3c 24                  	movq	(%r12), %rdi
   15299: 0f b7 87 a8 00 00 00         	movzwl	0xa8(%rdi), %eax
   152a0: 66 83 f8 2f                  	cmpw	$0x2f, %ax
   152a4: 0f 83 5b 04 00 00            	jae	0x15705 <_Z3runN2at6TensorES0_S0_d+0x12a5>
   152aa: 66 83 f8 0f                  	cmpw	$0xf, %ax
   152ae: 0f 85 d7 00 00 00            	jne	0x1538b <_Z3runN2at6TensorES0_S0_d+0xf2b>
   152b4: 0f b7 87 ad 00 00 00         	movzwl	0xad(%rdi), %eax
   152bb: a9 00 0c 00 00               	testl	$0xc00, %eax            # imm = 0xC00
   152c0: 0f 85 5e 04 00 00            	jne	0x15724 <_Z3runN2at6TensorES0_S0_d+0x12c4>
   152c6: a9 00 10 00 00               	testl	$0x1000, %eax           # imm = 0x1000
   152cb: 75 09                        	jne	0x152d6 <_Z3runN2at6TensorES0_S0_d+0xe76>
   152cd: a8 01                        	testb	$0x1, %al
   152cf: 75 49                        	jne	0x1531a <_Z3runN2at6TensorES0_S0_d+0xeba>
   152d1: e9 b5 00 00 00               	jmp	0x1538b <_Z3runN2at6TensorES0_S0_d+0xf2b>
   152d6: 48 8b 47 20                  	movq	0x20(%rdi), %rax
   152da: 48 85 c0                     	testq	%rax, %rax
   152dd: 0f 84 9c 05 00 00            	je	0x1587f <_Z3runN2at6TensorES0_S0_d+0x141f>
   152e3: 48 8b 38                     	movq	(%rax), %rdi
   152e6: 48 85 ff                     	testq	%rdi, %rdi
   152e9: 0f 84 90 05 00 00            	je	0x1587f <_Z3runN2at6TensorES0_S0_d+0x141f>
   152ef: 8b 47 7c                     	movl	0x7c(%rdi), %eax
   152f2: a8 02                        	testb	$0x2, %al
   152f4: 0f 84 56 04 00 00            	je	0x15750 <_Z3runN2at6TensorES0_S0_d+0x12f0>
   152fa: 48 81 c7 b0 00 00 00         	addq	$0xb0, %rdi
   15301: 48 8d 35 89 7b ff ff         	leaq	-0x8477(%rip), %rsi     # 0xce91 <strncmp+0xce91>
   15308: ba 42 03 00 00               	movl	$0x342, %edx            # imm = 0x342
   1530d: e8 5e bc 01 00               	callq	0x30f70 <_ZNK3c107SymBool10guard_boolEPKcl@plt>
   15312: 84 c0                        	testb	%al, %al
   15314: 74 75                        	je	0x1538b <_Z3runN2at6TensorES0_S0_d+0xf2b>
   15316: 49 8b 3c 24                  	movq	(%r12), %rdi
   1531a: 48 89 7c 24 30               	movq	%rdi, 0x30(%rsp)
   1531f: 48 3b 3d ea df 01 00         	cmpq	0x1dfea(%rip), %rdi     # 0x33310 <strncmp+0x33310>
   15326: 0f 84 4e 01 00 00            	je	0x1547a <_Z3runN2at6TensorES0_S0_d+0x101a>
   1532c: b8 01 00 00 00               	movl	$0x1, %eax
   15331: f0                           	lock
   15332: 0f c1 47 08                  	xaddl	%eax, 0x8(%rdi)
   15336: 85 c0                        	testl	%eax, %eax
   15338: 0f 85 3c 01 00 00            	jne	0x1547a <_Z3runN2at6TensorES0_S0_d+0x101a>
   1533e: 48 8d 3d 7b 8a ff ff         	leaq	-0x7585(%rip), %rdi     # 0xddc0 <strncmp+0xddc0>
   15345: 48 8d 35 0a 89 ff ff         	leaq	-0x76f6(%rip), %rsi     # 0xdc56 <strncmp+0xdc56>
   1534c: 48 8d 0d f3 6b ff ff         	leaq	-0x940d(%rip), %rcx     # 0xbf46 <strncmp+0xbf46>
   15353: 4c 8d 05 32 84 ff ff         	leaq	-0x7bce(%rip), %r8      # 0xd78c <strncmp+0xd78c>
   1535a: ba 14 01 00 00               	movl	$0x114, %edx            # imm = 0x114
   1535f: e8 1c bc 01 00               	callq	0x30f80 <_ZN3c106detail23torchInternalAssertFailEPKcS2_jS2_S2_@plt>
   15364: 48 8b 07                     	movq	(%rdi), %rax
   15367: ff 50 68                     	callq	*0x68(%rax)
   1536a: 3c 01                        	cmpb	$0x1, %al
   1536c: 75 1d                        	jne	0x1538b <_Z3runN2at6TensorES0_S0_d+0xf2b>
   1536e: 49 8b 3c 24                  	movq	(%r12), %rdi
   15372: 66 83 bf ad 00 00 00 00      	cmpw	$0x0, 0xad(%rdi)
   1537a: 0f 89 ea fe ff ff            	jns	0x1526a <_Z3runN2at6TensorES0_S0_d+0xe0a>
   15380: 48 8b 07                     	movq	(%rdi), %rax
   15383: ff 50 68                     	callq	*0x68(%rax)
   15386: e9 f3 fe ff ff               	jmp	0x1527e <_Z3runN2at6TensorES0_S0_d+0xe1e>
   1538b: 0f b7 54 24 16               	movzwl	0x16(%rsp), %edx
   15390: c7 04 24 00 00 00 00         	movl	$0x0, (%rsp)
   15397: 48 8d 7c 24 18               	leaq	0x18(%rsp), %rdi
   1539c: 4c 89 e6                     	movq	%r12, %rsi
   1539f: b9 0f 00 00 00               	movl	$0xf, %ecx
   153a4: 45 31 c0                     	xorl	%r8d, %r8d
   153a7: 45 31 c9                     	xorl	%r9d, %r9d
   153aa: e8 e1 bb 01 00               	callq	0x30f90 <_ZN2at4_ops9to_device4callERKNS_6TensorEN3c106DeviceENS5_10ScalarTypeEbbSt8optionalINS5_12MemoryFormatEE@plt>
   153af: 48 8d 7c 24 18               	leaq	0x18(%rsp), %rdi
   153b4: 31 f6                        	xorl	%esi, %esi
   153b6: e8 e5 bb 01 00               	callq	0x30fa0 <_ZNK2at10TensorBase22is_contiguous_or_falseEN3c1012MemoryFormatE@plt>
   153bb: 84 c0                        	testb	%al, %al
   153bd: 74 4b                        	je	0x1540a <_Z3runN2at6TensorES0_S0_d+0xfaa>
   153bf: 48 8b 44 24 18               	movq	0x18(%rsp), %rax
   153c4: 48 89 44 24 50               	movq	%rax, 0x50(%rsp)
   153c9: 48 3b 05 40 df 01 00         	cmpq	0x1df40(%rip), %rax     # 0x33310 <strncmp+0x33310>
   153d0: 0f 84 98 00 00 00            	je	0x1546e <_Z3runN2at6TensorES0_S0_d+0x100e>
   153d6: b9 01 00 00 00               	movl	$0x1, %ecx
   153db: f0                           	lock
   153dc: 0f c1 48 08                  	xaddl	%ecx, 0x8(%rax)
   153e0: 85 c9                        	testl	%ecx, %ecx
   153e2: 75 37                        	jne	0x1541b <_Z3runN2at6TensorES0_S0_d+0xfbb>
   153e4: 48 8d 3d d5 89 ff ff         	leaq	-0x762b(%rip), %rdi     # 0xddc0 <strncmp+0xddc0>
   153eb: 48 8d 35 64 88 ff ff         	leaq	-0x779c(%rip), %rsi     # 0xdc56 <strncmp+0xdc56>
   153f2: 48 8d 0d 4d 6b ff ff         	leaq	-0x94b3(%rip), %rcx     # 0xbf46 <strncmp+0xbf46>
   153f9: 4c 8d 05 8c 83 ff ff         	leaq	-0x7c74(%rip), %r8      # 0xd78c <strncmp+0xd78c>
   15400: ba 14 01 00 00               	movl	$0x114, %edx            # imm = 0x114
   15405: e8 76 bb 01 00               	callq	0x30f80 <_ZN3c106detail23torchInternalAssertFailEPKcS2_jS2_S2_@plt>
   1540a: 48 8d 7c 24 50               	leaq	0x50(%rsp), %rdi
   1540f: 48 8d 74 24 18               	leaq	0x18(%rsp), %rsi
   15414: 31 d2                        	xorl	%edx, %edx
   15416: e8 95 bb 01 00               	callq	0x30fb0 <_ZNK2at10TensorBase21__dispatch_contiguousEN3c1012MemoryFormatE@plt>
   1541b: 48 8b 44 24 18               	movq	0x18(%rsp), %rax
   15420: 48 8b 4c 24 50               	movq	0x50(%rsp), %rcx
   15425: 48 89 4c 24 30               	movq	%rcx, 0x30(%rsp)
   1542a: 48 3b 05 df de 01 00         	cmpq	0x1dedf(%rip), %rax     # 0x33310 <strncmp+0x33310>
   15431: 74 47                        	je	0x1547a <_Z3runN2at6TensorES0_S0_d+0x101a>
   15433: f0                           	lock
   15434: ff 48 08                     	decl	0x8(%rax)
   15437: 75 41                        	jne	0x1547a <_Z3runN2at6TensorES0_S0_d+0x101a>
   15439: 48 8b 44 24 18               	movq	0x18(%rsp), %rax
   1543e: 8b 40 0c                     	movl	0xc(%rax), %eax
   15441: 83 f8 01                     	cmpl	$0x1, %eax
   15444: 74 16                        	je	0x1545c <_Z3runN2at6TensorES0_S0_d+0xffc>
   15446: 48 8b 7c 24 18               	movq	0x18(%rsp), %rdi
   1544b: 48 8b 07                     	movq	(%rdi), %rax
   1544e: ff 50 10                     	callq	*0x10(%rax)
   15451: 48 8b 44 24 18               	movq	0x18(%rsp), %rax
   15456: f0                           	lock
   15457: ff 48 0c                     	decl	0xc(%rax)
   1545a: 75 1e                        	jne	0x1547a <_Z3runN2at6TensorES0_S0_d+0x101a>
   1545c: 48 8b 7c 24 18               	movq	0x18(%rsp), %rdi
   15461: 48 85 ff                     	testq	%rdi, %rdi
   15464: 74 14                        	je	0x1547a <_Z3runN2at6TensorES0_S0_d+0x101a>
   15466: 48 8b 07                     	movq	(%rdi), %rax
   15469: ff 50 08                     	callq	*0x8(%rax)
   1546c: eb 0c                        	jmp	0x1547a <_Z3runN2at6TensorES0_S0_d+0x101a>
   1546e: 48 8b 05 9b de 01 00         	movq	0x1de9b(%rip), %rax     # 0x33310 <strncmp+0x33310>
   15475: 48 89 44 24 30               	movq	%rax, 0x30(%rsp)
   1547a: 4c 89 6c 24 50               	movq	%r13, 0x50(%rsp)
   1547f: 48 c7 44 24 58 08 00 00 00   	movq	$0x8, 0x58(%rsp)
   15488: 48 8b 84 24 80 00 00 00      	movq	0x80(%rsp), %rax
   15490: 48 89 44 24 60               	movq	%rax, 0x60(%rsp)
   15495: 48 c7 44 24 68 80 00 00 00   	movq	$0x80, 0x68(%rsp)
   1549e: 48 8d 7c 24 28               	leaq	0x28(%rsp), %rdi
   154a3: e8 18 bb 01 00               	callq	0x30fc0 <_ZNK2at10TensorBase7optionsEv@plt>
   154a8: 48 bb ff ff ff ff ff ff ff 00	movabsq	$0xffffffffffffff, %rbx # imm = 0xFFFFFFFFFFFFFF
   154b2: 48 21 d8                     	andq	%rbx, %rax
   154b5: 48 89 44 24 18               	movq	%rax, 0x18(%rsp)
   154ba: 48 8d 7c 24 18               	leaq	0x18(%rsp), %rdi
   154bf: be 0f 01 00 00               	movl	$0x10f, %esi            # imm = 0x10F
   154c4: e8 07 bb 01 00               	callq	0x30fd0 <_ZNR3c1013TensorOptions9set_dtypeESt8optionalINS_10ScalarTypeEE@plt>
   154c9: 48 23 5c 24 18               	andq	0x18(%rsp), %rbx
   154ce: 48 8d 7c 24 20               	leaq	0x20(%rsp), %rdi
   154d3: 48 8d 74 24 50               	leaq	0x50(%rsp), %rsi
   154d8: ba 04 00 00 00               	movl	$0x4, %edx
   154dd: 48 89 d9                     	movq	%rbx, %rcx
   154e0: 45 31 c0                     	xorl	%r8d, %r8d
   154e3: e8 f8 ba 01 00               	callq	0x30fe0 <_ZN5torch5emptyEN3c108ArrayRefIlEENS0_13TensorOptionsESt8optionalINS0_12MemoryFormatEE@plt>
   154e8: f2 0f 10 84 24 a8 00 00 00   	movsd	0xa8(%rsp), %xmm0
   154f1: f2 0f 5a c0                  	cvtsd2ss	%xmm0, %xmm0
   154f5: 48 8d 7c 24 28               	leaq	0x28(%rsp), %rdi
   154fa: 48 8d 74 24 38               	leaq	0x38(%rsp), %rsi
   154ff: 48 8d 54 24 30               	leaq	0x30(%rsp), %rdx
   15504: 48 8d 4c 24 20               	leaq	0x20(%rsp), %rcx
   15509: e8 e2 ba 01 00               	callq	0x30ff0 <_Z46launch_dense_qkv_prefill_causal_h8_kv1or8_d128RKN2at6TensorES2_S2_fRS0_@plt>
   1550e: 40 84 ed                     	testb	%bpl, %bpl
   15511: 74 0d                        	je	0x15520 <_Z3runN2at6TensorES0_S0_d+0x10c0>
   15513: 48 8b 44 24 20               	movq	0x20(%rsp), %rax
   15518: 49 89 07                     	movq	%rax, (%r15)
   1551b: e9 ad 00 00 00               	jmp	0x155cd <_Z3runN2at6TensorES0_S0_d+0x116d>
   15520: 48 8d 7c 24 20               	leaq	0x20(%rsp), %rdi
   15525: e8 96 ba 01 00               	callq	0x30fc0 <_ZNK2at10TensorBase7optionsEv@plt>
   1552a: 48 89 c3                     	movq	%rax, %rbx
   1552d: 66 c7 44 24 50 00 ff         	movw	$0xff00, 0x50(%rsp)     # imm = 0xFF00
   15534: 48 8d 7c 24 50               	leaq	0x50(%rsp), %rdi
   15539: e8 12 ba 01 00               	callq	0x30f50 <_ZN3c106Device8validateEv@plt>
   1553e: b8 00 00 01 00               	movl	$0x10000, %eax          # imm = 0x10000
   15543: a9 00 00 01 00               	testl	$0x10000, %eax          # imm = 0x10000
   15548: 0f b7 44 24 50               	movzwl	0x50(%rsp), %eax
   1554d: 48 0f 44 c3                  	cmoveq	%rbx, %rax
   15551: 48 b9 00 00 ff ff ff ff fb 00	movabsq	$0xfbffffffff0000, %rcx # imm = 0xFBFFFFFFFF0000
   1555b: 48 21 d9                     	andq	%rbx, %rcx
   1555e: 0f b7 c0                     	movzwl	%ax, %eax
   15561: 48 09 c8                     	orq	%rcx, %rax
   15564: 48 ba 00 00 00 00 00 00 04 00	movabsq	$0x4000000000000, %rdx  # imm = 0x4000000000000
   1556e: 48 09 c2                     	orq	%rax, %rdx
   15571: 48 8d 74 24 20               	leaq	0x20(%rsp), %rsi
   15576: 4c 89 ff                     	movq	%r15, %rdi
   15579: 31 c9                        	xorl	%ecx, %ecx
   1557b: 45 31 c0                     	xorl	%r8d, %r8d
   1557e: 45 31 c9                     	xorl	%r9d, %r9d
   15581: e8 7a ba 01 00               	callq	0x31000 <_ZNK2at6Tensor2toEN3c1013TensorOptionsEbbSt8optionalINS1_12MemoryFormatEE@plt>
   15586: 48 8b 44 24 20               	movq	0x20(%rsp), %rax
   1558b: 48 3b 05 7e dd 01 00         	cmpq	0x1dd7e(%rip), %rax     # 0x33310 <strncmp+0x33310>
   15592: 74 39                        	je	0x155cd <_Z3runN2at6TensorES0_S0_d+0x116d>
   15594: f0                           	lock
   15595: ff 48 08                     	decl	0x8(%rax)
   15598: 75 33                        	jne	0x155cd <_Z3runN2at6TensorES0_S0_d+0x116d>
   1559a: 48 8b 44 24 20               	movq	0x20(%rsp), %rax
   1559f: 8b 40 0c                     	movl	0xc(%rax), %eax
   155a2: 83 f8 01                     	cmpl	$0x1, %eax
   155a5: 74 16                        	je	0x155bd <_Z3runN2at6TensorES0_S0_d+0x115d>
   155a7: 48 8b 7c 24 20               	movq	0x20(%rsp), %rdi
   155ac: 48 8b 07                     	movq	(%rdi), %rax
   155af: ff 50 10                     	callq	*0x10(%rax)
   155b2: 48 8b 44 24 20               	movq	0x20(%rsp), %rax
   155b7: f0                           	lock
   155b8: ff 48 0c                     	decl	0xc(%rax)
   155bb: 75 10                        	jne	0x155cd <_Z3runN2at6TensorES0_S0_d+0x116d>
   155bd: 48 8b 7c 24 20               	movq	0x20(%rsp), %rdi
   155c2: 48 85 ff                     	testq	%rdi, %rdi
   155c5: 74 06                        	je	0x155cd <_Z3runN2at6TensorES0_S0_d+0x116d>
   155c7: 48 8b 07                     	movq	(%rdi), %rax
   155ca: ff 50 08                     	callq	*0x8(%rax)
   155cd: 48 8b 44 24 30               	movq	0x30(%rsp), %rax
   155d2: 48 3b 05 37 dd 01 00         	cmpq	0x1dd37(%rip), %rax     # 0x33310 <strncmp+0x33310>
   155d9: 74 39                        	je	0x15614 <_Z3runN2at6TensorES0_S0_d+0x11b4>
   155db: f0                           	lock
   155dc: ff 48 08                     	decl	0x8(%rax)
   155df: 75 33                        	jne	0x15614 <_Z3runN2at6TensorES0_S0_d+0x11b4>
   155e1: 48 8b 44 24 30               	movq	0x30(%rsp), %rax
   155e6: 8b 40 0c                     	movl	0xc(%rax), %eax
   155e9: 83 f8 01                     	cmpl	$0x1, %eax
   155ec: 74 16                        	je	0x15604 <_Z3runN2at6TensorES0_S0_d+0x11a4>
   155ee: 48 8b 7c 24 30               	movq	0x30(%rsp), %rdi
   155f3: 48 8b 07                     	movq	(%rdi), %rax
   155f6: ff 50 10                     	callq	*0x10(%rax)
   155f9: 48 8b 44 24 30               	movq	0x30(%rsp), %rax
   155fe: f0                           	lock
   155ff: ff 48 0c                     	decl	0xc(%rax)
   15602: 75 10                        	jne	0x15614 <_Z3runN2at6TensorES0_S0_d+0x11b4>
   15604: 48 8b 7c 24 30               	movq	0x30(%rsp), %rdi
   15609: 48 85 ff                     	testq	%rdi, %rdi
   1560c: 74 06                        	je	0x15614 <_Z3runN2at6TensorES0_S0_d+0x11b4>
   1560e: 48 8b 07                     	movq	(%rdi), %rax
   15611: ff 50 08                     	callq	*0x8(%rax)
   15614: 48 8b 44 24 38               	movq	0x38(%rsp), %rax
   15619: 48 3b 05 f0 dc 01 00         	cmpq	0x1dcf0(%rip), %rax     # 0x33310 <strncmp+0x33310>
   15620: 74 39                        	je	0x1565b <_Z3runN2at6TensorES0_S0_d+0x11fb>
   15622: f0                           	lock
   15623: ff 48 08                     	decl	0x8(%rax)
   15626: 75 33                        	jne	0x1565b <_Z3runN2at6TensorES0_S0_d+0x11fb>
   15628: 48 8b 44 24 38               	movq	0x38(%rsp), %rax
   1562d: 8b 40 0c                     	movl	0xc(%rax), %eax
   15630: 83 f8 01                     	cmpl	$0x1, %eax
   15633: 74 16                        	je	0x1564b <_Z3runN2at6TensorES0_S0_d+0x11eb>
   15635: 48 8b 7c 24 38               	movq	0x38(%rsp), %rdi
   1563a: 48 8b 07                     	movq	(%rdi), %rax
   1563d: ff 50 10                     	callq	*0x10(%rax)
   15640: 48 8b 44 24 38               	movq	0x38(%rsp), %rax
   15645: f0                           	lock
   15646: ff 48 0c                     	decl	0xc(%rax)
   15649: 75 10                        	jne	0x1565b <_Z3runN2at6TensorES0_S0_d+0x11fb>
   1564b: 48 8b 7c 24 38               	movq	0x38(%rsp), %rdi
   15650: 48 85 ff                     	testq	%rdi, %rdi
   15653: 74 06                        	je	0x1565b <_Z3runN2at6TensorES0_S0_d+0x11fb>
   15655: 48 8b 07                     	movq	(%rdi), %rax
   15658: ff 50 08                     	callq	*0x8(%rax)
   1565b: 48 8b 44 24 28               	movq	0x28(%rsp), %rax
   15660: 48 3b 05 a9 dc 01 00         	cmpq	0x1dca9(%rip), %rax     # 0x33310 <strncmp+0x33310>
   15667: 74 39                        	je	0x156a2 <_Z3runN2at6TensorES0_S0_d+0x1242>
   15669: f0                           	lock
   1566a: ff 48 08                     	decl	0x8(%rax)
   1566d: 75 33                        	jne	0x156a2 <_Z3runN2at6TensorES0_S0_d+0x1242>
   1566f: 48 8b 44 24 28               	movq	0x28(%rsp), %rax
   15674: 8b 40 0c                     	movl	0xc(%rax), %eax
   15677: 83 f8 01                     	cmpl	$0x1, %eax
   1567a: 74 16                        	je	0x15692 <_Z3runN2at6TensorES0_S0_d+0x1232>
   1567c: 48 8b 7c 24 28               	movq	0x28(%rsp), %rdi
   15681: 48 8b 07                     	movq	(%rdi), %rax
   15684: ff 50 10                     	callq	*0x10(%rax)
   15687: 48 8b 44 24 28               	movq	0x28(%rsp), %rax
   1568c: f0                           	lock
   1568d: ff 48 0c                     	decl	0xc(%rax)
   15690: 75 10                        	jne	0x156a2 <_Z3runN2at6TensorES0_S0_d+0x1242>
   15692: 48 8b 7c 24 28               	movq	0x28(%rsp), %rdi
   15697: 48 85 ff                     	testq	%rdi, %rdi
   1569a: 74 06                        	je	0x156a2 <_Z3runN2at6TensorES0_S0_d+0x1242>
   1569c: 48 8b 07                     	movq	(%rdi), %rax
   1569f: ff 50 08                     	callq	*0x8(%rax)
   156a2: 0f b6 84 24 a0 00 00 00      	movzbl	0xa0(%rsp), %eax
   156aa: c6 84 24 a0 00 00 00 00      	movb	$0x0, 0xa0(%rsp)
   156b2: 3c 01                        	cmpb	$0x1, %al
   156b4: 75 15                        	jne	0x156cb <_Z3runN2at6TensorES0_S0_d+0x126b>
   156b6: 48 8b bc 24 90 00 00 00      	movq	0x90(%rsp), %rdi
   156be: 48 8b 07                     	movq	(%rdi), %rax
   156c1: 8b b4 24 98 00 00 00         	movl	0x98(%rsp), %esi
   156c8: ff 50 20                     	callq	*0x20(%rax)
   156cb: 0f b6 7c 24 15               	movzbl	0x15(%rsp), %edi
   156d0: e8 4b b8 01 00               	callq	0x30f20 <_ZN3c108GradMode11set_enabledEb@plt>
   156d5: 4c 89 f8                     	movq	%r15, %rax
   156d8: 48 81 c4 d8 00 00 00         	addq	$0xd8, %rsp
   156df: 5b                           	popq	%rbx
   156e0: 41 5c                        	popq	%r12
   156e2: 41 5d                        	popq	%r13
   156e4: 41 5e                        	popq	%r14
   156e6: 41 5f                        	popq	%r15
   156e8: 5d                           	popq	%rbp
   156e9: c3                           	retq
   156ea: 48 8b 07                     	movq	(%rdi), %rax
   156ed: ff 50 68                     	callq	*0x68(%rax)
   156f0: 89 c6                        	movl	%eax, %esi
   156f2: e9 8e f6 ff ff               	jmp	0x14d85 <_Z3runN2at6TensorES0_S0_d+0x925>
   156f7: 89 c7                        	movl	%eax, %edi
   156f9: e8 12 b9 01 00               	callq	0x31010 <_ZN6caffe28TypeMeta26error_unsupported_typemetaES0_@plt>
   156fe: 89 c7                        	movl	%eax, %edi
   15700: e8 0b b9 01 00               	callq	0x31010 <_ZN6caffe28TypeMeta26error_unsupported_typemetaES0_@plt>
   15705: 89 c7                        	movl	%eax, %edi
   15707: e8 04 b9 01 00               	callq	0x31010 <_ZN6caffe28TypeMeta26error_unsupported_typemetaES0_@plt>
   1570c: 31 f6                        	xorl	%esi, %esi
   1570e: e8 0d b9 01 00               	callq	0x31020 <_ZNK3c1010TensorImpl20is_contiguous_customENS_12MemoryFormatE@plt>
   15713: e9 66 f7 ff ff               	jmp	0x14e7e <_Z3runN2at6TensorES0_S0_d+0xa1e>
   15718: 31 f6                        	xorl	%esi, %esi
   1571a: e8 01 b9 01 00               	callq	0x31020 <_ZNK3c1010TensorImpl20is_contiguous_customENS_12MemoryFormatE@plt>
   1571f: e9 a8 f9 ff ff               	jmp	0x150cc <_Z3runN2at6TensorES0_S0_d+0xc6c>
   15724: 31 f6                        	xorl	%esi, %esi
   15726: e8 f5 b8 01 00               	callq	0x31020 <_ZNK3c1010TensorImpl20is_contiguous_customENS_12MemoryFormatE@plt>
   1572b: e9 e2 fb ff ff               	jmp	0x15312 <_Z3runN2at6TensorES0_S0_d+0xeb2>
   15730: 49 89 fe                     	movq	%rdi, %r14
   15733: e8 f8 b8 01 00               	callq	0x31030 <_ZNK3c1017SymbolicShapeMeta18init_is_contiguousEv@plt>
   15738: 4c 89 f7                     	movq	%r14, %rdi
   1573b: e9 26 f7 ff ff               	jmp	0x14e66 <_Z3runN2at6TensorES0_S0_d+0xa06>
   15740: 48 89 fb                     	movq	%rdi, %rbx
   15743: e8 e8 b8 01 00               	callq	0x31030 <_ZNK3c1017SymbolicShapeMeta18init_is_contiguousEv@plt>
   15748: 48 89 df                     	movq	%rbx, %rdi
   1574b: e9 64 f9 ff ff               	jmp	0x150b4 <_Z3runN2at6TensorES0_S0_d+0xc54>
   15750: 48 89 fb                     	movq	%rdi, %rbx
   15753: e8 d8 b8 01 00               	callq	0x31030 <_ZNK3c1017SymbolicShapeMeta18init_is_contiguousEv@plt>
   15758: 48 89 df                     	movq	%rbx, %rdi
   1575b: e9 9a fb ff ff               	jmp	0x152fa <_Z3runN2at6TensorES0_S0_d+0xe9a>
   15760: 48 8d 0d 7e 69 ff ff         	leaq	-0x9682(%rip), %rcx     # 0xc0e5 <strncmp+0xc0e5>
   15767: ba 0b 00 00 00               	movl	$0xb, %edx
   1576c: eb 1a                        	jmp	0x15788 <_Z3runN2at6TensorES0_S0_d+0x1328>
   1576e: 48 8d 0d 3d 84 ff ff         	leaq	-0x7bc3(%rip), %rcx     # 0xdbb2 <strncmp+0xdbb2>
   15775: ba 0c 00 00 00               	movl	$0xc, %edx
   1577a: eb 0c                        	jmp	0x15788 <_Z3runN2at6TensorES0_S0_d+0x1328>
   1577c: 48 8d 0d cf 79 ff ff         	leaq	-0x8631(%rip), %rcx     # 0xd152 <strncmp+0xd152>
   15783: ba 0d 00 00 00               	movl	$0xd, %edx
   15788: 48 8d 3d 54 7b ff ff         	leaq	-0x84ac(%rip), %rdi     # 0xd2e3 <strncmp+0xd2e3>
   1578f: 48 8d 35 b1 86 ff ff         	leaq	-0x794f(%rip), %rsi     # 0xde47 <strncmp+0xde47>
   15796: e8 95 b7 01 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   1579b: 48 8d 0d af 94 ff ff         	leaq	-0x6b51(%rip), %rcx     # 0xec51 <strncmp+0xec51>
   157a2: ba 16 00 00 00               	movl	$0x16, %edx
   157a7: eb 28                        	jmp	0x157d1 <_Z3runN2at6TensorES0_S0_d+0x1371>
   157a9: 48 8d 0d 96 91 ff ff         	leaq	-0x6e6a(%rip), %rcx     # 0xe946 <strncmp+0xe946>
   157b0: ba 18 00 00 00               	movl	$0x18, %edx
   157b5: eb 1a                        	jmp	0x157d1 <_Z3runN2at6TensorES0_S0_d+0x1371>
   157b7: 48 8d 0d 01 73 ff ff         	leaq	-0x8cff(%rip), %rcx     # 0xcabf <strncmp+0xcabf>
   157be: ba 19 00 00 00               	movl	$0x19, %edx
   157c3: eb 0c                        	jmp	0x157d1 <_Z3runN2at6TensorES0_S0_d+0x1371>
   157c5: 48 8d 0d 22 6e ff ff         	leaq	-0x91de(%rip), %rcx     # 0xc5ee <strncmp+0xc5ee>
   157cc: ba 17 00 00 00               	movl	$0x17, %edx
   157d1: 48 8d 3d 0b 7b ff ff         	leaq	-0x84f5(%rip), %rdi     # 0xd2e3 <strncmp+0xd2e3>
   157d8: 48 8d 35 68 86 ff ff         	leaq	-0x7998(%rip), %rsi     # 0xde47 <strncmp+0xde47>
   157df: e8 4c b7 01 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   157e4: 48 8d 3d cc 74 ff ff         	leaq	-0x8b34(%rip), %rdi     # 0xccb7 <strncmp+0xccb7>
   157eb: 48 8d 35 9f 76 ff ff         	leaq	-0x8961(%rip), %rsi     # 0xce91 <strncmp+0xce91>
   157f2: 48 8d 0d cd 74 ff ff         	leaq	-0x8b33(%rip), %rcx     # 0xccc6 <strncmp+0xccc6>
   157f9: ba 11 05 00 00               	movl	$0x511, %edx            # imm = 0x511
   157fe: e8 2d b7 01 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   15803: 48 8d 3d ad 74 ff ff         	leaq	-0x8b53(%rip), %rdi     # 0xccb7 <strncmp+0xccb7>
   1580a: 48 8d 35 80 76 ff ff         	leaq	-0x8980(%rip), %rsi     # 0xce91 <strncmp+0xce91>
   15811: 48 8d 0d ae 74 ff ff         	leaq	-0x8b52(%rip), %rcx     # 0xccc6 <strncmp+0xccc6>
   15818: ba 11 05 00 00               	movl	$0x511, %edx            # imm = 0x511
   1581d: e8 0e b7 01 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   15822: 48 8d 3d 8e 74 ff ff         	leaq	-0x8b72(%rip), %rdi     # 0xccb7 <strncmp+0xccb7>
   15829: 48 8d 35 61 76 ff ff         	leaq	-0x899f(%rip), %rsi     # 0xce91 <strncmp+0xce91>
   15830: 48 8d 0d 8f 74 ff ff         	leaq	-0x8b71(%rip), %rcx     # 0xccc6 <strncmp+0xccc6>
   15837: ba 11 05 00 00               	movl	$0x511, %edx            # imm = 0x511
   1583c: e8 ef b6 01 00               	callq	0x30f30 <_ZN3c106detail14torchCheckFailEPKcS2_jS2_@plt>
   15841: 48 8d 3d 6e 8b ff ff         	leaq	-0x7492(%rip), %rdi     # 0xe3b6 <strncmp+0xe3b6>
   15848: 48 8d 35 42 76 ff ff         	leaq	-0x89be(%rip), %rsi     # 0xce91 <strncmp+0xce91>
   1584f: 48 8d 0d 05 92 ff ff         	leaq	-0x6dfb(%rip), %rcx     # 0xea5b <strncmp+0xea5b>
   15856: ba e5 06 00 00               	movl	$0x6e5, %edx            # imm = 0x6E5
   1585b: e8 e0 b7 01 00               	callq	0x31040 <_ZN3c106detail23torchInternalAssertFailEPKcS2_jS2_NS0_22CompileTimeEmptyStringE@plt>
   15860: 48 8d 3d 4f 8b ff ff         	leaq	-0x74b1(%rip), %rdi     # 0xe3b6 <strncmp+0xe3b6>
   15867: 48 8d 35 23 76 ff ff         	leaq	-0x89dd(%rip), %rsi     # 0xce91 <strncmp+0xce91>
   1586e: 48 8d 0d e6 91 ff ff         	leaq	-0x6e1a(%rip), %rcx     # 0xea5b <strncmp+0xea5b>
   15875: ba e5 06 00 00               	movl	$0x6e5, %edx            # imm = 0x6E5
   1587a: e8 c1 b7 01 00               	callq	0x31040 <_ZN3c106detail23torchInternalAssertFailEPKcS2_jS2_NS0_22CompileTimeEmptyStringE@plt>
   1587f: 48 8d 3d 30 8b ff ff         	leaq	-0x74d0(%rip), %rdi     # 0xe3b6 <strncmp+0xe3b6>
   15886: 48 8d 35 04 76 ff ff         	leaq	-0x89fc(%rip), %rsi     # 0xce91 <strncmp+0xce91>
   1588d: 48 8d 0d c7 91 ff ff         	leaq	-0x6e39(%rip), %rcx     # 0xea5b <strncmp+0xea5b>
   15894: ba e5 06 00 00               	movl	$0x6e5, %edx            # imm = 0x6E5
   15899: e8 a2 b7 01 00               	callq	0x31040 <_ZN3c106detail23torchInternalAssertFailEPKcS2_jS2_NS0_22CompileTimeEmptyStringE@plt>
   1589e: e9 0e 01 00 00               	jmp	0x159b1 <_Z3runN2at6TensorES0_S0_d+0x1551>
   158a3: e9 18 01 00 00               	jmp	0x159c0 <_Z3runN2at6TensorES0_S0_d+0x1560>
   158a8: e9 22 01 00 00               	jmp	0x159cf <_Z3runN2at6TensorES0_S0_d+0x156f>
   158ad: 48 89 c3                     	movq	%rax, %rbx
   158b0: e9 46 01 00 00               	jmp	0x159fb <_Z3runN2at6TensorES0_S0_d+0x159b>
   158b5: 48 89 c3                     	movq	%rax, %rbx
   158b8: e9 3e 01 00 00               	jmp	0x159fb <_Z3runN2at6TensorES0_S0_d+0x159b>
   158bd: 48 89 c3                     	movq	%rax, %rbx
   158c0: e9 36 01 00 00               	jmp	0x159fb <_Z3runN2at6TensorES0_S0_d+0x159b>
   158c5: 48 89 c3                     	movq	%rax, %rbx
   158c8: e9 2e 01 00 00               	jmp	0x159fb <_Z3runN2at6TensorES0_S0_d+0x159b>
   158cd: 48 89 c3                     	movq	%rax, %rbx
   158d0: e9 26 01 00 00               	jmp	0x159fb <_Z3runN2at6TensorES0_S0_d+0x159b>
   158d5: 48 89 c3                     	movq	%rax, %rbx
   158d8: e9 1e 01 00 00               	jmp	0x159fb <_Z3runN2at6TensorES0_S0_d+0x159b>
   158dd: 48 89 c3                     	movq	%rax, %rbx
   158e0: e9 16 01 00 00               	jmp	0x159fb <_Z3runN2at6TensorES0_S0_d+0x159b>
   158e5: 48 89 c3                     	movq	%rax, %rbx
   158e8: e9 0e 01 00 00               	jmp	0x159fb <_Z3runN2at6TensorES0_S0_d+0x159b>
   158ed: 48 89 c3                     	movq	%rax, %rbx
   158f0: e9 06 01 00 00               	jmp	0x159fb <_Z3runN2at6TensorES0_S0_d+0x159b>
   158f5: 48 89 c3                     	movq	%rax, %rbx
   158f8: e9 fe 00 00 00               	jmp	0x159fb <_Z3runN2at6TensorES0_S0_d+0x159b>
   158fd: 48 89 c3                     	movq	%rax, %rbx
   15900: e9 f6 00 00 00               	jmp	0x159fb <_Z3runN2at6TensorES0_S0_d+0x159b>
   15905: 48 89 c3                     	movq	%rax, %rbx
   15908: e9 ee 00 00 00               	jmp	0x159fb <_Z3runN2at6TensorES0_S0_d+0x159b>
   1590d: 48 89 c7                     	movq	%rax, %rdi
   15910: e8 bb 0d 00 00               	callq	0x166d0 <__clang_call_terminate>
   15915: 48 89 c7                     	movq	%rax, %rdi
   15918: e8 b3 0d 00 00               	callq	0x166d0 <__clang_call_terminate>
   1591d: 48 89 c7                     	movq	%rax, %rdi
   15920: e8 ab 0d 00 00               	callq	0x166d0 <__clang_call_terminate>
   15925: 48 89 c7                     	movq	%rax, %rdi
   15928: e8 a3 0d 00 00               	callq	0x166d0 <__clang_call_terminate>
   1592d: 48 89 c7                     	movq	%rax, %rdi
   15930: e8 9b 0d 00 00               	callq	0x166d0 <__clang_call_terminate>
   15935: 48 89 c7                     	movq	%rax, %rdi
   15938: e8 93 0d 00 00               	callq	0x166d0 <__clang_call_terminate>
   1593d: 48 89 c7                     	movq	%rax, %rdi
   15940: e8 8b 0d 00 00               	callq	0x166d0 <__clang_call_terminate>
   15945: 48 89 c7                     	movq	%rax, %rdi
   15948: e8 83 0d 00 00               	callq	0x166d0 <__clang_call_terminate>
   1594d: eb 62                        	jmp	0x159b1 <_Z3runN2at6TensorES0_S0_d+0x1551>
   1594f: eb 6f                        	jmp	0x159c0 <_Z3runN2at6TensorES0_S0_d+0x1560>
   15951: eb 7c                        	jmp	0x159cf <_Z3runN2at6TensorES0_S0_d+0x156f>
   15953: 48 89 c7                     	movq	%rax, %rdi
   15956: e8 75 0d 00 00               	callq	0x166d0 <__clang_call_terminate>
   1595b: 48 89 c3                     	movq	%rax, %rbx
   1595e: e9 98 00 00 00               	jmp	0x159fb <_Z3runN2at6TensorES0_S0_d+0x159b>
   15963: 48 89 c3                     	movq	%rax, %rbx
   15966: e9 90 00 00 00               	jmp	0x159fb <_Z3runN2at6TensorES0_S0_d+0x159b>
   1596b: 48 89 c3                     	movq	%rax, %rbx
   1596e: 48 8d 7c 24 18               	leaq	0x18(%rsp), %rdi
   15973: eb 2b                        	jmp	0x159a0 <_Z3runN2at6TensorES0_S0_d+0x1540>
   15975: 48 89 c3                     	movq	%rax, %rbx
   15978: 48 8d 7c 24 18               	leaq	0x18(%rsp), %rdi
   1597d: eb 3a                        	jmp	0x159b9 <_Z3runN2at6TensorES0_S0_d+0x1559>
   1597f: 48 89 c3                     	movq	%rax, %rbx
   15982: 48 8d 7c 24 18               	leaq	0x18(%rsp), %rdi
   15987: eb 3f                        	jmp	0x159c8 <_Z3runN2at6TensorES0_S0_d+0x1568>
   15989: 48 89 c3                     	movq	%rax, %rbx
   1598c: 48 8d 7c 24 20               	leaq	0x20(%rsp), %rdi
   15991: e8 ba b6 01 00               	callq	0x31050 <_ZN2at10TensorBaseD2Ev@plt>
   15996: eb 03                        	jmp	0x1599b <_Z3runN2at6TensorES0_S0_d+0x153b>
   15998: 48 89 c3                     	movq	%rax, %rbx
   1599b: 48 8d 7c 24 30               	leaq	0x30(%rsp), %rdi
   159a0: e8 ab b6 01 00               	callq	0x31050 <_ZN2at10TensorBaseD2Ev@plt>
   159a5: eb 0d                        	jmp	0x159b4 <_Z3runN2at6TensorES0_S0_d+0x1554>
   159a7: 48 89 c3                     	movq	%rax, %rbx
   159aa: eb 4f                        	jmp	0x159fb <_Z3runN2at6TensorES0_S0_d+0x159b>
   159ac: 48 89 c3                     	movq	%rax, %rbx
   159af: eb 4a                        	jmp	0x159fb <_Z3runN2at6TensorES0_S0_d+0x159b>
   159b1: 48 89 c3                     	movq	%rax, %rbx
   159b4: 48 8d 7c 24 38               	leaq	0x38(%rsp), %rdi
   159b9: e8 92 b6 01 00               	callq	0x31050 <_ZN2at10TensorBaseD2Ev@plt>
   159be: eb 03                        	jmp	0x159c3 <_Z3runN2at6TensorES0_S0_d+0x1563>
   159c0: 48 89 c3                     	movq	%rax, %rbx
   159c3: 48 8d 7c 24 28               	leaq	0x28(%rsp), %rdi
   159c8: e8 83 b6 01 00               	callq	0x31050 <_ZN2at10TensorBaseD2Ev@plt>
   159cd: eb 03                        	jmp	0x159d2 <_Z3runN2at6TensorES0_S0_d+0x1572>
   159cf: 48 89 c3                     	movq	%rax, %rbx
   159d2: 0f b6 84 24 a0 00 00 00      	movzbl	0xa0(%rsp), %eax
   159da: c6 84 24 a0 00 00 00 00      	movb	$0x0, 0xa0(%rsp)
   159e2: 3c 01                        	cmpb	$0x1, %al
   159e4: 75 15                        	jne	0x159fb <_Z3runN2at6TensorES0_S0_d+0x159b>
   159e6: 48 8b bc 24 90 00 00 00      	movq	0x90(%rsp), %rdi
   159ee: 48 8b 07                     	movq	(%rdi), %rax
   159f1: 8b b4 24 98 00 00 00         	movl	0x98(%rsp), %esi
   159f8: ff 50 20                     	callq	*0x20(%rax)
   159fb: 0f b6 44 24 15               	movzbl	0x15(%rsp), %eax
   15a00: 0f b6 f8                     	movzbl	%al, %edi
   15a03: e8 18 b5 01 00               	callq	0x30f20 <_ZN3c108GradMode11set_enabledEb@plt>
   15a08: 48 89 df                     	movq	%rbx, %rdi
   15a0b: e8 e0 b4 01 00               	callq	0x30ef0 <_Unwind_Resume@plt>
   15a10: 48 89 c7                     	movq	%rax, %rdi
   15a13: e8 b8 0c 00 00               	callq	0x166d0 <__clang_call_terminate>
   15a18: 0f 1f 84 00 00 00 00 00      	nopl	(%rax,%rax)
