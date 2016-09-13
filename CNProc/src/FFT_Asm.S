#if defined HAVE_BGP

_C1:	.long	0x3F800000, 0x3F800000
_W:	.long	0x3F6C835E, 0xBEC3EF15
_W2:	.long	0x3F3504F3, 0xBF3504F3

twiddle:
	.long	0x3F7FEC43, 0xBCC90AB0
	.long	0x3F7FB10F, 0xBD48FB30
	.long	0x3F7F4E6D, 0xBD96A905
	.long	0x3F7EC46D, 0xBDC8BD36
	.long	0x3F7E1324, 0xBDFAB272
	.long	0x3F7D3AAC, 0xBE164083
	.long	0x3F7C3B28, 0xBE2F10A2
	.long	0x3F7B14BE, 0xBE47C5C2
	.long	0x3F79C79D, 0xBE605C13
	.long	0x3F7853F8, 0xBE78CFCC
	.long	0x3F76BA07, 0xBE888E93
	.long	0x3F74FA0B, 0xBE94A031
	.long	0x3F731448, 0xBEA09AE5
	.long	0x3F710908, 0xBEAC7CD3
	.long	0x3F6ED89D, 0xBEB8442A
	.long	0x3F7FB10F, 0xBD48FB30
	.long	0x3F7EC46D, 0xBDC8BD36
	.long	0x3F7D3AAC, 0xBE164083
	.long	0x3F7B14BE, 0xBE47C5C2
	.long	0x3F7853F8, 0xBE78CFCC
	.long	0x3F74FA0B, 0xBE94A031
	.long	0x3F710908, 0xBEAC7CD3
	.long	0x3F6C835F, 0xBEC3EF16
	.long	0x3F676BD7, 0xBEDAE880
	.long	0x3F61C598, 0xBEF15AE9
	.long	0x3F5B941A, 0xBF039C3C
	.long	0x3F54DB31, 0xBF0E39DA
	.long	0x3F4D9F02, 0xBF187FC0
	.long	0x3F45E403, 0xBF226799
	.long	0x3F3DAEFA, 0xBF2BEB4A
	.long	0x3F7F4E6D, 0xBD96A905
	.long	0x3F7D3AAC, 0xBE164083
	.long	0x3F79C79D, 0xBE605C13
	.long	0x3F74FA0B, 0xBE94A031
	.long	0x3F6ED89D, 0xBEB8442A
	.long	0x3F676BD7, 0xBEDAE880
	.long	0x3F5EBE05, 0xBEFC5D26
	.long	0x3F54DB31, 0xBF0E39DA
	.long	0x3F49D113, 0xBF1D7FD1
	.long	0x3F3DAEFA, 0xBF2BEB4A
	.long	0x3F3085BB, 0xBF396841
	.long	0x3F226799, 0xBF45E403
	.long	0x3F13682B, 0xBF514D3C
	.long	0x3F039C3E, 0xBF5B9419
	.long	0x3EE63374, 0xBF64AA5A
	.long	0x3F7EC46D, 0xBDC8BD36
	.long	0x3F7B14BE, 0xBE47C5C2
	.long	0x3F74FA0B, 0xBE94A031
	.long	0x3F6C835F, 0xBEC3EF16
	.long	0x3F61C598, 0xBEF15AE9
	.long	0x3F54DB31, 0xBF0E39DA
	.long	0x3F45E403, 0xBF226799
	.long	0x3F3504F3, 0xBF3504F4
	.long	0x3F226799, 0xBF45E403
	.long	0x3F0E39DA, 0xBF54DB31
	.long	0x3EF15AEB, 0xBF61C598
	.long	0x3EC3EF15, 0xBF6C835F
	.long	0x3E94A030, 0xBF74FA0B
	.long	0x3E47C5C4, 0xBF7B14BE
	.long	0x3DC8BD35, 0xBF7EC46D
	.long	0x3F7E1324, 0xBDFAB272
	.long	0x3F7853F8, 0xBE78CFCC
	.long	0x3F6ED89D, 0xBEB8442A
	.long	0x3F61C598, 0xBEF15AE9
	.long	0x3F514D3D, 0xBF13682B
	.long	0x3F3DAEFA, 0xBF2BEB4A
	.long	0x3F273656, 0xBF41D870
	.long	0x3F0E39DA, 0xBF54DB31
	.long	0x3EE63374, 0xBF64AA5A
	.long	0x3EAC7CD3, 0xBF710908
	.long	0x3E605C12, 0xBF79C79D
	.long	0x3DC8BD35, 0xBF7EC46D
	.long	0xBCC90AAD, 0xBF7FEC43
	.long	0xBE164082, 0xBF7D3AAC
	.long	0xBE888E92, 0xBF76BA07
	.long	0x3F7D3AAC, 0xBE164083
	.long	0x3F74FA0B, 0xBE94A031
	.long	0x3F676BD7, 0xBEDAE880
	.long	0x3F54DB31, 0xBF0E39DA
	.long	0x3F3DAEFA, 0xBF2BEB4A
	.long	0x3F226799, 0xBF45E403
	.long	0x3F039C3E, 0xBF5B9419
	.long	0x3EC3EF15, 0xBF6C835F
	.long	0x3E78CFD0, 0xBF7853F8
	.long	0x3DC8BD35, 0xBF7EC46D
	.long	0xBD48FB21, 0xBF7FB10F
	.long	0xBE47C5C2, 0xBF7B14BE
	.long	0xBEAC7CD2, 0xBF710908
	.long	0xBEF15AE6, 0xBF61C598
	.long	0xBF187FC1, 0xBF4D9F01
	.long	0x3F7C3B28, 0xBE2F10A2
	.long	0x3F710908, 0xBEAC7CD3
	.long	0x3F5EBE05, 0xBEFC5D26
	.long	0x3F45E403, 0xBF226799
	.long	0x3F273656, 0xBF41D870
	.long	0x3F039C3E, 0xBF5B9419
	.long	0x3EB8442B, 0xBF6ED89D
	.long	0x3E47C5C4, 0xBF7B14BE
	.long	0x3CC90ABE, 0xBF7FEC43
	.long	0xBE164082, 0xBF7D3AAC
	.long	0xBEA09AE4, 0xBF731448
	.long	0xBEF15AE6, 0xBF61C598
	.long	0xBF1D7FD0, 0xBF49D113
	.long	0xBF3DAEF8, 0xBF2BEB4B
	.long	0xBF584852, 0xBF08F59C
	.long	0x3F7B14BE, 0xBE47C5C2
	.long	0x3F6C835F, 0xBEC3EF16
	.long	0x3F54DB31, 0xBF0E39DA
	.long	0x3F3504F3, 0xBF3504F4
	.long	0x3F0E39DA, 0xBF54DB31
	.long	0x3EC3EF15, 0xBF6C835F
	.long	0x3E47C5C4, 0xBF7B14BE
	.long	0xB33BBD2E, 0xBF800000
	.long	0xBE47C5C2, 0xBF7B14BE
	.long	0xBEC3EF14, 0xBF6C835F
	.long	0xBF0E39D9, 0xBF54DB32
	.long	0xBF3504F3, 0xBF3504F3
	.long	0xBF54DB32, 0xBF0E39D9
	.long	0xBF6C835E, 0xBEC3EF17
	.long	0xBF7B14BF, 0xBE47C5C1
	.long	0x3F79C79D, 0xBE605C13
	.long	0x3F676BD7, 0xBEDAE880
	.long	0x3F49D113, 0xBF1D7FD1
	.long	0x3F226799, 0xBF45E403
	.long	0x3EE63374, 0xBF64AA5A
	.long	0x3E78CFD0, 0xBF7853F8
	.long	0x3CC90ABE, 0xBF7FEC43
	.long	0xBE47C5C2, 0xBF7B14BE
	.long	0xBECF7BCB, 0xBF6A09A6
	.long	0xBF187FC1, 0xBF4D9F01
	.long	0xBF41D870, 0xBF273656
	.long	0xBF61C596, 0xBEF15AED
	.long	0xBF76BA07, 0xBE888E92
	.long	0xBF7FB10F, 0xBD48FB3D
	.long	0xBF7C3B28, 0x3E2F10A9
	.long	0x3F7853F8, 0xBE78CFCC
	.long	0x3F61C598, 0xBEF15AE9
	.long	0x3F3DAEFA, 0xBF2BEB4A
	.long	0x3F0E39DA, 0xBF54DB31
	.long	0x3EAC7CD3, 0xBF710908
	.long	0x3DC8BD35, 0xBF7EC46D
	.long	0xBE164082, 0xBF7D3AAC
	.long	0xBEC3EF14, 0xBF6C835F
	.long	0xBF187FC1, 0xBF4D9F01
	.long	0xBF45E404, 0xBF226799
	.long	0xBF676BD7, 0xBEDAE87F
	.long	0xBF7B14BF, 0xBE47C5C1
	.long	0xBF7FB10F, 0x3D48FB2C
	.long	0xBF74FA0B, 0x3E94A030
	.long	0xBF5B941B, 0x3F039C3C
	.long	0x3F76BA07, 0xBE888E93
	.long	0x3F5B941A, 0xBF039C3C
	.long	0x3F3085BB, 0xBF396841
	.long	0x3EF15AEB, 0xBF61C598
	.long	0x3E605C12, 0xBF79C79D
	.long	0xBD48FB21, 0xBF7FB10F
	.long	0xBEA09AE4, 0xBF731448
	.long	0xBF0E39D9, 0xBF54DB32
	.long	0xBF41D870, 0xBF273656
	.long	0xBF676BD7, 0xBEDAE87F
	.long	0xBF7C3B28, 0xBE2F109D
	.long	0xBF7EC46D, 0x3DC8BD27
	.long	0xBF6ED89E, 0x3EB84428
	.long	0xBF4D9F03, 0x3F187FC0
	.long	0xBF1D7FD1, 0x3F49D113
	.long	0x3F74FA0B, 0xBE94A031
	.long	0x3F54DB31, 0xBF0E39DA
	.long	0x3F226799, 0xBF45E403
	.long	0x3EC3EF15, 0xBF6C835F
	.long	0x3DC8BD35, 0xBF7EC46D
	.long	0xBE47C5C2, 0xBF7B14BE
	.long	0xBEF15AE6, 0xBF61C598
	.long	0xBF3504F3, 0xBF3504F3
	.long	0xBF61C596, 0xBEF15AED
	.long	0xBF7B14BF, 0xBE47C5C1
	.long	0xBF7EC46D, 0x3DC8BD27
	.long	0xBF6C835F, 0x3EC3EF16
	.long	0xBF45E405, 0x3F226798
	.long	0xBF0E39DD, 0x3F54DB2F
	.long	0xBE94A02D, 0x3F74FA0B
	.long	0x3F731448, 0xBEA09AE5
	.long	0x3F4D9F02, 0xBF187FC0
	.long	0x3F13682B, 0xBF514D3C
	.long	0x3E94A030, 0xBF74FA0B
	.long	0xBCC90AAD, 0xBF7FEC43
	.long	0xBEAC7CD2, 0xBF710908
	.long	0xBF1D7FD0, 0xBF49D113
	.long	0xBF54DB32, 0xBF0E39D9
	.long	0xBF76BA07, 0xBE888E92
	.long	0xBF7FB10F, 0x3D48FB2C
	.long	0xBF6ED89E, 0x3EB84428
	.long	0xBF45E405, 0x3F226798
	.long	0xBF08F59D, 0x3F584851
	.long	0xBE78CFD9, 0x3F7853F7
	.long	0x3D96A922, 0x3F7F4E6D
	.long	0x3F710908, 0xBEAC7CD3
	.long	0x3F45E403, 0xBF226799
	.long	0x3F039C3E, 0xBF5B9419
	.long	0x3E47C5C4, 0xBF7B14BE
	.long	0xBE164082, 0xBF7D3AAC
	.long	0xBEF15AE6, 0xBF61C598
	.long	0xBF3DAEF8, 0xBF2BEB4B
	.long	0xBF6C835E, 0xBEC3EF17
	.long	0xBF7FB10F, 0xBD48FB3D
	.long	0xBF74FA0B, 0x3E94A030
	.long	0xBF4D9F03, 0x3F187FC0
	.long	0xBF0E39DD, 0x3F54DB2F
	.long	0xBE78CFD9, 0x3F7853F7
	.long	0x3DC8BD1D, 0x3F7EC46D
	.long	0x3EDAE87B, 0x3F676BD9
	.long	0x3F6ED89D, 0xBEB8442A
	.long	0x3F3DAEFA, 0xBF2BEB4A
	.long	0x3EE63374, 0xBF64AA5A
	.long	0x3DC8BD35, 0xBF7EC46D
	.long	0xBE888E92, 0xBF76BA07
	.long	0xBF187FC1, 0xBF4D9F01
	.long	0xBF584852, 0xBF08F59C
	.long	0xBF7B14BF, 0xBE47C5C1
	.long	0xBF7C3B28, 0x3E2F10A9
	.long	0xBF5B941B, 0x3F039C3C
	.long	0xBF1D7FD1, 0x3F49D113
	.long	0xBE94A02D, 0x3F74FA0B
	.long	0x3D96A922, 0x3F7F4E6D
	.long	0x3EDAE87B, 0x3F676BD9
	.long	0x3F396841, 0x3F3085BC


.align	5
.global	_fft256

_fft256:

#define C1	31
#define	W	30
#define W2	29

#define	a0	0
#define a1	4
#define a2	8
#define a3	12
#define	a4	1
#define a5	5
#define a6	9
#define a7	13
#define	a8	2
#define a9	6
#define a10	10
#define a11	14
#define	a12	3
#define a13	7
#define a14	11
#define a15	15

#define	b0	16
#define b1	a0
#define b2	a2
#define b3	a1
#define	b4	a3
#define b5	a4
#define b6	a6
#define b7	a5
#define	b8	a7
#define b9	a8
#define b10	a10
#define b11	a9
#define	b12	a11
#define b13	a12
#define b14	a14
#define b15	a13

#define c0	a15
#define c2	b0
#define c1	b2
#define c3	b1
#define	c4	b3
#define	c6	b4
#define c5	b6
#define	c7	b5
#define c8	b7
#define	c10	b8
#define	c9	b10
#define	c11	b9
#define	c12	b11
#define	c14	b12
#define	c13	b14
#define	c15	b13

#define	x5	b15
#define	x6	17
#define	x7	18
#define	x9	19
#define	x11	20
#define	x13	21
#define	x14	22
#define	x15	23

#define	d5	c5
#define d6	c6
#define	d7	c7
#define	d9	c9
#define	d11	c11
#define	d13	c13
#define	d14	c14
#define	d15	c15

#define	e0	30
#define	e1	29
#define	e2	28
#define	e3	27
#define	e4	26
#define	e5	25
#define	e6	24
#define	e7	23
#define	e8	22
#define	e9	21
#define	e10	20
#define	e11	19
#define	e12	18
#define	e13	17
#define	e14	16
#define	e15	15

#define	f0	14
#define	f2	e0
#define	f1	e2
#define	f3	e1
#define	f4	e3
#define	f6	e4
#define	f5	e6
#define	f7	e5
#define	f8	e7
#define	f10	e8
#define	f9	e10
#define	f11	e9
#define	f12	e11
#define	f14	e12
#define	f13	e14
#define	f15	e13

#define	f0	14
#define	f2	e0
#define	f1	e2
#define	f3	e1
#define	f4	e3
#define	f6	e4
#define	f5	e6
#define	f7	e5
#define	f8	e7
#define	f10	e8
#define	f9	e10
#define	f11	e9
#define	f12	e11
#define	f14	e12
#define	f13	e14
#define	f15	e13

#define t4	10
#define t8	11
#define t12	12
#define t1	13
#define t5	14
#define t9	1
#define t13	3
#define t2	5
#define t6	7
#define t10	9
#define t14	t4
#define	t3	t8
#define	t7	t12
#define	t11	t1
#define	t15	t5

#define	y4	0
#define	y8	2
#define	y12	4
#define	y1	6
#define	y5	8
#define	y9	y4
#define	y13	y8
#define	y2	y12
#define	y6	y1
#define	y10	y5
#define y14	g4
#define	y3	g8
#define	y7	g12
#define	y11	g1
#define	y15	g5

#define	g8	f8
#define	g12	f12
#define	g1	f1
#define	g5	f5
#define	g4	f4
#define	g9	f9
#define	g13	f13
#define	g2	f2
#define	g6	f6
#define	g10	f10
#define	g14	f14
#define	g3	f3
#define	g7	f7
#define	g11	f11
#define g15	f15

	li	9,-16
					; stfpdux	14,1,9
					; stfpdux	15,1,9
					; stfpdux	16,1,9
					; stfpdux	17,1,9
					; stfpdux	18,1,9
					; stfpdux	19,1,9
					; stfpdux	20,1,9
					; stfpdux	21,1,9
					; stfpdux	22,1,9
					; stfpdux	23,1,9
					; stfpdux	24,1,9
					; stfpdux	25,1,9
					; stfpdux	26,1,9
					; stfpdux	27,1,9
					; stfpdux	28,1,9
	li	10,17*16*8/32		; stfpdux	29,1,9
	li	8,0			; stfpdux	30,1,9
	mtctr	10			; stfpdux	31,1,9

0:	dcbt	3,8
	addi	8,8,32			; bdnz	0b


	li	9,16*8			; lfpsx		a0,0,3
	lis	7,_C1@ha		; lfpsux	a4,3,9
	la	7,_C1@l(7)		; lfpsux	a8,3,9
					; lfpsx		C1,0,7
					; lfpsux	a12,3,9
	addi	7,7,8			; lfpsux	a1,3,9
	li	8,8			; lfpsux	a5,3,9
	li	0,15			; lfpsux	a9,3,9
	mtctr	0			; lfpsux	a13,3,9
	lis	6,twiddle-8@ha		; lfpsux	a2,3,9
	la	6,twiddle-8@l(6)	; lfpsux	a6,3,9
					; lfpsux	a10,3,9
					; lfpsux	a14,3,9
					; lfpsux	a3,3,9
					; lfpsux	a7,3,9
					; lfpsux	a11,3,9
					; lfpsux	a15,3,9

	fpadd	b0,a0,a2
	fpsub	b1,a0,a2
	fpadd	b2,a1,a3
	fpsub	b3,a1,a3

	fpadd	b4,a4,a6
	fpsub	b5,a4,a6
	fpadd	b6,a5,a7
	fpsub	b7,a5,a7

	fpadd	b8,a8,a10
	fpsub	b9,a8,a10
	fpadd	b10,a9,a11
	fpsub	b11,a9,a11

	fpadd	b12,a12,a14
	fpsub	b13,a12,a14
	fpadd	b14,a13,a15
	fpsub	b15,a13,a15

	fpadd	c0,b0,b2
	fpsub	c2,b0,b2		; lfpsx		W,0,7
	fxcxnsma c1,C1,b3,b1		; lfpsx		W2,7,8
	fxcxnpma c3,C1,b3,b1

	fpadd	c4,b4,b6
	fpsub	c6,b4,b6
	fxcxnsma c5,C1,b7,b5
	fxcxnpma c7,C1,b7,b5

	fpadd	c8,b8,b10
	fpsub	c10,b8,b10
	fxcxnsma c9,C1,b11,b9
	fxcxnpma c11,C1,b11,b9

	fpadd	c12,b12,b14
	fpsub	c14,b12,b14
	fxcxnsma c13,C1,b15,b13
	fxcxnpma c15,C1,b15,b13

	fxpmul	x5,c5,W
	fxpmul	x6,c6,W2
	fxpmul	x7,c7,W

	fxpmul	x9,c9,W2
	fxpmul	x11,W2,c11

	fxpmul	x13,c13,W
	fxpmul	x14,W2,c14
	fxpmul	x15,c15,W

	fxcxnpma d5,c5,W,x5
	fxcxnpma d6,c6,W2,x6
	fxcxnsma d7,c7,W,x7

	fxcxnpma d9,c9,W2,x9
	fxcxnsma d11,W2,c11,x11

	fxcxnsma d13,c13,W,x13
	fxcxnsma d14,W2,c14,x14
	fxcxnpma d15,c15,W,x15

	fpadd	e0,c0,c8
	fpsub	e1,c0,c8
	fpadd	e2,c4,c12
	fpsub	e3,c4,c12

	fpadd	e4,c1,d9
	fpsub	e5,c1,d9
	fxnmsub	e6,d13,C1,d5
	fxmadd	e7,d13,C1,d5

	fxcxnsma e8,C1,c10,c2
	fxcxnpma e9,C1,c10,c2
	fpsub	e10,d6,d14
	fpadd	e11,d6,d14

	fpsub	e12,c3,d11
	fpadd	e13,c3,d11
	fxnmadd	e14,d7,C1,d15
	fxnmsub	e15,d7,C1,d15		; addi		3,3,-239*8

	fpadd	f0,e0,e2		; lfpsx		a0,0,3
	fpsub	f2,e0,e2		; lfpsux	a4,3,9
	fxcxnsma f1,C1,e3,e1		; lfpsux	a8,3,9
	fxcxnpma f3,C1,e3,e1		; lfpsux	a12,3,9

	fpadd	f4,e4,e6		; lfpsux	a1,3,9
	fpsub	f6,e4,e6		; stfpsx	f0,0,4
	fxcxnsma f5,C1,e7,e5		; lfpsux	a5,3,9
	fxcxnpma f7,C1,e7,e5		; lfpsux	a9,3,9

	fpadd	f8,e8,e10		; lfpsux	a13,3,9
	fpsub	f10,e8,e10		; stfpsux	f4,4,8
	fxcxnsma f9,C1,e11,e9		; lfpsux	a2,3,9
	fxcxnpma f11,C1,e11,e9		; lfpsux	a6,3,9

	fpadd	f12,e12,e14		; lfpsux	a10,3,9
	fpsub	f14,e12,e14		; stfpsux	f8,4,8
	fxcxnsma f13,C1,e15,e13		; lfpsux	a14,3,9
	fxcxnpma f15,C1,e15,e13		; lfpsux	a3,3,9
					; lfpsux	a7,3,9
					; stfpsux	f12,4,8
					; stfpsux	f1,4,8
					; stfpsux	f5,4,8

			
1:
					; stfpsux	g9,4,8
					; stfpsux	g13,4,8
	fpadd	b0,a0,a2		; lfpsux	a11,3,9
	fpsub	b1,a0,a2		; lfpsux	a15,3,9
	fpadd	b2,a1,a3
	fpsub	b3,a1,a3

	fpadd	b4,a4,a6		; stfpsux	g2,4,8
	fpsub	b5,a4,a6
	fpadd	b6,a5,a7		; stfpsux	g6,4,8
	fpsub	b7,a5,a7

	fpadd	b8,a8,a10		; stfpsux	g10,4,8
	fpsub	b9,a8,a10
	fpadd	b10,a9,a11		; stfpsux	g14,4,8
	fpsub	b11,a9,a11

	fpadd	b12,a12,a14		; stfpsux	g3,4,8
	fpsub	b13,a12,a14
	fpadd	b14,a13,a15		; stfpsux	g7,4,8
	fpsub	b15,a13,a15

	fpadd	c0,b0,b2		; stfpsux	g11,4,8
	fpsub	c2,b0,b2		; stfpsux	g15,4,8
	fxcxnsma c1,C1,b3,b1		; lfpsx		W,0,7
	fxcxnpma c3,C1,b3,b1		; lfpsx		W2,7,8

	fpadd	c4,b4,b6
	fpsub	c6,b4,b6
	fxcxnsma c5,C1,b7,b5
	fxcxnpma c7,C1,b7,b5

	fpadd	c8,b8,b10
	fpsub	c10,b8,b10
	fxcxnsma c9,C1,b11,b9
	fxcxnpma c11,C1,b11,b9

	fpadd	c12,b12,b14
	fpsub	c14,b12,b14
	fxcxnsma c13,C1,b15,b13
	fxcxnpma c15,C1,b15,b13

	fxpmul	x5,c5,W
	fxpmul	x6,c6,W2
	fxpmul	x7,c7,W

	fxpmul	x9,c9,W2
	fxpmul	x11,W2,c11

	fxpmul	x13,c13,W
	fxpmul	x14,W2,c14
	fxpmul	x15,c15,W

	fxcxnpma d5,c5,W,x5
	fxcxnpma d6,c6,W2,x6
	fxcxnsma d7,c7,W,x7

	fxcxnpma d9,c9,W2,x9
	fxcxnsma d11,W2,c11,x11

	fxcxnsma d13,c13,W,x13
	fxcxnsma d14,W2,c14,x14
	fxcxnpma d15,c15,W,x15

	fpadd	e0,c0,c8
	fpsub	e1,c0,c8
	fpadd	e2,c4,c12
	fpsub	e3,c4,c12

	fpadd	e4,c1,d9
	fpsub	e5,c1,d9
	fxnmsub	e6,d13,C1,d5
	fxmadd	e7,d13,C1,d5

	fxcxnsma e8,C1,c10,c2
	fxcxnpma e9,C1,c10,c2
	fpsub	e10,d6,d14
	fpadd	e11,d6,d14

	fpsub	e12,c3,d11
	fpadd	e13,c3,d11
	fxnmadd	e14,d7,C1,d15
	fxnmsub	e15,d7,C1,d15

	fpadd	f0,e0,e2		; lfpsux	t4,6,8
	fpsub	f2,e0,e2		; lfpsux	t8,6,8
	fxcxnsma f1,C1,e3,e1		; lfpsux	t12,6,8
	fxcxnpma f3,C1,e3,e1

	fpadd	f4,e4,e6
	fpsub	f6,e4,e6		; stfpsux	f0,4,8
	fxcxnsma f5,C1,e7,e5
	fxcxnpma f7,C1,e7,e5

	fpadd	f8,e8,e10		; lfpsux	t1,6,8
	fpsub	f10,e8,e10		; lfpsux	t5,6,8
	fxcxnsma f9,C1,e11,e9		; lfpsux	t9,6,8
	fxcxnpma f11,C1,e11,e9

	fpadd	f12,e12,e14		; lfpsux	t13,6,8
	fpsub	f14,e12,e14		; lfpsux	t2,6,8
	fxcxnsma f13,C1,e15,e13		; lfpsux	t6,6,8
	fxcxnpma f15,C1,e15,e13		; lfpsux	t10,6,8
				
			
			
	fxpmul	 y4,f4,t4
	fxpmul	 y8,f8,t8
	fxpmul	 y12,f12,t12
	fxpmul	 y1,f1,t1
	fxpmul	 y5,f5,t5		; addi		3,3,-239*8

	fxcxnpma g4,f4,t4,y4		; lfpsux	t14,6,8
	fxcxnpma g8,f8,t8,y8		; lfpsux	t3,6,8
	fxcxnpma g12,f12,t12,y12	; lfpsux	t7,6,8
	fxcxnpma g1,f1,t1,y1		; lfpsux	t11,6,8
	fxcxnpma g5,f5,t5,y5		; lfpsux	t15,6,8

	fxpmul	 y9,f9,t9		; stfpsux	g4,4,8
	fxpmul	 y13,f13,t13		; stfpsux	g8,4,8
	fxpmul	 y2,f2,t2		; stfpsux	g12,4,8
	fxpmul	 y6,f6,t6		; stfpsux	g1,4,8
	fxpmul	 y10,f10,t10		; stfpsux	g5,4,8

	fxcxnpma g9,f9,t9,y9		; lfpsx		a0,0,3
	fxcxnpma g13,f13,t13,y13	; lfpsux	a4,3,9
	fxcxnpma g2,f2,t2,y2		; lfpsux	a8,3,9
	fxcxnpma g6,f6,t6,y6		; lfpsux	a12,3,9
	fxcxnpma g10,f10,t10,y10	; lfpsux	a1,3,9

	fxpmul	 y14,f14,t14		; lfpsux	a5,3,9
	fxpmul	 y3,f3,t3		; lfpsux	a9,3,9
	fxpmul	 y7,f7,t7		; lfpsux	a13,3,9
	fxpmul	 y11,f11,t11		; lfpsux	a2,3,9
	fxpmul	 y15,f15,t15		; lfpsux	a6,3,9

	fxcxnpma g14,f14,t14,y14	; lfpsux	a10,3,9
	fxcxnpma g3,f3,t3,y3		; lfpsux	a14,3,9
	fxcxnpma g7,f7,t7,y7		; lfpsux	a3,3,9
	fxcxnpma g11,f11,t11,y11	; lfpsux	a7,3,9
	fxcxnpma g15,f15,t15,y15	; bdnz		1b
			
					; stfpsux	g9,4,8
					; stfpsux	g13,4,8
					; stfpsux	g2,4,8
					; stfpsux	g6,4,8
					; stfpsux	g10,4,8
				
					; stfpsux	g14,4,8
					; stfpsux	g3,4,8
					; stfpsux	g7,4,8
					; stfpsux	g11,4,8
					; stfpsux	g15,4,8




	# second phase



	la	3,-255*8(4)
	mr	4,3 			; lfpsx		a0,0,3
					; lfpsux	a4,3,9
					; lfpsux	a8,3,9
					; lfpsux	a12,3,9
					; lfpsux	a1,3,9
					; lfpsux	a5,3,9
					; lfpsux	a9,3,9
	mtctr	0			; lfpsux	a13,3,9
					; lfpsux	a2,3,9
					; lfpsux	a6,3,9
					; lfpsux	a10,3,9
					; lfpsux	a14,3,9
					; lfpsux	a3,3,9
					; lfpsux	a7,3,9
					; lfpsux	a11,3,9
					; lfpsux	a15,3,9

	fpadd	b0,a0,a2
	fpsub	b1,a0,a2
	fpadd	b2,a1,a3
	fpsub	b3,a1,a3

	fpadd	b4,a4,a6
	fpsub	b5,a4,a6
	fpadd	b6,a5,a7
	fpsub	b7,a5,a7

	fpadd	b8,a8,a10
	fpsub	b9,a8,a10
	fpadd	b10,a9,a11
	fpsub	b11,a9,a11

	fpadd	b12,a12,a14
	fpsub	b13,a12,a14
	fpadd	b14,a13,a15
	fpsub	b15,a13,a15

	fpadd	c0,b0,b2
	fpsub	c2,b0,b2		;  lfpsx	W,0,7
	fxcxnsma c1,C1,b3,b1		;  lfpsx	W2,7,8
	fxcxnpma c3,C1,b3,b1

	fpadd	c4,b4,b6
	fpsub	c6,b4,b6
	fxcxnsma c5,C1,b7,b5
	fxcxnpma c7,C1,b7,b5

	fpadd	c8,b8,b10
	fpsub	c10,b8,b10
	fxcxnsma c9,C1,b11,b9
	fxcxnpma c11,C1,b11,b9

	fpadd	c12,b12,b14
	fpsub	c14,b12,b14
	fxcxnsma c13,C1,b15,b13
	fxcxnpma c15,C1,b15,b13

	fxpmul	x5,c5,W
	fxpmul	x6,c6,W2
	fxpmul	x7,c7,W

	fxpmul	x9,c9,W2
	fxpmul	x11,W2,c11

	fxpmul	x13,c13,W
	fxpmul	x14,W2,c14
	fxpmul	x15,c15,W

	fxcxnpma d5,c5,W,x5
	fxcxnpma d6,c6,W2,x6
	fxcxnsma d7,c7,W,x7

	fxcxnpma d9,c9,W2,x9
	fxcxnsma d11,W2,c11,x11

	fxcxnsma d13,c13,W,x13
	fxcxnsma d14,W2,c14,x14
	fxcxnpma d15,c15,W,x15

	fpadd	e0,c0,c8
	fpsub	e1,c0,c8
	fpadd	e2,c4,c12
	fpsub	e3,c4,c12

	fpadd	e4,c1,d9
	fpsub	e5,c1,d9
	fxnmsub	e6,d13,C1,d5
	fxmadd	e7,d13,C1,d5

	fxcxnsma e8,C1,c10,c2
	fxcxnpma e9,C1,c10,c2
	fpsub	e10,d6,d14
	fpadd	e11,d6,d14

	fpsub	e12,c3,d11
	fpadd	e13,c3,d11
	fxnmadd	e14,d7,C1,d15
	fxnmsub	e15,d7,C1,d15		; addi		3,3,-239*8

	fpadd	f0,e0,e2		; lfpsx		a0,0,3
	fpsub	f2,e0,e2		; lfpsux	a4,3,9
	fxcxnsma f1,C1,e3,e1		; lfpsux	a8,3,9
	fxcxnpma f3,C1,e3,e1		; lfpsux	a12,3,9

	fpadd	f4,e4,e6		; lfpsux	a1,3,9
	fpsub	f6,e4,e6		; stfpsx	f0,0,4
	fxcxnsma f5,C1,e7,e5		; lfpsux	a5,3,9
	fxcxnpma f7,C1,e7,e5		; lfpsux	a9,3,9

	fpadd	f8,e8,e10		; lfpsux	a13,3,9
	fpsub	f10,e8,e10		; stfpsux	f4,4,9
	fxcxnsma f9,C1,e11,e9		; lfpsux	a2,3,9
	fxcxnpma f11,C1,e11,e9		; lfpsux	a6,3,9

	fpadd	f12,e12,e14		; lfpsux	a10,3,9
	fpsub	f14,e12,e14		; stfpsux	f8,4,9
	fxcxnsma f13,C1,e15,e13		; lfpsux	a14,3,9
	fxcxnpma f15,C1,e15,e13		; lfpsux	a3,3,9

			
2:
					; stfpsux	f12,4,9
					; stfpsux	f1,4,9
					; stfpsux	f5,4,9
					; stfpsux	f9,4,9
					; stfpsux	f13,4,9
	fpadd	b0,a0,a2		; lfpsux	a7,3,9
	fpsub	b1,a0,a2		; lfpsux	a11,3,9
	fpadd	b2,a1,a3		; lfpsux	a15,3,9
	fpsub	b3,a1,a3

	fpadd	b4,a4,a6		; stfpsux	f2,4,9
	fpsub	b5,a4,a6
	fpadd	b6,a5,a7		; stfpsux	f6,4,9
	fpsub	b7,a5,a7

	fpadd	b8,a8,a10		; stfpsux	f10,4,9
	fpsub	b9,a8,a10
	fpadd	b10,a9,a11		; stfpsux	f14,4,9
	fpsub	b11,a9,a11

	fpadd	b12,a12,a14		; stfpsux	f3,4,9
	fpsub	b13,a12,a14
	fpadd	b14,a13,a15		; stfpsux	f7,4,9
	fpsub	b15,a13,a15

	fpadd	c0,b0,b2		; stfpsux	f11,4,9
	fpsub	c2,b0,b2		; stfpsux	f15,4,9
	fxcxnsma c1,C1,b3,b1		; lfpsx		W,0,7
	fxcxnpma c3,C1,b3,b1		; lfpsx		W2,7,8

	fpadd	c4,b4,b6
	fpsub	c6,b4,b6
	fxcxnsma c5,C1,b7,b5
	fxcxnpma c7,C1,b7,b5

	fpadd	c8,b8,b10
	fpsub	c10,b8,b10
	fxcxnsma c9,C1,b11,b9
	fxcxnpma c11,C1,b11,b9

	fpadd	c12,b12,b14
	fpsub	c14,b12,b14
	fxcxnsma c13,C1,b15,b13
	fxcxnpma c15,C1,b15,b13

	fxpmul	x5,c5,W
	fxpmul	x6,c6,W2
	fxpmul	x7,c7,W

	fxpmul	x9,c9,W2
	fxpmul	x11,W2,c11

	fxpmul	x13,c13,W
	fxpmul	x14,W2,c14
	fxpmul	x15,c15,W

	fxcxnpma d5,c5,W,x5
	fxcxnpma d6,c6,W2,x6
	fxcxnsma d7,c7,W,x7

	fxcxnpma d9,c9,W2,x9
	fxcxnsma d11,W2,c11,x11

	fxcxnsma d13,c13,W,x13
	fxcxnsma d14,W2,c14,x14
	fxcxnpma d15,c15,W,x15

	fpadd	e0,c0,c8
	fpsub	e1,c0,c8
	fpadd	e2,c4,c12
	fpsub	e3,c4,c12

	fpadd	e4,c1,d9
	fpsub	e5,c1,d9
	fxnmsub	e6,d13,C1,d5
	fxmadd	e7,d13,C1,d5

	fxcxnsma e8,C1,c10,c2
	fxcxnpma e9,C1,c10,c2
	fpsub	e10,d6,d14
	fpadd	e11,d6,d14

	fpsub	e12,c3,d11
	fpadd	e13,c3,d11
	fxnmadd	e14,d7,C1,d15		; addi		4,4,-239*8
	fxnmsub	e15,d7,C1,d15		; addi		3,3,-239*8

	fpadd	f0,e0,e2		; lfpsx		a0,0,3
	fpsub	f2,e0,e2		; lfpsux	a4,3,9
	fxcxnsma f1,C1,e3,e1		; lfpsux	a8,3,9
	fxcxnpma f3,C1,e3,e1		; lfpsux	a12,3,9

	fpadd	f4,e4,e6		; lfpsux	a1,3,9
	fpsub	f6,e4,e6		; stfpsx	f0,0,4
	fxcxnsma f5,C1,e7,e5		; lfpsux	a5,3,9
	fxcxnpma f7,C1,e7,e5		; lfpsux	a9,3,9

	fpadd	f8,e8,e10		; lfpsux	a13,3,9
	fpsub	f10,e8,e10		; stfpsux	f4,4,9
	fxcxnsma f9,C1,e11,e9		; lfpsux	a2,3,9
	fxcxnpma f11,C1,e11,e9		; lfpsux	a6,3,9

	fpadd	f12,e12,e14		; lfpsux	a10,3,9
	fpsub	f14,e12,e14		; stfpsux	f8,4,9
	fxcxnsma f13,C1,e15,e13		; lfpsux	a14,3,9
	fxcxnpma f15,C1,e15,e13		; lfpsux	a3,3,9
					; bdnz		2b
			
					; stfpsux	f12,4,9
					; stfpsux	f1,4,9
					; stfpsux	f5,4,9
					; stfpsux	f9,4,9
					; stfpsux	f13,4,9
					; stfpsux	f2,4,9
					; stfpsux	f6,4,9
					; stfpsux	f10,4,9
					; stfpsux	f14,4,9
					; stfpsux	f3,4,9
					; stfpsux	f7,4,9
					; stfpsux	f11,4,9
					; stfpsux	f15,4,9


	li	9,16			; lfpdx		31,0,1
					; lfpdux	30,1,9
					; lfpdux	29,1,9
					; lfpdux	28,1,9
					; lfpdux	27,1,9
					; lfpdux	26,1,9
					; lfpdux	25,1,9
					; lfpdux	24,1,9
					; lfpdux	23,1,9
					; lfpdux	22,1,9
					; lfpdux	21,1,9
					; lfpdux	20,1,9
					; lfpdux	19,1,9
					; lfpdux	18,1,9
					; lfpdux	17,1,9
					; lfpdux	16,1,9
					; lfpdux	15,1,9
					; lfpdux	14,1,9

	addi	1,1,16			; blr

#endif
