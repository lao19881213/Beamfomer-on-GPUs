#if defined HAVE_BGP

.global	_apply_chirp
_apply_chirp:

	li	12,-16		# push call-saved registers
	stfpdux	14,1,12
	stfpdux	15,1,12
	stfpdux	16,1,12
	stfpdux	17,1,12
	stfpdux	18,1,12
	stfpdux	19,1,12

	srwi	6,6,2
	subi	6,6,1
	mtctr	6
	li	8,8
	mr	9,3
	subi	10,4,8

	lfpsx	0,0,3
	lfpsx	4,0,4
	lfpsx	8,0,5

	lfpsux	1,3,8
	lfpsux	5,4,8
	lfpsux	9,5,8

	lfpsux	2,3,8
	fxpmul	12,0,8
	lfpsux	6,4,8
	fxpmul	16,4,8
	lfpsux	10,5,8

	lfpsux	3,3,8
	fxpmul	13,1,9
	lfpsux	7,4,8
	fxpmul	17,5,9
	lfpsux	11,5,8
	fxcxnpma 12,0,8,12
	fxcxnpma 16,4,8,16

	lfpsux	0,3,8
	fxpmul	14,2,10
	lfpsux	4,4,8
	fxpmul	18,6,10
	lfpsux	8,5,8
	fxcxnpma 13,1,9,13
	stfpsx	12,0,9
	fxcxnpma 17,5,9,17

0:
	lfpsux	1,3,8
	stfpsux	16,10,8
	fxpmul	15,3,11
	lfpsux	5,4,8
	fxpmul	19,7,11
	lfpsux	9,5,8
	fxcxnpma 14,2,10,14
	stfpsux	13,9,8
	fxcxnpma 18,6,10,18

	lfpsux	2,3,8
	stfpsux	17,10,8
	fxpmul	12,0,8
	lfpsux	6,4,8
	fxpmul	16,4,8
	lfpsux	10,5,8
	fxcxnpma 15,3,11,15
	stfpsux	14,9,8
	fxcxnpma 19,7,11,19

	lfpsux	3,3,8
	stfpsux	18,10,8
	fxpmul	13,1,9
	lfpsux	7,4,8
	fxpmul	17,5,9
	lfpsux	11,5,8
	fxcxnpma 12,0,8,12
	stfpsux	15,9,8
	fxcxnpma 16,4,8,16

	lfpsux	0,3,8
	stfpsux	19,10,8
	fxpmul	14,2,10
	lfpsux	4,4,8
	fxpmul	18,6,10
	lfpsux	8,5,8
	fxcxnpma 13,1,9,13
	stfpsux	12,9,8
	fxcxnpma 17,5,9,17

	bdnz	0b

	stfpsux	16,10,8
	fxpmul	15,3,11
	fxpmul	19,7,11
	fxcxnpma 14,2,10,14
	stfpsux	13,9,8
	fxcxnpma 18,6,10,18
	stfpsux	17,10,8

	fxcxnpma 15,3,11,15
	stfpsux	14,9,8
	fxcxnpma 19,7,11,19
	stfpsux	18,10,8

	stfpsux	15,9,8
	stfpsux	19,10,8

	li	12,16		# restore call-saved registers
	lfpdx	19,0,1
	lfpdux	18,1,12
	lfpdux	17,1,12
	lfpdux	16,1,12
	lfpdux	15,1,12
	lfpdux	14,1,12

	addi	1,1,16		# reset stack pointer

	blr
#endif
