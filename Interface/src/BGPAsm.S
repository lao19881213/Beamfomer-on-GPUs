#if defined HAVE_BGP

.global	_add_2_single_precision_vectors
_add_2_single_precision_vectors:

	li	 0,-16
	stfpdux  14,1,0
	stfpdux  15,1,0

	srwi	 6,6,4
	li	 11,8
	subi	 6,6,1
	mtctr	 6

	cmpwi	 0,6,0
	lfpsx	 0,0,4
	lfpsx	 1,0,5
	lfpsux	 2,4,11
	lfpsux	 3,5,11
	lfpsux	 4,4,11
	lfpsux	 5,5,11
	lfpsux	 6,4,11
	lfpsux	 7,5,11
	lfpsux	 8,4,11
	lfpsux	 9,5,11
	lfpsux	 10,4,11
	lfpsux	 11,5,11
	fpadd	 0,0,1
	lfpsux	 12,4,11
	lfpsux	 13,5,11
	fpadd	 2,2,3
	lfpsux	 14,4,11
	lfpsux	 15,5,11
	stfpsx	 0,0,3
	fpadd	 4,4,5
	beq-	 0,1f

0:	lfpsux	 0,4,11
	lfpsux	 1,5,11
	stfpsux	 2,3,11
	fpadd	 6,6,7
	lfpsux	 2,4,11
	lfpsux	 3,5,11
	stfpsux	 4,3,11
	fpadd	 8,8,9
	lfpsux	 4,4,11
	lfpsux	 5,5,11
	stfpsux	 6,3,11
	fpadd	 10,10,11
	lfpsux	 6,4,11
	lfpsux	 7,5,11
	stfpsux	 8,3,11
	fpadd	 12,12,13
	lfpsux	 8,4,11
	lfpsux	 9,5,11
	stfpsux	 10,3,11
	fpadd	 14,14,15
	lfpsux	 10,4,11
	lfpsux	 11,5,11
	stfpsux	 12,3,11
	fpadd	 0,0,1
	lfpsux	 12,4,11
	lfpsux	 13,5,11
	stfpsux	 14,3,11
	fpadd	 2,2,3
	lfpsux	 14,4,11
	lfpsux	 15,5,11
	stfpsux	 0,3,11
	fpadd	 4,4,5
	bdnz	 0b

1:	stfpsux	 2,3,11
	fpadd	 6,6,7
	stfpsux	 4,3,11
	fpadd	 8,8,9
	stfpsux	 6,3,11
	fpadd	 10,10,11
	stfpsux	 8,3,11
	fpadd	 12,12,13
	stfpsux	 10,3,11
	fpadd	 14,14,15
	stfpsux	 12,3,11
	stfpsux	 14,3,11

	li	 0,16		# restore call-saved registers
	lfpdx	 15,0,1
	lfpdux	 14,1,0
	addi	 1,1,16

	blr


.global	_add_3_single_precision_vectors
_add_3_single_precision_vectors:

	li	 0,-16
	stfpdux  14,1,0
	stfpdux  15,1,0
	stfpdux  16,1,0
	stfpdux  17,1,0
	stfpdux  18,1,0
	stfpdux  19,1,0
	stfpdux  20,1,0
	stfpdux  21,1,0
	stfpdux  22,1,0
	stfpdux  23,1,0

	srwi	 7,7,4
	li	 11,8
	subi	 7,7,1
	mtctr	 7

	cmpwi	 0,7,0
	lfpsx	 0,0,4
	lfpsx	 1,0,5
	lfpsx	 2,0,6
	lfpsux	 3,4,11
	lfpsux	 4,5,11
	lfpsux	 5,6,11
	lfpsux	 6,4,11
	lfpsux	 7,5,11
	lfpsux	 8,6,11
	lfpsux	 9,4,11
	lfpsux	 10,5,11
	lfpsux	 11,6,11
	lfpsux	 12,4,11
	lfpsux	 13,5,11
	lfpsux	 14,6,11
	lfpsux	 15,4,11
	lfpsux	 16,5,11
	fpadd	 0,0,1
	lfpsux	 17,6,11
	lfpsux	 18,4,11
	lfpsux	 19,5,11
	fpadd	 3,3,4
	lfpsux	 20,6,11
	fpadd	 0,0,2
	lfpsux	 21,4,11
	lfpsux	 22,5,11
	fpadd	 6,6,7
	lfpsux	 23,6,11
	fpadd	 3,3,5
	stfpsx	 0,0,3
	beq-	 0,1f
	li	 10,32

0:	lfpsux	 0,4,11
	lfpsux	 1,5,11
	dcbt	 4,10
	fpadd	 9,9,10
	lfpsux	 2,6,11
	fpadd	 6,6,8
	stfpsux	 3,3,11
	lfpsux	 3,4,11
	lfpsux	 4,5,11
	fpadd	 12,12,13
	lfpsux	 5,6,11
	fpadd	 9,9,11
	stfpsux	 6,3,11
	lfpsux	 6,4,11
	lfpsux	 7,5,11
	fpadd	 15,15,16
	lfpsux	 8,6,11
	fpadd	 12,12,14
	stfpsux	 9,3,11
	lfpsux	 9,4,11
	lfpsux	 10,5,11
	fpadd	 18,18,19
	lfpsux	 11,6,11
	fpadd	 15,15,17
	stfpsux	 12,3,11
	lfpsux	 12,4,11
	lfpsux	 13,5,11
	fpadd	 21,21,22
	lfpsux	 14,6,11
	fpadd	 18,18,20
	stfpsux	 15,3,11
	lfpsux	 15,4,11
	lfpsux	 16,5,11
	fpadd	 0,0,1
	lfpsux	 17,6,11
	fpadd	 21,21,23
	stfpsux	 18,3,11
	lfpsux	 18,4,11
	lfpsux	 19,5,11
	fpadd	 3,3,4
	lfpsux	 20,6,11
	fpadd	 0,0,2
	stfpsux	 21,3,11
	lfpsux	 21,4,11
	lfpsux	 22,5,11
	fpadd	 6,6,7
	lfpsux	 23,6,11
	fpadd	 3,3,5
	stfpsux	 0,3,11
	bdnz	 0b

1:	fpadd	 9,9,10
	fpadd	 6,6,8
	stfpsux	 3,3,11
	fpadd	 12,12,13
	fpadd	 9,9,11
	stfpsux	 6,3,11
	fpadd	 15,15,16
	fpadd	 12,12,14
	stfpsux	 9,3,11
	fpadd	 18,18,19
	fpadd	 15,15,17
	stfpsux	 12,3,11
	fpadd	 21,21,22
	fpadd	 18,18,20
	stfpsux	 15,3,11
	fpadd	 21,21,23
	stfpsux	 18,3,11
	stfpsux	 21,3,11

	li	 0,16		# restore call-saved registers
	lfpdx	 23,0,1
	lfpdux	 22,1,0
	lfpdux	 21,1,0
	lfpdux	 20,1,0
	lfpdux	 19,1,0
	lfpdux	 18,1,0
	lfpdux	 17,1,0
	lfpdux	 16,1,0
	lfpdux	 15,1,0
	lfpdux	 14,1,0
	addi	 1,1,16

	blr


.global	_add_4_single_precision_vectors
_add_4_single_precision_vectors:

	li	 0,-16
	stfpdux  14,1,0
	stfpdux  15,1,0
	stfpdux  16,1,0
	stfpdux  17,1,0
	stfpdux  18,1,0
	stfpdux  19,1,0
	stfpdux  20,1,0
	stfpdux  21,1,0
	stfpdux  22,1,0
	stfpdux  23,1,0
	stfpdux  24,1,0
	stfpdux  25,1,0
	stfpdux  26,1,0
	stfpdux  27,1,0
	stfpdux  28,1,0
	stfpdux  29,1,0
	stfpdux  30,1,0
	stfpdux  31,1,0

	srwi	 8,8,4
	li	 11,8
	subi	 8,8,1
	mtctr	 8

	cmpwi	 0,8,0
	lfpsx	 0,0,4
	lfpsx	 1,0,5
	lfpsx	 2,0,6
	lfpsx	 3,0,7
	lfpsux	 4,4,11
	lfpsux	 5,5,11
	lfpsux	 6,6,11
	lfpsux	 7,7,11
	lfpsux	 8,4,11
	lfpsux	 9,5,11
	lfpsux	 10,6,11
	lfpsux	 11,7,11
	lfpsux	 12,4,11
	lfpsux	 13,5,11
	lfpsux	 14,6,11
	lfpsux	 15,7,11
	lfpsux	 16,4,11
	lfpsux	 17,5,11
	lfpsux	 18,6,11
	lfpsux	 19,7,11
	lfpsux	 20,4,11
	lfpsux	 21,5,11
	fpadd	 0,0,1
	fpadd	 2,2,3
	lfpsux	 22,6,11
	lfpsux	 23,7,11
	lfpsux	 24,4,11
	lfpsux	 25,5,11
	fpadd	 4,4,5
	fpadd	 6,6,7
	lfpsux	 26,6,11
	lfpsux	 27,7,11
	fpadd	 0,0,2
	lfpsux	 28,4,11
	lfpsux	 29,5,11
	fpadd	 8,8,9
	fpadd	 10,10,11
	lfpsux	 30,6,11
	lfpsux	 31,7,11

	stfpsx	 0,0,3
	fpadd	 4,4,6
	beq-	 0,1f

0:	lfpsux	 0,4,11
	lfpsux	 1,5,11
	fpadd	 12,12,13
	fpadd	 14,14,15
	lfpsux	 2,6,11
	lfpsux	 3,7,11

	stfpsux	 4,3,11
	fpadd	 8,8,10

	lfpsux	 4,4,11
	lfpsux	 5,5,11
	fpadd	 16,16,17
	fpadd	 18,18,19
	lfpsux	 6,6,11
	lfpsux	 7,7,11

	stfpsux	 8,3,11
	fpadd	 12,12,14

	lfpsux	 8,4,11
	lfpsux	 9,5,11
	fpadd	 20,20,21
	fpadd	 22,22,23
	lfpsux	 10,6,11
	lfpsux	 11,7,11

	stfpsux	 12,3,11
	fpadd	 16,16,18

	lfpsux	 12,4,11
	lfpsux	 13,5,11
	fpadd	 24,24,25
	fpadd	 26,26,27
	lfpsux	 14,6,11
	lfpsux	 15,7,11

	stfpsux	 16,3,11
	fpadd	 20,20,22

	lfpsux	 16,4,11
	lfpsux	 17,5,11
	fpadd	 28,28,29
	fpadd	 30,30,31
	lfpsux	 18,6,11
	lfpsux	 19,7,11

	stfpsux	 20,3,11
	fpadd	 24,24,26

	lfpsux	 20,4,11
	lfpsux	 21,5,11
	fpadd	 0,0,1
	fpadd	 2,2,3
	lfpsux	 22,6,11
	lfpsux	 23,7,11

	stfpsux	 24,3,11
	fpadd	 28,28,30

	lfpsux	 24,4,11
	lfpsux	 25,5,11
	fpadd	 4,4,5
	fpadd	 6,6,7
	lfpsux	 26,6,11
	lfpsux	 27,7,11

	stfpsux	 28,3,11
	fpadd	 0,0,2

	lfpsux	 28,4,11
	lfpsux	 29,5,11
	fpadd	 8,8,9
	fpadd	 10,10,11
	lfpsux	 30,6,11
	lfpsux	 31,7,11

	stfpsux	 0,3,11
	fpadd	 4,4,6

	bdnz	 0b

1:	fpadd	 12,12,13
	fpadd	 14,14,15

	stfpsux	 4,3,11
	fpadd	 8,8,10

	fpadd	 16,16,17
	fpadd	 18,18,19

	stfpsux	 8,3,11
	fpadd	 12,12,14

	fpadd	 20,20,21
	fpadd	 22,22,23

	stfpsux	 12,3,11
	fpadd	 16,16,18

	fpadd	 24,24,25
	fpadd	 26,26,27

	stfpsux	 16,3,11
	fpadd	 20,20,22

	lfpsux	 16,4,11
	lfpsux	 17,5,11
	fpadd	 28,28,29
	fpadd	 30,30,31
	stfpsux	 20,3,11
	fpadd	 24,24,26
	stfpsux	 24,3,11
	fpadd	 28,28,30
	stfpsux	 28,3,11

	li	 0,16		# restore call-saved registers
	lfpdx	 31,0,1
	lfpdux	 30,1,0
	lfpdux	 29,1,0
	lfpdux	 28,1,0
	lfpdux	 27,1,0
	lfpdux	 26,1,0
	lfpdux	 25,1,0
	lfpdux	 24,1,0
	lfpdux	 23,1,0
	lfpdux	 22,1,0
	lfpdux	 21,1,0
	lfpdux	 20,1,0
	lfpdux	 19,1,0
	lfpdux	 18,1,0
	lfpdux	 17,1,0
	lfpdux	 16,1,0
	lfpdux	 15,1,0
	lfpdux	 14,1,0
	addi	 1,1,16

	blr


.global	_add_5_single_precision_vectors
_add_5_single_precision_vectors:

	li	 0,-16
	stfpdux  14,1,0
	stfpdux  15,1,0
	stfpdux  16,1,0
	stfpdux  17,1,0
	stfpdux  18,1,0
	stfpdux  19,1,0
	stfpdux  24,1,0
	stfpdux  25,1,0
	stfpdux  26,1,0
	stfpdux  27,1,0
	stfpdux  28,1,0
	stfpdux  29,1,0

	srwi	 9,9,3
	li	 11,8
	subi	 9,9,1
	mtctr	 9
	subi	 3,3,8

	cmpwi	 0,9,0

	lfpsx	 0,0,4
	lfpsx	 1,0,5
	lfpsx	 2,0,6
	lfpsx	 3,0,7
	lfpsx	 4,0,8

	lfpsux	 5,4,11
	lfpsux	 6,5,11
	lfpsux	 7,6,11
	lfpsux	 8,7,11
	lfpsux	 9,8,11

	lfpsux	 10,4,11
	lfpsux	 11,5,11
	lfpsux	 12,6,11
	lfpsux	 13,7,11
	lfpsux	 14,8,11
	fpadd	 24,0,1
	fpadd	 25,2,3

	lfpsux	 15,4,11
	lfpsux	 16,5,11
	lfpsux	 17,6,11
	fpadd	 24,24,25
	lfpsux	 18,7,11
	lfpsux	 19,8,11
	fpadd	 27,5,6
	fpadd	 28,7,8

	beq-	 0,1f
	li	 10,48

0:
	dcbt	 4,10
	lfpsux	 0,4,11
	fpadd	 24,24,4
	lfpsux	 1,5,11
	lfpsux	 2,6,11
	fpadd	 27,27,28
	lfpsux	 3,7,11
	lfpsux	 4,8,11
	stfpsux	 24,3,11
	fpadd	 24,10,11
	dcbt	 5,10
	fpadd	 25,12,13

	lfpsux	 5,4,11
	fpadd	 27,27,9
	lfpsux	 6,5,11
	lfpsux	 7,6,11
	fpadd	 24,24,25
	lfpsux	 8,7,11
	lfpsux	 9,8,11
	stfpsux	 27,3,11
	fpadd	 27,15,16
	dcbt	 6,10
	fpadd	 28,17,18

	lfpsux	 10,4,11
	fpadd	 24,24,14
	lfpsux	 11,5,11
	lfpsux	 12,6,11
	fpadd	 27,27,28
	lfpsux	 13,7,11
	lfpsux	 14,8,11
	stfpsux	 24,3,11
	fpadd	 24,0,1
	dcbt	 7,10
	fpadd	 25,2,3

	lfpsux	 15,4,11
	fpadd	 27,27,19
	lfpsux	 16,5,11
	lfpsux	 17,6,11
	fpadd	 24,24,25
	dcbt	 8,10
	lfpsux	 18,7,11
	lfpsux	 19,8,11
	stfpsux	 27,3,11
	fpadd	 27,5,6
	fpadd	 28,7,8

	bdnz	 0b

1:
	fpadd	 24,24,4
	fpadd	 27,27,28
	stfpsux	 24,3,11
	fpadd	 24,10,11
	fpadd	 25,12,13

	fpadd	 27,27,9
	fpadd	 24,24,25
	stfpsux	 27,3,11
	fpadd	 27,15,16
	fpadd	 28,17,18

	fpadd	 24,24,14
	fpadd	 27,27,28
	stfpsux	 24,3,11

	fpadd	 27,27,19
	stfpsux	 27,3,11

	li	 0,16		# restore call-saved registers
	lfpdx	 29,0,1
	lfpdux	 28,1,0
	lfpdux	 27,1,0
	lfpdux	 26,1,0
	lfpdux	 25,1,0
	lfpdux	 24,1,0
	lfpdux	 19,1,0
	lfpdux	 18,1,0
	lfpdux	 17,1,0
	lfpdux	 16,1,0
	lfpdux	 15,1,0
	lfpdux	 14,1,0
	addi	 1,1,16
	blr


.global	_add_6_single_precision_vectors
_add_6_single_precision_vectors:

	li	 0,-16
	stfpdux  14,1,0
	stfpdux  15,1,0
	stfpdux  16,1,0
	stfpdux  17,1,0
	stfpdux  18,1,0
	stfpdux  19,1,0
	stfpdux  20,1,0
	stfpdux  21,1,0
	stfpdux  22,1,0
	stfpdux  23,1,0
	stfpdux  24,1,0
	stfpdux  25,1,0
	stfpdux  26,1,0
	stfpdux  27,1,0
	stfpdux  28,1,0
	stfpdux  29,1,0

	srwi	 10,10,3
	li	 11,8
	subi	 10,10,1
	mtctr	 10
	subi	 3,3,8

	cmpwi	 0,10,0

	lfpsx	 0,0,4
	lfpsx	 1,0,5
	lfpsx	 2,0,6
	lfpsx	 3,0,7
	lfpsx	 4,0,8
	lfpsx	 5,0,9

	lfpsux	 6,4,11
	lfpsux	 7,5,11
	lfpsux	 8,6,11
	lfpsux	 9,7,11
	lfpsux	 10,8,11
	lfpsux	 11,9,11

	lfpsux	 12,4,11
	lfpsux	 13,5,11
	lfpsux	 14,6,11
	lfpsux	 15,7,11
	lfpsux	 16,8,11
	lfpsux	 17,9,11
	fpadd	 24,0,1
	fpadd	 25,2,3

	lfpsux	 18,4,11
	lfpsux	 19,5,11
	lfpsux	 20,6,11
	fpadd	 26,4,5
	fpadd	 24,24,25
	lfpsux	 21,7,11
	lfpsux	 22,8,11
	lfpsux	 23,9,11
	fpadd	 27,6,7
	fpadd	 28,8,9

	beq-	 0,1f
	li	 10,48

0:
	dcbt	 4,10
	lfpsux	 0,4,11
	fpadd	 24,24,26
	lfpsux	 1,5,11
	lfpsux	 2,6,11
	fpadd	 29,10,11
	fpadd	 27,27,28
	lfpsux	 3,7,11
	lfpsux	 4,8,11
	lfpsux	 5,9,11
	dcbt	 5,10
	stfpsux	 24,3,11
	fpadd	 24,12,13
	fpadd	 25,14,15

	lfpsux	 6,4,11
	fpadd	 27,27,29
	lfpsux	 7,5,11
	lfpsux	 8,6,11
	dcbt	 6,10
	fpadd	 26,16,17
	fpadd	 24,24,25
	lfpsux	 9,7,11
	lfpsux	 10,8,11
	lfpsux	 11,9,11
	stfpsux	 27,3,11
	fpadd	 27,18,19
	fpadd	 28,20,21

	dcbt	 7,10
	lfpsux	 12,4,11
	fpadd	 24,24,26
	lfpsux	 13,5,11
	lfpsux	 14,6,11
	fpadd	 29,22,23
	fpadd	 27,27,28
	lfpsux	 15,7,11
	lfpsux	 16,8,11
	lfpsux	 17,9,11
	dcbt	 8,10
	stfpsux	 24,3,11
	fpadd	 24,0,1
	fpadd	 25,2,3

	lfpsux	 18,4,11
	fpadd	 27,27,29
	lfpsux	 19,5,11
	lfpsux	 20,6,11
	dcbt	 9,10
	fpadd	 26,4,5
	fpadd	 24,24,25
	lfpsux	 21,7,11
	lfpsux	 22,8,11
	lfpsux	 23,9,11
	stfpsux	 27,3,11
	fpadd	 27,6,7
	fpadd	 28,8,9

	bdnz	 0b

1:
	fpadd	 24,24,26
	fpadd	 29,10,11
	fpadd	 27,27,28
	stfpsux	 24,3,11
	fpadd	 24,12,13
	fpadd	 25,14,15

	fpadd	 27,27,29
	fpadd	 26,16,17
	fpadd	 24,24,25
	stfpsux	 27,3,11
	fpadd	 27,18,19
	fpadd	 28,20,21

	fpadd	 24,24,26
	fpadd	 29,22,23
	fpadd	 27,27,28
	stfpsux	 24,3,11

	fpadd	 27,27,29
	stfpsux	 27,3,11

	li	 0,16		# restore call-saved registers
	lfpdx	 29,0,1
	lfpdux	 28,1,0
	lfpdux	 27,1,0
	lfpdux	 26,1,0
	lfpdux	 25,1,0
	lfpdux	 24,1,0
	lfpdux	 23,1,0
	lfpdux	 22,1,0
	lfpdux	 21,1,0
	lfpdux	 20,1,0
	lfpdux	 19,1,0
	lfpdux	 18,1,0
	lfpdux	 17,1,0
	lfpdux	 16,1,0
	lfpdux	 15,1,0
	lfpdux	 14,1,0
	addi	 1,1,16
	blr

#endif
