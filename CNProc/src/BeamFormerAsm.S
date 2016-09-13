#if defined HAVE_BGP

.align	5

.global	_beamform_3stations_6beams
_beamform_3stations_6beams:

	srwi	9,9,1
	addi	9,9,-1
	mtctr	9

	subi	 1,1,12*4	# push call-saved registers
	li	 0,-16

	stmw	 20,0(1)

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

	mr	23,5
	add	24,23,6
	add	25,24,6

	mr	26,3
	add	27,26,4
	add	28,27,4
	add	29,28,4
	add	30,29,4
	add	31,30,4

	li	0,8
	addi	11,8,-5*8

	lfpsx	0,0,7
	lfpsux	1,7,0
	lfpsux	2,7,0
	lfpsux	3,7,0
	lfpsux	4,7,0
	lfpsux	5,7,0
	lfpsux	6,7,11
	lfpsux	7,7,0
	lfpsux	8,7,0
	lfpsux	9,7,0
	lfpsux	10,7,0
	lfpsux	11,7,0
	lfpsux	12,7,11
	lfpsux	13,7,0
	lfpsux	14,7,0
	lfpsux	15,7,0
	lfpsux	16,7,0
	lfpsux	17,7,0

	li	8,-8
	li	9,32
	cmpwi	0,10,0
	beq	not_first_time_3_6

#define NR_STATIONS 3
#define FIRST_TIME
#include "BeamFormerAsm3St6Bm.inc"
	b	epilogue_3_6

not_first_time_3_6:

#undef FIRST_TIME
#include "BeamFormerAsm3St6Bm.inc"
#undef NR_STATIONS

epilogue_3_6:

	li	0,16		# restore call-saved registers
	lfpdx	31,0,1
	lfpdux	30,1,0
	lfpdux	29,1,0
	lfpdux	28,1,0
	lfpdux	27,1,0
	lfpdux	26,1,0
	lfpdux	25,1,0
	lfpdux	24,1,0
	lfpdux	23,1,0
	lfpdux	22,1,0
	lfpdux	21,1,0
	lfpdux	20,1,0
	lfpdux	19,1,0
	lfpdux	18,1,0
	lfpdux	17,1,0
	lfpdux	16,1,0
	lfpdux	15,1,0
	lfpdux	14,1,0

	lmw	20,16(1)
	addi	1,1,12*4+16	# reset stack pointer

	blr


.align	5

.global	_beamform_up_to_6_stations_and_3_beams
_beamform_up_to_6_stations_and_3_beams:

	lwz	11,8(1)		
	slwi	10,10,2
	slwi	11,11,3
	add	11,11,10
	lwz	10,12(1)		
	addis	11,11,jump_table@ha
	slwi	10,10,6
	add	11,11,10
	lwz	11,jump_table@l(11)

	srwi	9,9,1
	addi	9,9,-1

	subi	 1,1,12*4	# push call-saved registers
	li	 0,-16

	stmw	 20,0(1)

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

	mr	21,3
	add	23,21,4
	add	25,23,4

	li	0,8

	mtctr	11
	bctr

first_time_1_1:

#define NR_BEAMS    1
#define NR_STATIONS 1
#define FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue

not_first_time_1_1:

#undef FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue
#undef NR_STATIONS

first_time_2_1:

#define NR_STATIONS 2
#define FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue

not_first_time_2_1:

#undef FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue
#undef NR_STATIONS

first_time_3_1:

#define NR_STATIONS 3
#define FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue

not_first_time_3_1:

#undef FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue
#undef NR_STATIONS

first_time_4_1:

#define NR_STATIONS 4
#define FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue

not_first_time_4_1:

#undef FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue
#undef NR_STATIONS

first_time_5_1:

#define NR_STATIONS 5
#define FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue

not_first_time_5_1:

#undef FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue
#undef NR_STATIONS

first_time_6_1:

#define NR_STATIONS 6
#define FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue

not_first_time_6_1:

#undef FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
#undef NR_STATIONS
#undef NR_BEAMS
	b	epilogue

first_time_1_2:

#define NR_BEAMS    2
#define NR_STATIONS 1
#define FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue

not_first_time_1_2:

#undef FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue
#undef NR_STATIONS

first_time_2_2:

#define NR_STATIONS 2
#define FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue

not_first_time_2_2:

#undef FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue
#undef NR_STATIONS

first_time_3_2:

#define NR_STATIONS 3
#define FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue

not_first_time_3_2:

#undef FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue
#undef NR_STATIONS

first_time_4_2:

#define NR_STATIONS 4
#define FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue

not_first_time_4_2:

#undef FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue
#undef NR_STATIONS

first_time_5_2:

#define NR_STATIONS 5
#define FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue

not_first_time_5_2:

#undef FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue
#undef NR_STATIONS

first_time_6_2:

#define NR_STATIONS 6
#define FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue

not_first_time_6_2:

#undef FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
#undef NR_STATIONS
#undef NR_BEAMS
	b	epilogue

first_time_1_3:

#define NR_BEAMS    3
#define NR_STATIONS 1
#define FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue

not_first_time_1_3:

#undef FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue
#undef NR_STATIONS

first_time_2_3:

#define NR_STATIONS 2
#define FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue

not_first_time_2_3:

#undef FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue
#undef NR_STATIONS

first_time_3_3:

#define NR_STATIONS 3
#define FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue

not_first_time_3_3:

#undef FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue
#undef NR_STATIONS

first_time_4_3:

#define NR_STATIONS 4
#define FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue

not_first_time_4_3:

#undef FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue
#undef NR_STATIONS

first_time_5_3:

#define NR_STATIONS 5
#define FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue

not_first_time_5_3:

#undef FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue
#undef NR_STATIONS

first_time_6_3:

#define NR_STATIONS 6
#define FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
	b	epilogue

not_first_time_6_3:

#undef FIRST_TIME
#include "BeamFormerAsm6St3Bm.inc"
#undef NR_STATIONS
#undef NR_BEAMS

epilogue:

	li	0,16		# restore call-saved registers
	lfpdx	31,0,1
	lfpdux	30,1,0
	lfpdux	29,1,0
	lfpdux	28,1,0
	lfpdux	27,1,0
	lfpdux	26,1,0
	lfpdux	25,1,0
	lfpdux	24,1,0
	lfpdux	23,1,0
	lfpdux	22,1,0
	lfpdux	21,1,0
	lfpdux	20,1,0
	lfpdux	19,1,0
	lfpdux	18,1,0
	lfpdux	17,1,0
	lfpdux	16,1,0
	lfpdux	15,1,0
	lfpdux	14,1,0

	lmw	20,16(1)
	addi	1,1,12*4+16	# reset stack pointer

	blr

jump_table:
	.long	illegal, illegal
	.long	illegal, illegal
	.long	illegal, illegal
	.long	illegal, illegal
	.long	illegal, illegal
	.long	illegal, illegal
	.long	illegal, illegal
	.long	illegal, illegal
	.long	illegal, illegal
	.long	not_first_time_1_1, first_time_1_1
	.long	not_first_time_2_1, first_time_2_1
	.long	not_first_time_3_1, first_time_3_1
	.long	not_first_time_4_1, first_time_4_1
	.long	not_first_time_5_1, first_time_5_1
	.long	not_first_time_6_1, first_time_6_1
	.long	illegal, illegal
	.long	illegal, illegal
	.long	not_first_time_1_2, first_time_1_2
	.long	not_first_time_2_2, first_time_2_2
	.long	not_first_time_3_2, first_time_3_2
	.long	not_first_time_4_2, first_time_4_2
	.long	not_first_time_5_2, first_time_5_2
	.long	not_first_time_6_2, first_time_6_2
	.long	illegal, illegal
	.long	illegal, illegal
	.long	not_first_time_1_3, first_time_1_3
	.long	not_first_time_2_3, first_time_2_3
	.long	not_first_time_3_3, first_time_3_3
	.long	not_first_time_4_3, first_time_4_3
	.long	not_first_time_5_3, first_time_5_3
	.long	not_first_time_6_3, first_time_6_3
	.long	illegal, illegal

illegal:
	trap
	b	epilogue

#endif
