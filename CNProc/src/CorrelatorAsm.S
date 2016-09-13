#  Correlator.S: correlator assembly for BG/L double Hummer
#
#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#  $Id: CorrelatorAsm.S 18315 2011-06-22 13:43:35Z romein $

#if defined HAVE_BGP
#define CN_PROCESSING
#include <Interface/Config.h>

# C[s1][s2][p1][p2][ch] = sum over t of A[ch][s1][p1][t] * ~ A[ch][s2][p2][t]
# Cr = Ar*Br+Ai*Bi, Ci = Ai*Br-Ar*Bi

#	fxcpnsma 1,16,21,1	# f1r += f21r*f16r, f1i -= f21i*f16r
#	fxcxma   1,16,21,1	# f1r += f21i*f16i, f1i += f21r*f16i


.align	5

zero:	.long	0,0
one:	.long	0x3f800000,0x3f800000

.global _correlate_2x2
_correlate_2x2:

#	computes correlations of S0*~S2,S0*~S3,S1*~S2,S1*~S3
#	r3 :	pointer to S0 data
#	r4 :	pointer to S1 data
#	r5 :	pointer to S2 data
#	r6 :	pointer to S3 data
#	r7 :	pointer to S0*~S2 output
#	r8 :	pointer to S0*~S3 output
#	r9 :	pointer to S1*~S2 output
#	r10:	pointer to S1*~S3 output
#	8(r1):	nr_samples_to_integrate

#	local variables:
#	r11:	temporary
#	f0 :	sum of S0_X * ~S2_X
#	f1 :	sum of S0_X * ~S2_Y
#	f2 :	sum of S0_Y * ~S2_X
#	f3 :	sum of S0_Y * ~S2_Y
#	f4 :	sum of S0_X * ~S3_X
#	f5 :	sum of S0_X * ~S3_Y
#	f6 :	sum of S0_Y * ~S3_X
#	f7 :	sum of S0_Y * ~S3_Y
#	f8 :	sum of S1_X * ~S2_X
#	f9 :	sum of S1_X * ~S2_Y
#	f10:	sum of S1_Y * ~S2_X
#	f11:	sum of S1_Y * ~S2_Y
#	f12:	sum of S1_X * ~S3_X
#	f13:	sum of S1_X * ~S3_Y
#	f14:	sum of S1_Y * ~S3_X
#	f15:	sum of S1_Y * ~S3_Y
#	f16,f24:S0_X (sample from station S0, X polarization)
#	f17,f25:S0_Y
#	f18,f26:S1_X
#	f19,f27:S1_Y
#	f20,f28:S2_X
#	f21,f29:S2_Y
#	f22,f30:S3_X
#	f23,f31:S3_Y

	lis	 12,zero@ha	; lwz		 0,8(1)
	addi	 12,12,zero@l
				; lfpsx	 0,0,12

	li	 12,-16		# push call-saved registers
				; stfpdux	 14,1,12
				; stfpdux	 15,1,12
				; stfpdux	 16,1,12
				; stfpdux	 17,1,12
				; stfpdux	 18,1,12
				; stfpdux	 19,1,12
				; stfpdux	 20,1,12
				; stfpdux	 21,1,12
				; stfpdux	 22,1,12
				; stfpdux	 23,1,12
				; stfpdux	 24,1,12
				; stfpdux	 25,1,12
	li	 11,8		; stfpdux	 26,1,12
	srwi	 0,0,1		; stfpdux	 27,1,12
	mtctr	 0		; stfpdux	 28,1,12
	fpmr	 1,0		; stfpdux	 29,1,12
	fpmr	 2,0		; stfpdux	 30,1,12
	fpmr	 3,0		; stfpdux	 31,1,12

	fpmr	 4,0		; lfpsx		 16,0,3	
	fpmr	 5,0		; lfpsux	 17,3,11
	fpmr	 6,0		; lfpsx		 18,0,4	
	fpmr	 7,0		; lfpsux	 19,4,11
	fpmr	 8,0		; lfpsx		 20,0,5	
	fpmr	 9,0		; lfpsux	 21,5,11
	fpmr	 10,0		; lfpsx		 22,0,6	
	fpmr	 11,0		; lfpsux	 23,6,11
	fpmr	 12,0		; lfpsux	 24,3,11
	fpmr	 13,0		; lfpsux	 25,3,11
	fpmr	 14,0		; lfpsux	 28,5,11
	fpmr	 15,0	

0:	# loop over time

	# S0 * ~S2, phase 1
	fxcpnsma 0,16,20,0
	fxcpnsma 1,16,21,1	; lfpsux	 29,5,11
	fxcpnsma 2,17,20,2
	fxcpnsma 3,17,21,3

	# S0 * ~S3, phase 1
	fxcpnsma 4,16,22,4
	fxcpnsma 5,16,23,5	; lfpsux	 26,4,11
	fxcpnsma 6,17,22,6
	fxcpnsma 7,17,23,7

	# S1 * ~S2, phase 1
	fxcpnsma 8,18,20,8
	fxcpnsma 9,18,21,9	; lfpsux	 27,4,11
	fxcpnsma 10,19,20,10
	fxcpnsma 11,19,21,11

	# S1 * ~S3, phase 1
	fxcpnsma 12,18,22,12
	fxcpnsma 13,18,23,13	; lfpsux	 30,6,11
	fxcpnsma 14,19,22,14
	fxcpnsma 15,19,23,15

	# S0 * ~S2, phase 2
	fxcxma   0,16,20,0
	fxcxma   1,16,21,1	; lfpsux	 31,6,11
	fxcxma   2,17,20,2
	fxcxma   3,17,21,3

	# S0 * ~S3, phase 2
	fxcxma   4,16,22,4
	fxcxma   5,16,23,5	; lfpsux	 16,3,11
	fxcxma   6,17,22,6
	fxcxma   7,17,23,7

	# S1 * ~S2, phase 2
	fxcxma   8,18,20,8
	fxcxma   9,18,21,9	; lfpsux	 17,3,11
	fxcxma   10,19,20,10
	fxcxma   11,19,21,11

	# S1 * ~S3, phase 2
	fxcxma   12,18,22,12
	fxcxma   13,18,23,13	; lfpsux	 20,5,11
	fxcxma   14,19,22,14
	fxcxma   15,19,23,15

	# S0 * ~S2, phase 1
	fxcpnsma 0,24,28,0
	fxcpnsma 1,24,29,1	; lfpsux	 21,5,11
	fxcpnsma 2,25,28,2
	fxcpnsma 3,25,29,3

	# S0 * ~S3, phase 1
	fxcpnsma 4,24,30,4
	fxcpnsma 5,24,31,5	; lfpsux	 18,4,11
	fxcpnsma 6,25,30,6
	fxcpnsma 7,25,31,7

	# S1 * ~S2, phase 1
	fxcpnsma 8,26,28,8
	fxcpnsma 9,26,29,9	; lfpsux	 19,4,11
	fxcpnsma 10,27,28,10
	fxcpnsma 11,27,29,11

	# S1 * ~S3, phase 1
	fxcpnsma 12,26,30,12
	fxcpnsma 13,26,31,13	; lfpsux	 22,6,11
	fxcpnsma 14,27,30,14
	fxcpnsma 15,27,31,15

	# S0 * ~S2, phase 2
	fxcxma   0,24,28,0
	fxcxma   1,24,29,1	; lfpsux	 23,6,11
	fxcxma   2,25,28,2
	fxcxma   3,25,29,3

	# S0 * ~S3, phase 2
	fxcxma   4,24,30,4
	fxcxma   5,24,31,5	; lfpsux	 24,3,11
	fxcxma   6,25,30,6
	fxcxma   7,25,31,7

	# S1 * ~S2, phase 2
	fxcxma   8,26,28,8
	fxcxma   9,26,29,9	; lfpsux	 25,3,11
	fxcxma   10,27,28,10
	fxcxma   11,27,29,11

	# S1 * ~S3, phase 2
	fxcxma   12,26,30,12
	fxcxma   13,26,31,13	; lfpsux	 28,5,11
	fxcxma   14,27,30,14
	fxcxma   15,27,31,15

	bdnz	 0b

#if !defined HAVE_BGP
	dcbz	 0,7
#endif
	stfpsx	 0,0,7		# store results S0 * ~S2
	stfpsux	 1,7,11
	stfpsux	 2,7,11
	stfpsux	 3,7,11

#if !defined HAVE_BGP
	dcbz	 0,8
#endif
	stfpsx	 4,0,8		# store results S0 * ~S3
	stfpsux	 5,8,11
	stfpsux	 6,8,11
	stfpsux	 7,8,11

#if !defined HAVE_BGP
	dcbz	 0,9
#endif
	stfpsx	 8,0,9		# store results S1 * ~S2
	stfpsux	 9,9,11
	stfpsux	 10,9,11
	stfpsux	 11,9,11

#if !defined HAVE_BGP
	dcbz	 0,10
#endif
	stfpsx	 12,0,10	# store results S1 * ~S3
	stfpsux	 13,10,11
	stfpsux	 14,10,11
	stfpsux	 15,10,11

	li	 11,16		# restore call-saved registers
	lfpdx	 31,0,1
	lfpdux	 30,1,11
	lfpdux	 29,1,11
	lfpdux	 28,1,11
	lfpdux	 27,1,11
	lfpdux	 26,1,11
	lfpdux	 25,1,11
	lfpdux	 24,1,11
	lfpdux	 23,1,11
	lfpdux	 22,1,11
	lfpdux	 21,1,11
	lfpdux	 20,1,11
	lfpdux	 19,1,11
	lfpdux	 18,1,11
	lfpdux	 17,1,11
	lfpdux	 16,1,11
	lfpdux	 15,1,11
	lfpdux	 14,1,11

	addi	 1,1,16		# reset stack pointer

	blr			# return


.align	5
.global _correlate_3x2
_correlate_3x2:

#	computes correlations of S0*~S3,S0*~S4,S1*~S3,S1*~S4,S2*~S3,S2*~S4
#	r3 :	pointer to S0 data
#	r4 :	pointer to S1 data
#	r5 :	pointer to S2 data
#	r6 :	pointer to S3 data
#	r7 :	pointer to S4 data
#	r8 :	pointer to S0*~S3 output
#	r9 :	pointer to S0*~S4 output
#	r10:	pointer to S1*~S3 output
#	8(r1):	pointer to S1*~S4 output
#	12(r1):	pointer to S2*~S3 output
#	16(r1):	pointer to S2*~S4 output
#	20(r1): nr_samples_to_integrate

	li	 12,-16		; lwz		 0,20(1)
	lis	 11,zero@ha	; stfpdux	 14,1,12
	addi	 11,11,zero@l	; stfpdux	 15,1,12
				; lfpsx	 0,0,11

	srwi	 0,0,1		; stfpdux	 16,1,12
	mtctr	 0		; stfpdux	 17,1,12
	fpmr	 1,0		; stfpdux	 18,1,12
	fpmr	 2,0		; stfpdux	 19,1,12
	fpmr	 3,0		; stfpdux	 20,1,12
	fpmr	 4,0		; stfpdux	 21,1,12
	fpmr	 5,0		; stfpdux	 22,1,12
	fpmr	 6,0		; stfpdux	 23,1,12
	fpmr	 7,0		; stfpdux	 24,1,12
	fpmr	 8,0		; stfpdux	 25,1,12
	fpmr	 9,0		; stfpdux	 26,1,12
	fpmr	 10,0		; stfpdux	 27,1,12
	fpmr	 11,0		; stfpdux	 28,1,12
	fpmr	 12,0		; stfpdux	 29,1,12
	fpmr	 13,0		; stfpdux	 30,1,12
	fpmr	 14,0		; stfpdux	 31,1,12
	fpmr	 15,0		; li	 11,8

	fpmr	 16,0		; lfpsx		 24,0,3
	fpmr	 17,0		; lfpsux	 25,3,11
	fpmr	 18,0		; lfpsx		 26,0,4
	fpmr	 19,0		; lfpsx		 28,0,6
	fpmr	 20,0		; lfpsux	 29,6,11
	fpmr	 21,0		; lfpsx		 30,0,7
	fpmr	 22,0		; sub	 5,5,11
	fpmr	 23,0


0:
	# S0 * ~S3, phase 1
	fxcpnsma 0,24,28,0	; lfpsux	 31,7,11
	fxcpnsma 1,24,29,1	; lfpsux	 27,4,11
	fxcpnsma 2,25,28,2
	fxcpnsma 3,25,29,3

	# S0 * ~S4, phase 1
	fxcpnsma 4,24,30,4
	fxcpnsma 5,24,31,5
	fxcpnsma 6,25,30,6
	fxcpnsma 7,25,31,7

	# S0 * ~S3, phase 2
	fxcxma   0,24,28,0
	fxcxma   1,24,29,1
	fxcxma   2,25,28,2
	fxcxma   3,25,29,3

	# S0 * ~S4, phase 2	; # ld S2
	fxcxma   4,24,30,4
	fxcxma   5,24,31,5	; lfpsux	 24,5,11
	fxcxma   6,25,30,6
	fxcxma   7,25,31,7	; lfpsux	 25,5,11

	# S1 * ~S3, phase 1
	fxcpnsma 8,26,28,8
	fxcpnsma 9,26,29,9
	fxcpnsma 10,27,28,10
	fxcpnsma 11,27,29,11

	# S1 * ~S4, phase 1
	fxcpnsma 12,26,30,12
	fxcpnsma 13,26,31,13
	fxcpnsma 14,27,30,14
	fxcpnsma 15,27,31,15

	# S1 * ~S3, phase 2
	fxcxma   8,26,28,8
	fxcxma   9,26,29,9
	fxcxma   10,27,28,10
	fxcxma   11,27,29,11

	# S1 * ~S4, phase 2	; # ld S0
	fxcxma   12,26,30,12
	fxcxma   13,26,31,13	; lfpsux	 26,3,11
	fxcxma   14,27,30,14
	fxcxma   15,27,31,15	; lfpsux	 27,3,11

	# S2 * ~S3, phase 1
	fxcpnsma 16,24,28,16
	fxcpnsma 17,24,29,17
	fxcpnsma 18,25,28,18
	fxcpnsma 19,25,29,19

	# S2 * ~S4, phase 1
	fxcpnsma 20,24,30,20
	fxcpnsma 21,24,31,21
	fxcpnsma 22,25,30,22
	fxcpnsma 23,25,31,23

	# S2 * ~S3, phase 2	; # ld S3
	fxcxma   16,24,28,16
	fxcxma   17,24,29,17
	fxcxma   18,25,28,18	; lfpsux	 28,6,11
	fxcxma   19,25,29,19	; lfpsux	 29,6,11

	# S2 * ~S4, phase 2	; # ld S4
	fxcxma   20,24,30,20
	fxcxma   21,24,31,21
	fxcxma   22,25,30,22	; lfpsux	 30,7,11
	fxcxma   23,25,31,23

	# S0 * ~S3, phase 1	; # ld S1
	fxcpnsma 0,26,28,0
	fxcpnsma 1,26,29,1
	fxcpnsma 2,27,28,2	; lfpsux	 31,7,11
	fxcpnsma 3,27,29,3	; lfpsux	 24,4,11

	# S0 * ~S4, phase 1
	fxcpnsma 4,26,30,4
	fxcpnsma 5,26,31,5
	fxcpnsma 6,27,30,6	; lfpsux	 25,4,11
	fxcpnsma 7,27,31,7

	# S0 * ~S4, phase 2
	fxcxma   0,26,28,0
	fxcxma   1,26,29,1
	fxcxma   2,27,28,2
	fxcxma   3,27,29,3

	# S0 * ~S4, phase 2	; # ld S2
	fxcxma   4,26,30,4
	fxcxma   5,26,31,5	; lfpsux	 26,5,11
	fxcxma   6,27,30,6
	fxcxma   7,27,31,7 	; lfpsux	 27,5,11

	# S1 * ~S3, phase 1	; # fetch 2 station S3 samples from L2 to L1
	fxcpnsma 8,24,28,8	; dcbt	 6,11	
	fxcpnsma 9,24,29,9	; dcbt	 7,11		# ditto for S4
	fxcpnsma 10,25,28,10
	fxcpnsma 11,25,29,11

	# S1 * ~S4, phase 1
	fxcpnsma 12,24,30,12
	fxcpnsma 13,24,31,13
	fxcpnsma 14,25,30,14
	fxcpnsma 15,25,31,15

	# S1 * ~S3, phase 2
	fxcxma   8,24,28,8
	fxcxma   9,24,29,9
	fxcxma   10,25,28,10
	fxcxma   11,25,29,11

	# S1 * ~S4, phase 2	; # ld S0
	fxcxma   12,24,30,12
	fxcxma   13,24,31,13	; lfpsux	 24,3,11
	fxcxma   14,25,30,14
	fxcxma   15,25,31,15	; lfpsux	 25,3,11

	# S2 * ~S3, phase 1
	fxcpnsma 16,26,28,16
	fxcpnsma 17,26,29,17
	fxcpnsma 18,27,28,18
	fxcpnsma 19,27,29,19

	# S2 * ~S4, phase 1
	fxcpnsma 20,26,30,20
	fxcpnsma 21,26,31,21
	fxcpnsma 22,27,30,22
	fxcpnsma 23,27,31,23

	# S2 * ~S3, phase 2	; # ld S3
	fxcxma   16,26,28,16
	fxcxma   17,26,29,17
	fxcxma   18,27,28,18	; lfpsux	 28,6,11
	fxcxma   19,27,29,19	; lfpsux	 29,6,11

	# S2 * ~S4, phase 2	; # ld S1 and S4
	fxcxma   20,26,30,20
	fxcxma   21,26,31,21	; lfpsux	 26,4,11
	fxcxma   22,27,30,22	; lfpsux	 30,7,11
	fxcxma   23,27,31,23

	bdnz	 0b

	lwz	 3,288+8(1)	# load function arguments 9-11
	lwz	 4,288+12(1)
	lwz	 5,288+16(1)

#if !defined HAVE_BGP
	dcbz	 0,8
#endif
	stfpsx	 0,0,8		# store results S0 * ~S3
	stfpsux	 1,8,11
	stfpsux	 2,8,11
	stfpsux	 3,8,11

#if !defined HAVE_BGP
	dcbz	 0,9
#endif
	stfpsx	 4,0,9		# store results S0 * ~S4
	stfpsux	 5,9,11
	stfpsux	 6,9,11
	stfpsux	 7,9,11

#if !defined HAVE_BGP
	dcbz	 0,10
#endif
	stfpsx	 8,0,10		# store results S1 * ~S3
	stfpsux	 9,10,11
	stfpsux	 10,10,11
	stfpsux	 11,10,11

#if !defined HAVE_BGP
	dcbz	 0,3
#endif
	stfpsx	 12,0,3		# store results S1 * ~S4
	stfpsux	 13,3,11
	stfpsux	 14,3,11
	stfpsux	 15,3,11

#if !defined HAVE_BGP
	dcbz	 0,4
#endif
	stfpsx	 16,0,4		# store results S2 * ~S3
	stfpsux	 17,4,11
	stfpsux	 18,4,11
	stfpsux	 19,4,11

#if !defined HAVE_BGP
	dcbz	 0,5
#endif
	stfpsx	 20,0,5		# store results S2 * ~S4
	stfpsux	 21,5,11
	stfpsux	 22,5,11
	stfpsux	 23,5,11

	li	 11,16		# restore call-saved registers
	lfpdx	 31,0,1
	lfpdux	 30,1,11
	lfpdux	 29,1,11
	lfpdux	 28,1,11
	lfpdux	 27,1,11
	lfpdux	 26,1,11
	lfpdux	 25,1,11
	lfpdux	 24,1,11
	lfpdux	 23,1,11
	lfpdux	 22,1,11
	lfpdux	 21,1,11
	lfpdux	 20,1,11
	lfpdux	 19,1,11
	lfpdux	 18,1,11
	lfpdux	 17,1,11
	lfpdux	 16,1,11
	lfpdux	 15,1,11
	lfpdux	 14,1,11

	addi	 1,1,16		# reset stack pointer

	blr			# return


.align	5
.global _auto_correlate_1
_auto_correlate_1:

#	computes auto correlations of S0*~S0
#	r3 :	pointer to S0 data
#	r4 :	pointer to 4 results
#	r5 :	nr_samples_to_integrate

	srwi	 5,5,2
	lis	 12,zero@ha
	mtctr	 5
	addi	 12,12,zero@l
	lfpsx	 0,0,12		# f0r = 0, f0i = 0

	li	 8,8

	lfpsx	 4,0,3		# f8:f9 = S0
	lfpsux	 5,3,8
	lfpsux	 6,3,8		# f6:f7 = S0
	lfpsux	 7,3,8
	fpmr	 1,0
	lfpsux	 8,3,8		# f8:f9 = S0
	fpmr	 2,0
	lfpsux	 9,3,8

	fpmr	 3,0
	lfpsx	 12,12,8	# f12r = 1, f12i = 1


0:	# loop over time

	lfpsux	 10,3,8		# f6:f7 = S0
	fpmadd	 0,4,4,0	# S0 * ~S0
	lfpsux	 11,3,8
	fxcpnsma 1,4,5,1	# f1r += f5r*f4r, f1i -= f5i*f4r
	fxcxma   2,4,5,2	# f2r += f5i*f4i, f2i += f5r*f4i
	fpmadd	 3,5,5,3

	lfpsux	 4,3,8		# f4:f5 = S0
	fpmadd	 0,6,6,0	# S0 * ~S0
	lfpsux	 5,3,8
	fxcpnsma 1,6,7,1	# f1r += f7r*f6r, f1i -= f7i*f6r
	fxcxma   2,6,7,2	# f2r += f7i*f6i, f2i += f7r*f6i
	fpmadd	 3,7,7,3

	lfpsux	 6,3,8		# f6:f7 = S0
	fpmadd	 0,8,8,0	# S0 * ~S0
	lfpsux	 7,3,8
	fxcpnsma 1,8,9,1	# f1r += f9r*f8r, f1i -= f9i*f8r
	fxcxma   2,8,9,2	# f2r += f9i*f8i, f2i += f9r*f8i
	fpmadd	 3,9,9,3

	lfpsux	 8,3,8		# f8:f9 = S0
	fpmadd	 0,10,10,0	# S0 * ~S0
	lfpsux	 9,3,8
	fxcpnsma 1,10,11,1	# f1r += f11r*f10r, f1i -= f11i*f10r
	fxcxma   2,10,11,2	# f2r += f11i*f10i, f2i += f11r*f10i
	fpmadd	 3,11,11,3

	bdnz	 0b

	fxcsnsma 0,0,12,0	# f0r += 1*f0i, f0i += -1*f0i = 0
	fpadd	 1,1,2
	fxcsnsma 3,3,12,3

#if !defined HAVE_BGP
	dcbz	 0,4		# store result
#endif
	stfpsx	 0,0,4
	stfpsux	 1,4,8
	fsneg	 1,1
	stfpsux	 1,4,8
	stfpsux	 3,4,8

	blr			# return


.align	5
.global _auto_correlate_2
_auto_correlate_2:

#	computes correlations of S0*~S0,S0*~S1,S1*~S1
#	r3 :	pointer to S0 data
#	r4 :	pointer to S1 data
#	r5 :	pointer to S0*~S0 output
#	r6 :	pointer to S0*~S1 output
#	r7 :	pointer to S1*~S1 output
#	r8 :	nr_samples_to_integrate

#	local variables:
#	r11:	temporary
#	f0 :	sum of S0_X * S0_X
#	f1 :	sum of S0_X * ~S0_Y
#	f3 :	sum of S0_Y * S0_Y
#	f4 :	sum of S0_X * ~S1_X
#	f5 :	sum of S0_X * ~S1_Y
#	f6 :	sum of S0_Y * ~S1_X
#	f7 :	sum of S0_Y * ~S1_Y
#	f8 :	sum of S1_X * S1_X
#	f9 :	sum of S1_X * ~S1_Y
#	f11:	sum of S1_Y * S1_Y
#	f12,f16:S0_X (sample from station S0, X polarization)
#	f13,f17:S0_Y
#	f14,f18:S1_X
#	f15,f19:S1_Y

	srwi	 8,8,1
	lis	 12,zero@ha
	mtctr	 8
	addi	 12,12,zero@l
	lfpsx	 0,0,12

	li	 11,-16		# push call-saved registers
	stfpdux	 14,1,11
	fpmr	 1,0
	stfpdux	 15,1,11
	fpmr	 2,0
	stfpdux	 16,1,11
	fpmr	 3,0
	stfpdux	 17,1,11
	fpmr	 4,0
	stfpdux	 18,1,11
	fpmr	 5,0
	stfpdux	 19,1,11

	fpmr	 6,0
	li	 11,8		# prefetch station samples
	fpmr	 7,0
	lfpsx	 12,0,3	
	fpmr	 8,0
	lfpsux	 13,3,11
	fpmr	 9,0
	lfpsx	 14,0,4	
	fpmr	 10,0
	lfpsux	 15,4,11
	fpmr	 11,0

0:	# loop over time

	# S0 * ~S0, phase 1
	fpmadd	 0,12,12,0	; lfpsux	 16,3,11
	fxcpnsma 1,12,13,1	; lfpsux	 17,3,11
	fpmadd	 3,13,13,3	; lfpsux	 18,4,11

	# S0 * ~S1, phase 1
	fxcpnsma 4,12,14,4
	fxcpnsma 5,12,15,5
	fxcpnsma 6,13,14,6
	fxcpnsma 7,13,15,7

	# S1 * ~S1, phase 1
	fpmadd	 8,14,14,8	; lfpsux	 19,4,11
	fxcpnsma 9,14,15,9
	fpmadd	 11,15,15,11

	# S0 * ~S0, phase 2
	fxcxma   1,12,13,1

	# S0 * ~S1, phase 2
	fxcxma   4,12,14,4
	fxcxma   5,12,15,5	; lfpsux	 12,3,11
	fxcxma   6,13,14,6
	fxcxma   7,13,15,7	; lfpsux	 13,3,11

	# S1 * ~S1, phase 2
	fxcxma   9,14,15,9	; lfpsux	 14,4,11

	# S0 * ~S0, phase 1
	fpmadd	 0,16,16,0
	fxcpnsma 1,16,17,1
	fpmadd	 3,17,17,3

	# S0 * ~S1, phase 1
	fxcpnsma 4,16,18,4
	fxcpnsma 5,16,19,5	; lfpsux	 15,4,11
	fxcpnsma 6,17,18,6
	fxcpnsma 7,17,19,7

	# S1 * ~S1, phase 1
	fpmadd	 8,18,18,8
	fxcpnsma 9,18,19,9
	fpmadd	 11,19,19,11

	# S0 * ~S0, phase 2
	fxcxma   1,16,17,1

	# S0 * ~S1, phase 2
	fxcxma   4,16,18,4
	fxcxma   5,16,19,5
	fxcxma   6,17,18,6
	fxcxma   7,17,19,7

	# S1 * ~S1, phase 2
	fxcxma   9,18,19,9

	bdnz	 0b

	lfpsx	 12,12,11	# f12r = 1, f12i = 1

	fxcsnsma 0,0,12,0	# f0r += 1*f0i, f0i += -1*f0i = 0
	fxcsnsma 3,3,12,3
	fxcsnsma 8,8,12,8
	fxcsnsma 11,11,12,11

#if !defined HAVE_BGP
	dcbz	 0,5
#endif
	stfpsx	 0,0,5		# store results S0 * ~S0
	stfpsux	 1,5,11
	fsneg	 1,1
	stfpsux	 1,5,11
	stfpsux	 3,5,11

#if !defined HAVE_BGP
	dcbz	 0,6
#endif
	stfpsx	 4,0,6		# store results S0 * ~S1
	stfpsux	 5,6,11
	stfpsux	 6,6,11
	stfpsux	 7,6,11

#if !defined HAVE_BGP
	dcbz	 0,7
#endif
	stfpsx	 8,0,7		# store results S1 * ~S1
	stfpsux	 9,7,11
	fsneg	 9,9
	stfpsux	 9,7,11
	stfpsux	 11,7,11

	li	  11,16		# restore call-saved registers
	lfpdx	  19,0,1
	lfpdux	  18,1,11
	lfpdux	  17,1,11
	lfpdux	  16,1,11
	lfpdux	  15,1,11
	lfpdux	  14,1,11

	addi	  1,1,16	# reset stack pointer

	blr			# return


.align	5
.global _auto_correlate_3
_auto_correlate_3:

#	computes correlations of S0*~S1,S0*~S2,S1*~S1,S1*~S2,S2*~S2
#	r3 :	pointer to S0 data
#	r4 :	pointer to S1 data
#	r5 :	pointer to S2 data
#	r6 :	pointer to S0*~S1 output
#	r7 :	pointer to S0*~S2 output
#	r8 :	pointer to S1*~S1 output
#	r9 :	pointer to S1*~S2 output
#	r10 :	pointer to S2*~S2 output
#	8(r1):	nr_samples_to_integrate

#	local variables:
#	r11:	temporary
#	f0:	sum of S0_X * ~S1_X
#	f1:	sum of S0_X * ~S1_Y
#	f2:	sum of S0_Y * ~S1_X
#	f3:	sum of S0_Y * ~S1_Y
#	f4:	sum of S0_X * ~S1_X
#	f5:	sum of S0_X * ~S1_Y
#	f6:	sum of S0_Y * ~S1_X
#	f7:	sum of S0_Y * ~S1_Y
#	f8 :	sum of S1_X * S1_X
#	f9 :	sum of S1_X * ~S1_Y
#	f10 :	sum of S1_Y * S1_Y
#	f11 :	sum of S0_X * ~S1_X
#	f12 :	sum of S0_X * ~S1_Y
#	f13 :	sum of S0_Y * ~S1_X
#	f14 :	sum of S0_Y * ~S1_Y
#	f15 :	sum of S2_X * S2_X
#	f16 :	sum of S2_X * ~S2_Y
#	f17 :	sum of S2_Y * S2_Y
#	f18,f24:S0_X (sample from station S0, X polarization)
#	f19,f25:S0_Y
#	f20,f26:S1_X
#	f21,f27:S1_Y
#	f22,f28:S2_X
#	f22,f29:S2_Y


	lis	 12,zero@ha
	lwz	 0,8(1)
	addi	 12,12,zero@l
	lfpsx	 0,0,12

	li	 11,-16		# push call-saved registers
	stfpdux	 14,1,11
	stfpdux	 15,1,11
	stfpdux	 16,1,11
	srwi	 0,0,1
	stfpdux	 17,1,11
	mtctr	 0
	stfpdux	 18,1,11	; fpmr	 1,0
	stfpdux	 19,1,11
	fpmr	 2,0
	stfpdux	 20,1,11
	fpmr	 3,0
	stfpdux	 21,1,11
	fpmr	 4,0
	stfpdux	 22,1,11
	fpmr	 5,0
	stfpdux	 23,1,11
	fpmr	 6,0
	stfpdux	 24,1,11
	fpmr	 7,0
	stfpdux	 25,1,11
	fpmr	 8,0
	stfpdux	 26,1,11
	fpmr	 9,0
	stfpdux	 27,1,11
	fpmr	 10,0
	stfpdux	 28,1,11
	fpmr	 11,0
	stfpdux	 29,1,11

	li	 11,8		# prefetch station samples
	; lfpsx	 18,0,3	
	fpmr	 12,0
	; lfpsux	 19,3,11
	fpmr	 13,0
	; lfpsx	 20,0,4	
	fpmr	 14,0
	; lfpsux	 21,4,11
	fpmr	 15,0
	; lfpsx	 22,0,5	
	fpmr	 16,0
	; lfpsux	 23,5,11
	fpmr	 17,0

0:	# loop over time
	# S0 * ~S1, phase 1
	fxcpnsma 0,18,20,0	; lfpsux	 24,3,11
	fxcpnsma 1,18,21,1	; lfpsux	 25,3,11
	fxcpnsma 2,19,20,2	; lfpsux	 26,4,11
	fxcpnsma 3,19,21,3

	# S0 * ~S2, phase 1
	fxcpnsma 4,18,22,4
	fxcpnsma 5,18,23,5
	fxcpnsma 6,19,22,6
	fxcpnsma 7,19,23,7

	# S1 * ~S1, phase 1
	fpmadd	 8,20,20,8	; lfpsux	 27,4,11
	fxcpnsma 9,20,21,9	; lfpsux	 28,5,11
	fpmadd	 10,21,21,10	; lfpsux	 29,5,11

	# S1 * ~S2, phase 1
	fxcpnsma 11,20,22,11
	fxcpnsma 12,20,23,12
	fxcpnsma 13,21,22,13
	fxcpnsma 14,21,23,14

	# S2 * ~S2, phase 1
	fpmadd	 15,22,22,15
	fxcpnsma 16,22,23,16
	fpmadd	 17,23,23,17

	# S0 * ~S1, phase 2
	fxcxma   0,18,20,0
	fxcxma   1,18,21,1
	fxcxma   2,19,20,2
	fxcxma   3,19,21,3

	# S0 * ~S2, phase 2
	fxcxma	 4,18,22,4
	fxcxma	 5,18,23,5
	fxcxma	 6,19,22,6
	fxcxma	 7,19,23,7

	# S1_X * ~S1_Y, phase 2
	fxcxma   9,20,21,9

	# S1 * ~S2, phase 2
	fxcxma	 11,20,22,11
	fxcxma	 12,20,23,12
	fxcxma	 13,21,22,13
	fxcxma	 14,21,23,14

	# S2 * ~S2, phase 2
	fxcxma   16,22,23,16

	# S0 * ~S1, phase 1
	fxcpnsma 0,24,26,0	; lfpsux	 18,3,11
	fxcpnsma 1,24,27,1	; lfpsux	 19,3,11
	fxcpnsma 2,25,26,2	; lfpsux	 20,4,11
	fxcpnsma 3,25,27,3

	# S0 * ~S2, phase 1
	fxcpnsma 4,24,28,4
	fxcpnsma 5,24,29,5
	fxcpnsma 6,25,28,6
	fxcpnsma 7,25,29,7

	# S1 * ~S1, phase 1
	fpmadd	 8,26,26,8	; lfpsux	 21,4,11
	fxcpnsma 9,26,27,9	; lfpsux	 22,5,11
	fpmadd	 10,27,27,10	; lfpsux	 23,5,11

	# S1 * ~S2, phase 1
	fxcpnsma 11,26,28,11
	fxcpnsma 12,26,29,12
	fxcpnsma 13,27,28,13
	fxcpnsma 14,27,29,14

	# S2 * ~S2, phase 1
	fpmadd	 15,28,28,15
	fxcpnsma 16,28,29,16
	fpmadd	 17,29,29,17

	# S0 * ~S1, phase 2
	fxcxma   0,24,26,0
	fxcxma   1,24,27,1
	fxcxma   2,25,26,2
	fxcxma   3,25,27,3

	# S0 * ~S2, phase 2
	fxcxma	 4,24,28,4
	fxcxma	 5,24,29,5
	fxcxma	 6,25,28,6
	fxcxma	 7,25,29,7

	# S1_X * ~S1_Y, phase 2
	fxcxma   9,26,27,9

	# S1 * ~S2, phase 2
	fxcxma	 11,26,28,11
	fxcxma	 12,26,29,12
	fxcxma	 13,27,28,13
	fxcxma	 14,27,29,14

	# S2 * ~S2, phase 2
	fxcxma   16,28,29,16

	bdnz	 0b

	lfpsx	 18,12,11	# f18r = 1, f18i = 1

	fxcsnsma 8,8,18,8	# f8r += 1*f8i, f8i += -1*f8i = 0
	fxcsnsma 10,10,18,10
	fxcsnsma 15,15,18,15
	fxcsnsma 17,17,18,17

#if !defined HAVE_BGP
	dcbz	 0,6
#endif
	stfpsx	 0,0,6		# store results S0 * ~S1
	stfpsux	 1,6,11
	stfpsux	 2,6,11
	stfpsux	 3,6,11

#if !defined HAVE_BGP
	dcbz	 0,7
#endif
	stfpsx	 4,0,7		# store results S0 * ~S2
	stfpsux	 5,7,11
	stfpsux	 6,7,11
	stfpsux	 7,7,11

#if !defined HAVE_BGP
	dcbz	 0,8
#endif
	stfpsx	 8,0,8		# store results S1 * ~S1
	stfpsux	 9,8,11
	fsneg	 9,9
	stfpsux	 9,8,11
	stfpsux	 10,8,11

#if !defined HAVE_BGP
	dcbz	 0,9
#endif
	stfpsx	 11,0,9		# store results S1 * ~S2
	stfpsux	 12,9,11
	stfpsux	 13,9,11
	stfpsux	 14,9,11

#if !defined HAVE_BGP
	dcbz	 0,10
#endif
	stfpsx	 15,0,10	# store results S2 * ~S2
	stfpsux	 16,10,11
	fsneg	 16,16
	stfpsux	 16,10,11
	stfpsux	 17,10,11

	li	  11,16		# restore call-saved registers
	lfpdx	  29,0,1
	lfpdux	  28,1,11
	lfpdux	  27,1,11
	lfpdux	  26,1,11
	lfpdux	  25,1,11
	lfpdux	  24,1,11
	lfpdux	  23,1,11
	lfpdux	  22,1,11
	lfpdux	  21,1,11
	lfpdux	  20,1,11
	lfpdux	  19,1,11
	lfpdux	  18,1,11
	lfpdux	  17,1,11
	lfpdux	  16,1,11
	lfpdux	  15,1,11
	lfpdux	  14,1,11

	addi	  1,1,16	# reset stack pointer

	blr			# return


#if 0
.align	5
.global	_add_correlations
_add_correlations:

	li	 11,-16		# push call-saved registers
	subi	 1,1,32
	stmw	 24,0(1)
	stfpdux	 14,1,11
	stfpdux	 15,1,11

	srwi	7,7,1
	mtctr	7

	li	8,8

	sub	3,3,8
	sub	4,4,8

	mr	10,3

				; lfpsux	0,3,8
				; lfpsux	1,3,8
				; lfpsux	2,3,8
				; lfpsux	3,3,8
				; lfpsux	4,3,8
				; lfpsux	5,3,8
				; lfpsux	6,3,8
				; lfpsux	7,3,8

				; lfpsux	8,4,8
				; lfpsux	9,4,8
				; lfpsux	10,4,8
				; lfpsux	11,4,8
				; lfpsux	12,4,8
				; lfpsux	13,4,8
				; lfpsux	14,4,8
				; lfpsux	15,4,8

0:				
	fpadd	0,0,8		; lfpsux	8,4,8
	fpadd	1,1,9		; lfpsux	9,4,8
	fpadd	2,2,10		; lfpsux	10,4,8
	fpadd	3,3,11		; lfpsux	11,4,8
	fpadd	4,4,12		; lfpsux	12,4,8
	fpadd	5,5,13		; lfpsux	13,4,8
	fpadd	6,6,14		; lfpsux	14,4,8
	fpadd	7,7,15		; lfpsux	15,4,8

				; stfpsux	0,10,8
				; lfpsux	0,3,8
				; stfpsux	1,10,8
				; lfpsux	1,3,8
				; stfpsux	2,10,8
				; lfpsux	2,3,8
				; stfpsux	3,10,8
				; lfpsux	3,3,8
				; stfpsux	4,10,8
				; lfpsux	4,3,8
				; stfpsux	5,10,8
				; lfpsux	5,3,8
				; stfpsux	6,10,8
				; lfpsux	6,3,8
				; stfpsux	7,10,8
				; lfpsux	7,3,8

	bdnz	0b

	# now do all nrValidSamples; since overflows should not occur, we
	# treat two unsigned shorts as one word

	srwi	7,7,2		; lwzx		24,0,5
	mtctr	7		; lwzx		28,0,6
	li	9,4		; lwzux		25,5,9
				; lwzux		29,6,9
				; lwzux		26,5,9
				; lwzux		30,6,9
				; lwzux		27,5,9
				; lwzux		31,6,9

1:	add	24,24,28	; lwzux		28,6,9
				; stw		24,-28(5)
				; lwzux		24,5,9

	add	25,25,29	; lwzux		29,6,9
				; stw		25,-28(5)
				; lwzux		25,5,9

	add	26,26,30	; lwzux		30,6,9
				; stw		26,-28(5)
				; lwzux		26,5,9

	add	27,27,31	; lwzux		31,6,9
				; stw		27,-28(5)
				; lwzux		27,5,9

	bdnz	1b

	li	  11,16		# restore call-saved registers
	lfpdx	  15,0,1
	lfpdux	  14,1,11

	lmw	  24,16(1)
	addi	  1,1,48	# reset stack pointer

	blr			# return
#endif


.align	5
.global	_clear_correlation
_clear_correlation:
#if defined HAVE_BGP
	li	0,0
	stw	0,0(3)
	stw	0,4(3)
	stw	0,8(3)
	stw	0,12(3)
	stw	0,16(3)
	stw	0,20(3)
	stw	0,24(3)
	stw	0,28(3)
#else
	dcbz	0,3		# clear the entire cache line, it contains all
#endif
	blr			# polarizations


.align	5
.global	_weigh_visibilities_1
_weigh_visibilities_1:

	mullw	6,6,7
	li	8,8
	li	11,1
	srwi.	0,6,1
	sub	4,4,11
	mtctr	0
	sub	7,3,8

	lbzux	9,4,11
	lbzux	10,4,11

	lfpsx	0,0,3
	lfpsux	1,3,8
	lfpsux	2,3,8
	lfpsux	3,3,8

	beq	0,1f

0:	
				; slwi	9,9,2
				; lfsx	8,5,9

				; lfpsux 4,3,8
				; lfpsux 5,3,8
				; lfpsux 6,3,8
				; lfpsux 7,3,8

	fxpmul	0,8,0
	fxpmul	1,8,1
	fxpmul	2,8,2
	fxpmul	3,8,3

	lbzux	9,4,11

				; stfpsux 0,7,8
				; stfpsux 1,7,8
				; stfpsux 2,7,8
				; stfpsux 3,7,8

				; slwi	10,10,2
				; lfsx	8,5,10

				; lfpsux 0,3,8
				; lfpsux 1,3,8
				; lfpsux 2,3,8
				; lfpsux 3,3,8
	fxpmul	4,8,4	
	fxpmul	5,8,5
	fxpmul	6,8,6
	fxpmul	7,8,7

	lbzux	10,4,11

				; stfpsux 4,7,8
				; stfpsux 5,7,8
				; stfpsux 6,7,8
				; stfpsux 7,7,8

	bdnz	0b

1:	andi.	0,6,1
	beq	0,2f
				; slwi	9,9,2
				; lfsx	8,5,9
	fxpmul	0,8,0
	fxpmul	1,8,1
	fxpmul	2,8,2
	fxpmul	3,8,3
				; stfpsux 0,7,8
				; stfpsux 1,7,8
				; stfpsux 2,7,8
				; stfpsux 3,7,8

2:	blr


.align	5
.global	_weigh_visibilities_2
_weigh_visibilities_2:

	mullw	6,6,7
	li	8,8
	li	11,2
	srwi.	0,6,1
	sub	4,4,11
	mtctr	0
	sub	7,3,8

	lhzux	9,4,11
	lhzux	10,4,11

	lfpsx	0,0,3
	lfpsux	1,3,8
	lfpsux	2,3,8
	lfpsux	3,3,8

	beq	0,1f

0:	
				; slwi	9,9,2
				; lfsx	8,5,9

				; lfpsux 4,3,8
				; lfpsux 5,3,8
				; lfpsux 6,3,8
				; lfpsux 7,3,8

	fxpmul	0,8,0
	fxpmul	1,8,1
	fxpmul	2,8,2
	fxpmul	3,8,3

	lhzux	9,4,11

				; stfpsux 0,7,8
				; stfpsux 1,7,8
				; stfpsux 2,7,8
				; stfpsux 3,7,8

				; slwi	10,10,2
				; lfsx	8,5,10

				; lfpsux 0,3,8
				; lfpsux 1,3,8
				; lfpsux 2,3,8
				; lfpsux 3,3,8
	fxpmul	4,8,4	
	fxpmul	5,8,5
	fxpmul	6,8,6
	fxpmul	7,8,7

	lhzux	10,4,11

				; stfpsux 4,7,8
				; stfpsux 5,7,8
				; stfpsux 6,7,8
				; stfpsux 7,7,8

	bdnz	0b

1:	andi.	0,6,1
	beq	0,2f
				; slwi	9,9,2
				; lfsx	8,5,9
	fxpmul	0,8,0
	fxpmul	1,8,1
	fxpmul	2,8,2
	fxpmul	3,8,3
				; stfpsux 0,7,8
				; stfpsux 1,7,8
				; stfpsux 2,7,8
				; stfpsux 3,7,8

2:	blr


.align	5
.global	_weigh_visibilities_4
_weigh_visibilities_4:

	mullw	6,6,7
	li	8,8
	li	11,4
	srwi.	0,6,1
	sub	4,4,11
	mtctr	0
	sub	7,3,8

	lwzux	9,4,11
	lwzux	10,4,11

	lfpsx	0,0,3
	lfpsux	1,3,8
	lfpsux	2,3,8
	lfpsux	3,3,8

	beq	0,1f

0:	
				; slwi	9,9,2
				; lfsx	8,5,9

				; lfpsux 4,3,8
				; lfpsux 5,3,8
				; lfpsux 6,3,8
				; lfpsux 7,3,8

	fxpmul	0,8,0
	fxpmul	1,8,1
	fxpmul	2,8,2
	fxpmul	3,8,3

	lwzux	9,4,11

				; stfpsux 0,7,8
				; stfpsux 1,7,8
				; stfpsux 2,7,8
				; stfpsux 3,7,8

				; slwi	10,10,2
				; lfsx	8,5,10

				; lfpsux 0,3,8
				; lfpsux 1,3,8
				; lfpsux 2,3,8
				; lfpsux 3,3,8
	fxpmul	4,8,4	
	fxpmul	5,8,5
	fxpmul	6,8,6
	fxpmul	7,8,7

	lwzux	10,4,11

				; stfpsux 4,7,8
				; stfpsux 5,7,8
				; stfpsux 6,7,8
				; stfpsux 7,7,8

	bdnz	0b

1:	andi.	0,6,1
	beq	0,2f
				; slwi	9,9,2
				; lfsx	8,5,9
	fxpmul	0,8,0
	fxpmul	1,8,1
	fxpmul	2,8,2
	fxpmul	3,8,3
				; stfpsux 0,7,8
				; stfpsux 1,7,8
				; stfpsux 2,7,8
				; stfpsux 3,7,8

2:	blr

#endif
