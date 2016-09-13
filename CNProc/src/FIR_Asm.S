#  FIR.S: FIR filter assembly for BG/L double Hummer #
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
#  $Id: FIR_Asm.S 18090 2011-05-25 12:25:18Z romein $

#if defined HAVE_BGP
#define CN_PROCESSING
#include <Interface/Config.h>

#define I4COMPLEX_SIZE	 1
#define I8COMPLEX_SIZE	 2
#define I16COMPLEX_SIZE	 4
#define FCOMPLEX_SIZE	 8
#define DCOMPLEX_SIZE	16
#define FLOAT_SIZE	 4

.global	_FIR_constants_used
_FIR_constants_used:
	.long	NR_TAPS
	.long	NR_POLARIZATIONS


.align	5
.global _bgl_mutex_lock
_bgl_mutex_lock:
	sync
0:	lwz	0,0(3)
	cmpwi	0,0
	bne-	0b
	sync
	blr

.global	_bgl_mutex_unlock
_bgl_mutex_unlock:
	sync
	li	0,0
	stw	0,0(3)
	sync
	blr

.global _rdtsc
_rdtsc:
0:	mfspr	3,269
	mfspr	4,268
	mfspr	5,269
	cmpw	5,3
	bne-	0b
	blr
	
.align	5
.global _prefetch
_prefetch:
	mtctr	4

0:
	dcbt	0,3
	add	3,3,5
	bdnz	0b

	blr


.align	5
.global	_transpose_4x8
_transpose_4x8:

	mulli	8,6,3
	mtctr	5
	subi	7,7,3*FCOMPLEX_SIZE
	li	9,8
	subfic	8,8,FCOMPLEX_SIZE

	sub	3,3,7
	sub	4,4,8

L4:	
	lfpsux	0,4,8
	lfpsux	1,4,6
	lfpsux	2,4,6
	lfpsux	3,4,6

#if !defined HAVE_BGP
	dcbz	3,7
#endif
	stfpsux	0,3,7
	stfpsux	1,3,9
	stfpsux	2,3,9
	stfpsux	3,3,9

	bdnz	L4

	blr


.align	5
.global	_phase_shift_and_transpose
_phase_shift_and_transpose:

#	Performs partial transpose between FFT and correlator and multiplies
#	phase shifts on the fly.
#	The outputs of four FFTs are transposed: two time slots for each of
#	the two	polarizations.

#	arguments:
#	r3 :	fcomplex *out
#	r4 :	const fcomplex *in
#	r5 :	const dcomplex phases[2][2]: v0, delta_v0, v1, delta_v1
#	r6 :	stride
#	r7 :	nrChannels
#	r8 :    const float bandpass[nrChannels]

#	f8 :	v0;
#	f9 :	delta_v0
#	f10 :	v1
#	f11 :	delta_v1

	li	9,-16		# save registers
	stfpdux 14,1,9
	stfpdux 15,1,9
	stfpdux 16,1,9
	stfpdux 17,1,9
	stfpdux 18,1,9

	lfpdx	8,0,5
	li	9,16
	lfpdux	9,5,9
	lfpdux	10,5,9
	lfpdux	11,5,9

	lfpsx	18,0,8

	mulli	5,7,24
	srwi	0,7,1		# nrChannels / 2
	slwi	7,7,3		# nrChannels * FCOMPLEX_SIZE
	li	10,FCOMPLEX_SIZE
	mtctr	0

	lfpsx	0,0,4
	lfpsux	1,4,7
	lfpsux	2,4,7
	lfpsux	3,4,7

	sub	3,3,6
	subfic	5,5,8		# FCOMPLEX_SIZE*(1-3*nrChannels)

	fxpmul	12,0,8	
	fxpmul	13,1,8 	
	fxpmul	14,2,10	
	fxpmul	15,3,10	

L5:	
	fxpmul  16,8,9
	fxpmul  17,10,11

	fxcxnpma 0,0,8,12	; lfpsux	4,4,5
	fxcxnpma 1,1,8,13
	fxcxnpma 2,2,10,14	; lfpsux	5,4,7
	fxcxnpma 3,3,10,15

	# correct bandpass
	fxpmul	 0,18,0		; lfpsux	6,4,7
	fxpmul	 1,18,1
	fxpmul	 2,18,2
	fxpmul	 3,18,3		; lfpsux	7,4,7

	fxcxnpma 8,8,9,16
#if defined HAVE_BGP
	fxcxnpma 10,10,11,17
#else
	fxcxnpma 10,10,11,17	; dcbz		3,6
#endif
 
	fxpmul	12,4,8		; stfpsux	0,3,6
	fxpmul	13,5,8		; stfpsux	1,3,10
	fxpmul	14,6,10		; stfpsux	2,3,10
	fxpmul	15,7,10		; stfpsux	3,3,10

	fxpmul  16,8,9
	fxpmul  17,10,11

	fxcxnpma 4,4,8,12	; lfpsux	0,4,5
	fxcxnpma 5,5,8,13
	fxcxnpma 6,6,10,14	; lfpsux	1,4,7
	fxcxnpma 7,7,10,15

	fxcxnpma 8,8,9,16	; lfpsux	2,4,7
#if defined HAVE_BGP
	fxcxnpma 10,10,11,17
#else
	fxcxnpma 10,10,11,17	; dcbz		3,6
#endif
 
	# correct bandpass
	fxsmul	 4,18,4		; lfpsux	3,4,7
	fxsmul	 5,18,5
	fxsmul	 6,18,6
	fxsmul	 7,18,7		; lfpsux	18,8,10

	fxpmul	12,0,8		; stfpsux	4,3,6
	fxpmul	13,1,8 		; stfpsux	5,3,10
	fxpmul	14,2,10		; stfpsux	6,3,10
	fxpmul	15,3,10		; stfpsux	7,3,10

	bdnz	L5

	lfpdx	18,0,1		# restore registers
	lfpdux	17,1,9
	lfpdux	16,1,9
	lfpdux	15,1,9
	lfpdux	14,1,9
	addi	1,1,16

	blr


#	this is a union of all possible constant tables

.global	_ZN5LOFAR4RTCP13_FIR_fp_tableE
.comm	_ZN5LOFAR4RTCP13_FIR_fp_tableE,256*256*FCOMPLEX_SIZE,32



#	filter for i4complex

.global	_ZN5LOFAR4RTCP7_filterINS_5TYPES9i4complexEEEvjPKfPKT_PSt7complexIfEi
_ZN5LOFAR4RTCP7_filterINS_5TYPES9i4complexEEEvjPKfPKT_PSt7complexIfEi:
#	filters all samples for one station, one polarization

#	arguments:
#	r3 :	  nrChannels
#	r4 :	  pointer to weights line (const fcomplex[16])
#	r5 :	  pointer to first sample (const i4complex[16*r7])
#	r6 :	  pointer to result (fcomplex *)
#	r7 :	  number of samples / 16

#	internally used:
#	r9 :	  8
#	r10 :	  512
#	r11 :	  pointer to _ZN5LOFAR4RTCP13_FIR_fp_tableE
#	r28-r31 : prefetched samples
#	f0-f15 :  delay line (real in primary, imaginary in secondary unit)
#	f16-f23 : weights (these are real values alternately stored in primary
#			   and secondary units)
#	f24-f29 : sums

#	The implementation works on 5 or 6 time samples concurrently, to avoid
#	stalls in the double hummer.  This unfortunately leads to totally
#	incomprehensible code.  The loop processes 16 samples at a time.
#	The input is converted from i4complex to dcomplex by a table lookup.

	subi	1,1,272		# adjust stack pointer

	mtctr	7		# set number of iterations

	la	8,16(1)		# save call-saved registers
	li	9,DCOMPLEX_SIZE
	stfpdx	14,0,8
	stfpdux 15,8,9
	stfpdux 16,8,9
	stfpdux 17,8,9
	stfpdux 18,8,9
	stfpdux 19,8,9
	stfpdux 20,8,9
	stfpdux 21,8,9
	stfpdux 22,8,9
	stfpdux 23,8,9
	stfpdux 24,8,9
	stfpdux 25,8,9
	stfpdux 26,8,9
	stfpdux 27,8,9
	stfpdux 28,8,9
	stfpdux 29,8,9
	stmw	28,0(1)		# save r28 ... r31

	# convert 15 i4complex numbers to fcomplex
	slwi	10,3,2	# I4COMPLEX_SIZE*nrChannels*NR_POLARIZATIONS
				; lbzx	 29,0,5	# fetch FIR history samples
	lis     11,_ZN5LOFAR4RTCP13_FIR_fp_tableE@ha	; lbzux  30,5,10
	la      11,_ZN5LOFAR4RTCP13_FIR_fp_tableE@l(11);lbzux	 31,5,10
	slwi	29,29,3		; lbzux	 28,5,10
				; lfpsx  1,11,29
	slwi	30,30,3		; lbzux	 29,5,10
				; lfpsx  2,11,30
	slwi	31,31,3		; lbzux	 30,5,10
				; lfpsx  3,11,31
	slwi	28,28,3		; lbzux	 31,5,10
				; lfpsx  4,11,28
	slwi	29,29,3		; lbzux	 28,5,10
				; lfpsx  5,11,29
	slwi	30,30,3		; lbzux	 29,5,10
				; lfpsx  6,11,30
	slwi	31,31,3		; lbzux	 30,5,10
				; lfpsx  7,11,31
	slwi	28,28,3		; lbzux	 31,5,10
				; lfpsx  8,11,28
	slwi	29,29,3		; lbzux	 28,5,10
				; lfpsx  9,11,29
	slwi	30,30,3		; lbzux	 29,5,10
				; lfpsx  10,11,30
	slwi	31,31,3		; lbzux	 30,5,10
				; lfpsx  11,11,31
	slwi	28,28,3		; lbzux	 31,5,10
	slwi	29,29,3		; lfpsx	 12,11,28
	slwi	30,30,3		; lfpsx	 13,11,29
	slwi	31,31,3		; lfpsx	 14,11,30
				; lfpsx	 15,11,31

	li	9,FCOMPLEX_SIZE ; lfpsx	 16,0,4 # load weights line
	sub	6,6,9		; lfpsux 17,4,9
				; lfpsux 18,4,9
				; lfpsux 19,4,9
				; lfpsux 20,4,9
				; lfpsux 21,4,9
	# essentially part of loop, but interleaved with
	# the stores in the tail of the loop
	fxsmul	 24,16,15	; lfpsux 22,4,9
	fxpmul	 25,17,15	; lfpsux 23,4,9

	fxsmul	 26,17,15	; lbzux  28,5,10	# prefetch samples
	fxpmul	 27,18,15	; lbzux  29,5,10
	fxsmul	 28,18,15	; lbzux  30,5,10
	fxpmul	 29,19,15	; lbzux  31,5,10
0:
	# time steps 0-5

	fxcsmadd 24,23,1,24	; slwi	 28,28,3
	fxcpmadd 25,20,9,25	; lfpsx	 0,11,28
	fxcsmadd 26,20,9,26
	fxcpmadd 27,21,9,27
	fxcsmadd 28,21,9,28
	fxcpmadd 29,22,9,29	; lbzux	 28,5,10

	fxcpmadd 24,23,2,24
	fxcsmadd 25,23,2,25
	fxcpmadd 26,20,10,26
	fxcsmadd 27,20,10,27
	fxcpmadd 28,21,10,28
	fxcsmadd 29,21,10,29

	fxcsmadd 24,22,3,24	; slwi	 29,29,3
	fxcpmadd 25,23,3,25	; lfpsx	 1,11,29
	fxcsmadd 26,23,3,26
	fxcpmadd 27,20,11,27
	fxcsmadd 28,20,11,28
	fxcpmadd 29,21,11,29	; lbzux	 29,5,10

	fxcpmadd 24,22,4,24
	fxcsmadd 25,22,4,25
	fxcpmadd 26,23,4,26
	fxcsmadd 27,23,4,27
	fxcpmadd 28,20,12,28
	fxcsmadd 29,20,12,29

	fxcsmadd 24,21,5,24	; slwi	 30,30,3
	fxcpmadd 25,22,5,25	; lfpsx	 2,11,30
	fxcsmadd 26,22,5,26
	fxcpmadd 27,23,5,27
	fxcsmadd 28,23,5,28
	fxcpmadd 29,20,13,29	; lbzux	 30,5,10

	fxcpmadd 24,21,6,24
	fxcsmadd 25,21,6,25
	fxcpmadd 26,22,6,26
	fxcsmadd 27,22,6,27
	fxcpmadd 28,23,6,28
	fxcsmadd 29,23,6,29
	
	fxcsmadd 24,20,7,24	; slwi	 31,31,3
	fxcpmadd 25,21,7,25	; lfpsx	 3,11,31
	fxcsmadd 26,21,7,26
	fxcpmadd 27,22,7,27
	fxcsmadd 28,22,7,28
	fxcpmadd 29,23,7,29	; lbzux	 31,5,10

	fxcpmadd 24,20,8,24
	fxcsmadd 25,20,8,25
	fxcpmadd 26,21,8,26
	fxcsmadd 27,21,8,27
	fxcpmadd 28,22,8,28
	fxcsmadd 29,22,8,29

	fxcpmadd 24,17,14,24	; slwi	 28,28,3
	fxcsmadd 25,17,14,25	; lfpsx	 4,11,28
	fxcpmadd 26,18,14,26
	fxcsmadd 27,18,14,27
	fxcpmadd 28,19,14,28
	fxcsmadd 29,19,14,29	; lbzux	 28,5,10
	
	fxcpmadd 24,16,0,24
	fxcsmadd 25,16,0,25
	fxcpmadd 26,17,0,26
	fxcsmadd 27,17,0,27
	fxcpmadd 28,18,0,28
	fxcsmadd 29,18,0,29

	fxcsmadd 24,19,9,24	; slwi	 29,29,3
	fxcpmadd 25,16,1,25	; lfpsx	 5,11,29
	fxcsmadd 26,16,1,26
	fxcpmadd 27,17,1,27
	fxcsmadd 28,17,1,28
	fxcpmadd 29,18,1,29	; lbzux	 29,5,10

	fxcpmadd 24,19,10,24
	fxcsmadd 25,19,10,25
	fxcpmadd 26,16,2,26
	fxcsmadd 27,16,2,27
	fxcpmadd 28,17,2,28
	fxcsmadd 29,17,2,29

	fxcsmadd 24,18,11,24
	fxcpmadd 25,19,11,25
	fxcsmadd 26,19,11,26
	fxcpmadd 27,16,3,27
	fxcsmadd 28,16,3,28
	fxcpmadd 29,17,3,29

	fxcpmadd 24,18,12,24
	fxcsmadd 25,18,12,25
	fxcpmadd 26,19,12,26
	fxcsmadd 27,19,12,27
	fxcpmadd 28,16,4,28
	fxcsmadd 29,16,4,29

	fxcsmadd 24,17,13,24
	fxcpmadd 25,18,13,25
	fxcsmadd 26,18,13,26
	fxcpmadd 27,19,13,27
	fxcsmadd 28,19,13,28	; stfpsux 24,6,9
	fxcpmadd 29,16,5,29	; stfpsux 25,6,9


	# time steps 6-10

	fxsmul	 24,21,11	; stfpsux 26,6,9
	fxpmul	 25,22,11	; stfpsux 27,6,9
	fxsmul	 26,22,11	; stfpsux 28,6,9
	fxpmul	 27,23,11	; stfpsux 29,6,9
	fxsmul	 28,23,11

	fxcpmadd 24,17,4,24
	fxcsmadd 25,17,4,25
	fxcpmadd 26,18,4,26
	fxcsmadd 27,18,4,27
	fxcpmadd 28,19,4,28
	
	fxcsmadd 24,20,13,24	; slwi	 30,30,3
	fxcpmadd 25,21,13,25	; lfpsx	 6,11,30
	fxcsmadd 26,21,13,26	; lbzux	 30,5,10
	fxcpmadd 27,22,13,27
	fxcsmadd 28,22,13,28

	fxcsmadd 24,16,5,24
	fxcpmadd 25,17,5,25
	fxcsmadd 26,17,5,26
	fxcpmadd 27,18,5,27
	fxcsmadd 28,18,5,28
	
	fxcpmadd 24,20,14,24
	fxcsmadd 25,20,14,25
	fxcpmadd 26,21,14,26
	fxcsmadd 27,21,14,27
	fxcpmadd 28,22,14,28

	fxcpmadd 24,16,6,24
	fxcsmadd 25,16,6,25
	fxcpmadd 26,17,6,26
	fxcsmadd 27,17,6,27
	fxcpmadd 28,18,6,28	; slwi	 31,31,3

	fxcsmadd 24,23,7,24	; lfpsx	 7,11,31
	fxcpmadd 25,20,15,25	; lbzux	 31,5,10
	fxcsmadd 26,20,15,26
	fxcpmadd 27,21,15,27
	fxcsmadd 28,21,15,28

	fxcsmadd 24,19,15,24
	fxcpmadd 25,16,7,25
	fxcsmadd 26,16,7,26
	fxcpmadd 27,17,7,27
	fxcsmadd 28,17,7,28

	fxcpmadd 24,23,8,24	; slwi	 28,28,3
	fxcsmadd 25,23,8,25	; lfpsx	 8,11,28
	fxcpmadd 26,20,0,26	; lbzux	 28,5,10
	fxcsmadd 27,20,0,27
	fxcpmadd 28,21,0,28

	fxcpmadd 24,19,0,24
	fxcsmadd 25,19,0,25
	fxcpmadd 26,16,8,26
	fxcsmadd 27,16,8,27
	fxcpmadd 28,17,8,28

	fxcsmadd 24,17,3,24
	fxcpmadd 25,18,3,25
	fxcsmadd 26,18,3,26
	fxcpmadd 27,19,3,27
	fxcsmadd 28,19,3,28

	fxcsmadd 24,22,9,24
	fxcpmadd 25,23,9,25	; slwi	 29,29,3
	fxcsmadd 26,23,9,26	; lfpsx	 9,11,29
	fxcpmadd 27,20,1,27	; lbzux	 29,5,10
	fxcsmadd 28,20,1,28

	fxcsmadd 24,18,1,24
	fxcpmadd 25,19,1,25
	fxcsmadd 26,19,1,26
	fxcpmadd 27,16,9,27
	fxcsmadd 28,16,9,28

	fxcpmadd 24,21,12,24
	fxcsmadd 25,21,12,25
	fxcpmadd 26,22,12,26
	fxcsmadd 27,22,12,27
	fxcpmadd 28,23,12,28

	fxcpmadd 24,22,10,24
	fxcsmadd 25,22,10,25
	fxcpmadd 26,23,10,26	; slwi	 30,30,3
	fxcsmadd 27,23,10,27	; lfpsx	 10,11,30
	fxcpmadd 28,20,2,28	; lbzux	 30,5,10

	fxcpmadd 24,18,2,24
	fxcsmadd 25,18,2,25
	fxcpmadd 26,19,2,26
	fxcsmadd 27,19,2,27
	fxcpmadd 28,16,10,28


	# time steps 11-15
				; stfpsux 24,6,9
	fxsmul	 24,21,0	; stfpsux 25,6,9
	fxpmul	 25,22,0	; stfpsux 26,6,9
	fxsmul	 26,22,0	; stfpsux 27,6,9
	fxpmul	 27,23,0	; stfpsux 28,6,9
	fxsmul	 28,23,0

	fxcpmadd 24,17,9,24
	fxcsmadd 25,17,9,25
	fxcpmadd 26,18,9,26
	fxcsmadd 27,18,9,27
	fxcpmadd 28,19,9,28
	
	fxcsmadd 24,20,2,24	; slwi	 31,31,3
	fxcpmadd 25,21,2,25	; lfpsx	 11,11,31
	fxcsmadd 26,21,2,26	; lbzux	 31,5,10
	fxcpmadd 27,22,2,27
	fxcsmadd 28,22,2,28

	fxcsmadd 24,16,10,24
	fxcpmadd 25,17,10,25
	fxcsmadd 26,17,10,26
	fxcpmadd 27,18,10,27
	fxcsmadd 28,18,10,28

	fxcpmadd 24,20,3,24
	fxcsmadd 25,20,3,25
	fxcpmadd 26,21,3,26
	fxcsmadd 27,21,3,27
	fxcpmadd 28,22,3,28
	
	fxcpmadd 24,16,11,24
	fxcsmadd 25,16,11,25
	fxcpmadd 26,17,11,26
	fxcsmadd 27,17,11,27
	fxcpmadd 28,18,11,28	; slwi	 28,28,3

	fxcsmadd 24,23,12,24	; lfpsx	 12,11,28
	fxcpmadd 25,20,4,25	; lbzux	 28,5,10
	fxcsmadd 26,20,4,26
	fxcpmadd 27,21,4,27
	fxcsmadd 28,21,4,28

	fxcsmadd 24,19,4,24
	fxcpmadd 25,16,12,25
	fxcsmadd 26,16,12,26
	fxcpmadd 27,17,12,27
	fxcsmadd 28,17,12,28

	fxcpmadd 24,23,13,24	; slwi	 29,29,3
	fxcsmadd 25,23,13,25	; lfpsx	 13,11,29
	fxcpmadd 26,20,5,26	; lbzux	 29,5,10
	fxcsmadd 27,20,5,27
	fxcpmadd 28,21,5,28
	
	fxcsmadd 24,17,8,24
	fxcpmadd 25,18,8,25
	fxcsmadd 26,18,8,26
	fxcpmadd 27,19,8,27
	fxcsmadd 28,19,8,28

	fxcpmadd 24,19,5,24
	fxcsmadd 25,19,5,25
	fxcpmadd 26,16,13,26
	fxcsmadd 27,16,13,27
	fxcpmadd 28,17,13,28

	fxcsmadd 24,22,14,24
	fxcpmadd 25,23,14,25	; slwi	 30,30,3
	fxcsmadd 26,23,14,26	; lfpsx	 14,11,30
	fxcpmadd 27,20,6,27	; lbzux	 30,5,10
	fxcsmadd 28,20,6,28
	
	fxcsmadd 24,18,6,24
	fxcpmadd 25,19,6,25
	fxcsmadd 26,19,6,26
	fxcpmadd 27,16,14,27
	fxcsmadd 28,16,14,28

	fxcpmadd 24,21,1,24
	fxcsmadd 25,21,1,25
	fxcpmadd 26,22,1,26
	fxcsmadd 27,22,1,27
	fxcpmadd 28,23,1,28

	fxcpmadd 24,22,15,24
	fxcsmadd 25,22,15,25
	fxcpmadd 26,23,15,26	; slwi	 31,31,3
	fxcsmadd 27,23,15,27	; lfpsx	 15,11,31
	fxcpmadd 28,20,7,28	; lbzux	 31,5,10

	fxcpmadd 24,18,7,24
	fxcsmadd 25,18,7,25
	fxcpmadd 26,19,7,26
	fxcsmadd 27,19,7,27
	fxcpmadd 28,16,15,28

	fxpmul	 29,19,15	; stfpsux 24,6,9
	fxsmul	 24,16,15	; stfpsux 25,6,9
	fxpmul	 25,17,15	; stfpsux 26,6,9
	fxsmul	 26,17,15	; stfpsux 27,6,9
	fxpmul	 27,18,15	; stfpsux 28,6,9
	fxsmul	 28,18,15

	bdnz	0b

	la	8,16(1)	# restore call-saved registers
	li	9,DCOMPLEX_SIZE

	lfpdx	14,0,8
	lfpdux	15,8,9
	lfpdux	16,8,9
	lfpdux	17,8,9
	lfpdux	18,8,9
	lfpdux	19,8,9
	lfpdux	20,8,9
	lfpdux	21,8,9
	lfpdux	22,8,9
	lfpdux	23,8,9
	lfpdux	24,8,9
	lfpdux	25,8,9
	lfpdux	26,8,9
	lfpdux	27,8,9
	lfpdux	28,8,9
	lfpdux	29,8,9
	lmw	28,0(1)		# restore r28 ... r31

	addi	1,1,272		# restore stack pointer
	blr			# return


# 	filter for i8complex

.global	_ZN5LOFAR4RTCP7_filterISt7complexIaEEEvjPKfPKT_PS2_IfEi
_ZN5LOFAR4RTCP7_filterISt7complexIaEEEvjPKfPKT_PS2_IfEi:
#	filters all samples for one station, one polarization

#	arguments:
#	r3 :	  nrChannels
#	r4 :	  pointer to weights line (const fcomplex[16])
#	r5 :	  pointer to first sample (const i8complex[16*r7])
#	r6 :	  pointer to result (fcomplex *)
#	r7 :	  number of samples / 16

#	internally used:
#	r9 :	  8
#	r10 :	  1024
#	r11 :	  ptr to constant table
#	r28-r31 : prefetched samples
#	f0-f15 :  delay line (real in primary, imaginary in secondary unit)
#	f16-f23 : weights (these are real values alternately stored in primary
#			   and secondary units)
#	f24-f29 : sums

#	The implementation works on 5 or 6 time samples concurrently, to avoid
#	stalls in the double hummer.  This unfortunately leads to totally
#	incomprehensible code.  The loop processes 16 samples at a time.
#	The input is converted from int8complex to dcomplex by black magic,
#	making the code even harder to understand.

	mtctr	7		# set number of iterations

	li	9,-DCOMPLEX_SIZE
	stfpdux 14,1,9		# save call-saved registers
	stfpdux 15,1,9
	stfpdux 16,1,9
	stfpdux 17,1,9
	stfpdux 18,1,9
	stfpdux 19,1,9
	stfpdux 20,1,9
	stfpdux 21,1,9
	stfpdux 22,1,9
	stfpdux 23,1,9
	stfpdux 24,1,9
	stfpdux 25,1,9
	stfpdux 26,1,9
	stfpdux 27,1,9
	stfpdux 28,1,9
	stfpdux 29,1,9
	stfpdux 30,1,9
	stfpdux 31,1,9

	subi	1,1,16
	stmw	28,0(1)	# save r28 ... r31

	lis	11,_ZN5LOFAR4RTCP13_FIR_fp_tableE@ha
	li	9,FCOMPLEX_SIZE
	la	11,_ZN5LOFAR4RTCP13_FIR_fp_tableE@l(11)
	slwi	10,3,2	#I8COMPLEX_SIZE*nrChannels*NR_POLARIZATIONS
	subi	6,6,FCOMPLEX_SIZE


#if 0
	lfpsx	1,0,3		# load delay line
	lfpsux	2,3,9
	lfpsux	3,3,9
	lfpsux	4,3,9
	lfpsux	5,3,9
	lfpsux	6,3,9
	lfpsux	7,3,9
	lfpsux	8,3,9
	lfpsux	9,3,9
	lfpsux	10,3,9
	lfpsux	11,3,9
	lfpsux	12,3,9
	lfpsux	13,3,9
	lfpsux	14,3,9
	lfpsux	15,3,9
#endif

	# convert 15 i8complex numbers to fcomplex
	lhzx	29,0,5		# fetch FIR history samples
	lhzux	30,5,10
	lhzux	31,5,10
	lhzux	28,5,10

	rlwinm	12,29,3,13,28
	lfpsx	1,11,12
	lhzux	29,5,10

	rlwinm	12,30,3,13,28
	lfpsx	2,11,12
	lhzux	30,5,10

	rlwinm	12,31,3,13,28
	lfpsx	3,11,12
	lhzux	31,5,10

	rlwinm	12,28,3,13,28
	lfpsx	4,11,12
	lhzux	28,5,10

	rlwinm	12,29,3,13,28
	lfpsx	5,11,12
	lhzux	29,5,10

	rlwinm	12,30,3,13,28
	lfpsx	6,11,12
	lhzux	30,5,10

	rlwinm	12,31,3,13,28
	lfpsx	7,11,12
	lhzux	31,5,10

	rlwinm	12,28,3,13,28
	lfpsx	8,11,12
	lhzux	28,5,10

	rlwinm	12,29,3,13,28
	lfpsx	9,11,12
	lhzux	29,5,10

	rlwinm	12,30,3,13,28
	lfpsx	10,11,12
	lhzux	30,5,10

	rlwinm	12,31,3,13,28
	lfpsx	11,11,12
	lhzux	31,5,10

	rlwinm	12,28,3,13,28
	lfpsx	12,11,12
	lhzux	28,5,10 # prefetch samples before entering loop

	rlwinm	12,29,3,13,28
	lfpsx	13,11,12
	lhzux	29,5,10

	rlwinm	12,30,3,13,28
	lfpsx	14,11,12
	lhzux	30,5,10

	rlwinm	12,31,3,13,28
	lfpsx	15,11,12
	lhzux	31,5,10

				; lfpsx	  16,0,4	# load weights line
				; lfpsux  17,4,9
				; lfpsux  18,4,9
				; lfpsux  19,4,9
				; lfpsux  20,4,9
				; lfpsux  21,4,9
				; lfpsux  22,4,9
				; lfpsux  23,4,9

	# essentially part of loop, but interleaved with
	# the stores in the tail of the loop
	fxpmul	 24,20,8
	fxsmul	 25,20,8
	fxpmul	 26,21,8
	fxsmul	 27,21,8
	fxpmul	 28,22,8
	fxsmul	 29,22,8
	fxpmul	 30,23,8
	fxsmul	 31,23,8

0:
	# time steps 0-7

	# load 0

	fxcsmadd 24,23,1,24	; rlwinm 12,28,3,13,28
	# load 1
	fxcpmadd 25,20,9,25	; lfpsx  0,11,12
	fxcsmadd 26,20,9,26	; lhzux	 28,5,10
	fxcpmadd 27,21,9,27
	fxcsmadd 28,21,9,28
	fxcpmadd 29,22,9,29
	fxcsmadd 30,22,9,30
	fxcpmadd 31,23,9,31

	fxcpmadd 24,23,2,24	; rlwinm 12,29,3,13,28
	fxcsmadd 25,23,2,25	; lfpsx  1,11,12
	# load 2
	fxcpmadd 26,20,10,26	; lhzux	 29,5,10
	fxcsmadd 27,20,10,27
	fxcpmadd 28,21,10,28
	fxcsmadd 29,21,10,29
	fxcpmadd 30,22,10,30
	fxcsmadd 31,22,10,31

	fxcsmadd 24,22,3,24	; rlwinm 12,30,3,13,28
	fxcpmadd 25,23,3,25	; lfpsx  2,11,12
	fxcsmadd 26,23,3,26	; lhzux	 30,5,10
	# load 3
	fxcpmadd 27,20,11,27
	fxcsmadd 28,20,11,28
	fxcpmadd 29,21,11,29
	fxcsmadd 30,21,11,30
	fxcpmadd 31,22,11,31

	fxcpmadd 24,22,4,24	; rlwinm 12,31,3,13,28
	fxcsmadd 25,22,4,25	; lfpsx  3,11,12
	fxcpmadd 26,23,4,26	; lhzux	 31,5,10
	fxcsmadd 27,23,4,27
	# load 4
	fxcpmadd 28,20,12,28
	fxcsmadd 29,20,12,29
	fxcpmadd 30,21,12,30
	fxcsmadd 31,21,12,31

	fxcsmadd 24,21,5,24
	fxcpmadd 25,22,5,25
	fxcsmadd 26,22,5,26
	fxcpmadd 27,23,5,27
	fxcsmadd 28,23,5,28
	# load 5
	fxcpmadd 29,20,13,29
	fxcsmadd 30,20,13,30
	fxcpmadd 31,21,13,31

	fxcpmadd 24,21,6,24
	fxcsmadd 25,21,6,25
	fxcpmadd 26,22,6,26
	fxcsmadd 27,22,6,27
	fxcpmadd 28,23,6,28
	fxcsmadd 29,23,6,29
	# load 6
	fxcpmadd 30,20,14,30
	fxcsmadd 31,20,14,31

	fxcsmadd 24,20,7,24
	fxcpmadd 25,21,7,25
	fxcsmadd 26,21,7,26
	fxcpmadd 27,22,7,27	; rlwinm 12,28,3,13,28
	fxcsmadd 28,22,7,28	; lfpsx	 4,11,12
	fxcpmadd 29,23,7,29	; lhzux	 28,5,10
	fxcsmadd 30,23,7,30
	# load 7
	fxcpmadd 31,20,15,31

	# after loads
	fxcpmadd 24,16,0,24
	fxcsmadd 25,16,0,25
	fxcpmadd 26,17,0,26
	fxcsmadd 27,17,0,27
	fxcpmadd 28,18,0,28
	fxcsmadd 29,18,0,29
	fxcpmadd 30,19,0,30
	fxcsmadd 31,19,0,31	; rlwinm 12,29,3,13,28

	fxcsmadd 24,19,9,24	; lfpsx  5,11,12
	fxcpmadd 25,16,1,25	; lhzux	 29,5,10
	fxcsmadd 26,16,1,26
	fxcpmadd 27,17,1,27
	fxcsmadd 28,17,1,28
	fxcpmadd 29,18,1,29
	fxcsmadd 30,18,1,30
	fxcpmadd 31,19,1,31

	fxcpmadd 24,19,10,24
	fxcsmadd 25,19,10,25
	fxcpmadd 26,16,2,26
	fxcsmadd 27,16,2,27	; rlwinm 12,30,3,13,28
	fxcpmadd 28,17,2,28	; lfpsx  6,11,12
	fxcsmadd 29,17,2,29	; lhzux	 30,5,10
	fxcpmadd 30,18,2,30
	fxcsmadd 31,18,2,31

	fxcsmadd 24,18,11,24
	fxcpmadd 25,19,11,25
	fxcsmadd 26,19,11,26
	fxcpmadd 27,16,3,27
	fxcsmadd 28,16,3,28
	fxcpmadd 29,17,3,29
	fxcsmadd 30,17,3,30
	fxcpmadd 31,18,3,31	; rlwinm 12,31,3,13,28
	
	fxcpmadd 24,18,12,24	; lfpsx  7,11,12
	fxcsmadd 25,18,12,25	; lhzux	 31,5,10
	fxcpmadd 26,19,12,26
	fxcsmadd 27,19,12,27
	fxcpmadd 28,16,4,28
	fxcsmadd 29,16,4,29
	fxcpmadd 30,17,4,30
	fxcsmadd 31,17,4,31

	fxcsmadd 24,17,13,24
	fxcpmadd 25,18,13,25
	fxcsmadd 26,18,13,26
	fxcpmadd 27,19,13,27
	fxcsmadd 28,19,13,28
	fxcpmadd 29,16,5,29
	fxcsmadd 30,16,5,30
	fxcpmadd 31,17,5,31	; rlwinm 12,28,3,13,28

	fxcpmadd 24,17,14,24	; lfpsx  8,11,12
	fxcsmadd 25,17,14,25	; lhzux	 28,5,10
	fxcpmadd 26,18,14,26
	fxcsmadd 27,18,14,27
	fxcpmadd 28,19,14,28
	fxcsmadd 29,19,14,29
	fxcpmadd 30,16,6,30
	fxcsmadd 31,16,6,31

	fxcsmadd 24,16,15,24
	fxcpmadd 25,17,15,25
	fxcsmadd 26,17,15,26
	fxcpmadd 27,18,15,27
	fxcsmadd 28,18,15,28
	fxcpmadd 29,19,15,29	; stfpsux 24,6,9
	fxcsmadd 30,19,15,30	; stfpsux 25,6,9
	fxcpmadd 31,16,7,31	; stfpsux 26,6,9

	
	# time steps 8-15
	fxpmul	 24,20,0
	fxsmul	 25,20,0
	fxpmul	 26,21,0	; stfpsux 27,6,9
	fxsmul	 27,21,0	; stfpsux 28,6,9
	fxpmul	 28,22,0	; stfpsux 29,6,9
	fxsmul	 29,22,0	; stfpsux 30,6,9
	fxpmul	 30,23,0	; stfpsux 31,6,9
	fxsmul	 31,23,0

	# already loaded 8

	fxcsmadd 24,23,9,24
	# load 9
	fxcpmadd 25,20,1,25
	fxcsmadd 26,20,1,26
	fxcpmadd 27,21,1,27
	fxcsmadd 28,21,1,28
	fxcpmadd 29,22,1,29
	fxcsmadd 30,22,1,30
	fxcpmadd 31,23,1,31

	fxcpmadd 24,23,10,24
	fxcsmadd 25,23,10,25
	# load 10
	fxcpmadd 26,20,2,26
	fxcsmadd 27,20,2,27
	fxcpmadd 28,21,2,28
	fxcsmadd 29,21,2,29
	fxcpmadd 30,22,2,30
	fxcsmadd 31,22,2,31	; rlwinm 12,29,3,13,28

	fxcsmadd 24,22,11,24	; lfpsx  9,11,12
	fxcpmadd 25,23,11,25	; lhzux	 29,5,10
	fxcsmadd 26,23,11,26
	# load 11
	fxcpmadd 27,20,3,27
	fxcsmadd 28,20,3,28
	fxcpmadd 29,21,3,29
	fxcsmadd 30,21,3,30
	fxcpmadd 31,22,3,31

	fxcpmadd 24,22,12,24
	fxcsmadd 25,22,12,25
	fxcpmadd 26,23,12,26
	fxcsmadd 27,23,12,27	; rlwinm 12,30,3,13,28
	# load 12
	fxcpmadd 28,20,4,28	; lfpsx  10,11,12
	fxcsmadd 29,20,4,29	; lhzux	 30,5,10
	fxcpmadd 30,21,4,30
	fxcsmadd 31,21,4,31

	fxcsmadd 24,21,13,24
	fxcpmadd 25,22,13,25
	fxcsmadd 26,22,13,26
	fxcpmadd 27,23,13,27
	fxcsmadd 28,23,13,28
	# load 13
	fxcpmadd 29,20,5,29
	fxcsmadd 30,20,5,30
	fxcpmadd 31,21,5,31	; rlwinm 12,31,3,13,28

	fxcpmadd 24,21,14,24	; lfpsx  11,11,12
	fxcsmadd 25,21,14,25	; lhzux	 31,5,10
	fxcpmadd 26,22,14,26
	fxcsmadd 27,22,14,27
	fxcpmadd 28,23,14,28
	fxcsmadd 29,23,14,29
	# load 14
	fxcpmadd 30,20,6,30
	fxcsmadd 31,20,6,31

	fxcsmadd 24,20,15,24
	fxcpmadd 25,21,15,25
	fxcsmadd 26,21,15,26
	fxcpmadd 27,22,15,27	; rlwinm 12,28,3,13,28
	fxcsmadd 28,22,15,28	; lfpsx  12,11,12
	fxcpmadd 29,23,15,29	; lhzux	 28,5,10
	fxcsmadd 30,23,15,30
	# load 15
	fxcpmadd 31,20,7,31

	# after loads

	fxcpmadd 24,16,8,24
	fxcsmadd 25,16,8,25
	fxcpmadd 26,17,8,26
	fxcsmadd 27,17,8,27
	fxcpmadd 28,18,8,28
	fxcsmadd 29,18,8,29
	fxcpmadd 30,19,8,30
	fxcsmadd 31,19,8,31	; rlwinm 12,29,3,13,28

	fxcsmadd 24,19,1,24	; lfpsx  13,11,12
	fxcpmadd 25,16,9,25	; lhzux	 29,5,10
	fxcsmadd 26,16,9,26
	fxcpmadd 27,17,9,27
	fxcsmadd 28,17,9,28
	fxcpmadd 29,18,9,29
	fxcsmadd 30,18,9,30
	fxcpmadd 31,19,9,31

	fxcpmadd 24,19,2,24
	fxcsmadd 25,19,2,25
	fxcpmadd 26,16,10,26
	fxcsmadd 27,16,10,27	; rlwinm 12,30,3,13,28
	fxcpmadd 28,17,10,28	; lfpsx  14,11,12
	fxcsmadd 29,17,10,29	; lhzux	 30,5,10
	fxcpmadd 30,18,10,30
	fxcsmadd 31,18,10,31

	fxcsmadd 24,18,3,24
	fxcpmadd 25,19,3,25
	fxcsmadd 26,19,3,26
	fxcpmadd 27,16,11,27
	fxcsmadd 28,16,11,28
	fxcpmadd 29,17,11,29
	fxcsmadd 30,17,11,30
	fxcpmadd 31,18,11,31	; rlwinm 12,31,3,13,28

	fxcpmadd 24,18,4,24	; lfpsx  15,11,12
	fxcsmadd 25,18,4,25	; lhzux	 31,5,10
	fxcpmadd 26,19,4,26
	fxcsmadd 27,19,4,27
	fxcpmadd 28,16,12,28
	fxcsmadd 29,16,12,29
	fxcpmadd 30,17,12,30
	fxcsmadd 31,17,12,31
	
	fxcsmadd 24,17,5,24
	fxcpmadd 25,18,5,25
	fxcsmadd 26,18,5,26
	fxcpmadd 27,19,5,27
	fxcsmadd 28,19,5,28
	fxcpmadd 29,16,13,29
	fxcsmadd 30,16,13,30
	fxcpmadd 31,17,13,31

	fxcpmadd 24,17,6,24
	fxcsmadd 25,17,6,25
	fxcpmadd 26,18,6,26
	fxcsmadd 27,18,6,27
	fxcpmadd 28,19,6,28
	fxcsmadd 29,19,6,29
	fxcpmadd 30,16,14,30
	fxcsmadd 31,16,14,31

	fxcsmadd 24,16,7,24
	fxcpmadd 25,17,7,25
	fxcsmadd 26,17,7,26
	fxcpmadd 27,18,7,27
	fxcsmadd 28,18,7,28
	fxcpmadd 29,19,7,29	; stfpsux 24,6,9
	fxcsmadd 30,19,7,30	; stfpsux 25,6,9
	fxcpmadd 31,16,15,31	; stfpsux 26,6,9

	fxpmul	 24,20,8	# part of next loop
	fxsmul	 25,20,8
	fxpmul	 26,21,8	; stfpsux 27,6,9
	fxsmul	 27,21,8	; stfpsux 28,6,9
	fxpmul	 28,22,8	; stfpsux 29,6,9
	fxsmul	 29,22,8	; stfpsux 30,6,9
	fxpmul	 30,23,8	; stfpsux 31,6,9
	fxsmul	 31,23,8

	bdnz	 0b

#if 0
	addi	3,3,-120	# store delay line
	stfpsux	1,3,9
	stfpsux	2,3,9
	stfpsux	3,3,9
	stfpsux	4,3,9
	stfpsux	5,3,9
	stfpsux	6,3,9
	stfpsux	7,3,9
	stfpsux	8,3,9
	stfpsux	9,3,9
	stfpsux	10,3,9
	stfpsux	11,3,9
	stfpsux	12,3,9
	stfpsux	13,3,9
	stfpsux	14,3,9
	stfpsux	15,3,9
#endif

	li	9,DCOMPLEX_SIZE	# restore call-saved registers
	lmw	28,0(1)	# restore r28 ... r31

	#addi	1,1,16
	lfpdux	31,1,9
	lfpdux	30,1,9
	lfpdux	29,1,9
	lfpdux	28,1,9
	lfpdux	27,1,9
	lfpdux	26,1,9
	lfpdux	25,1,9
	lfpdux	24,1,9
	lfpdux	23,1,9
	lfpdux	22,1,9
	lfpdux	21,1,9
	lfpdux	20,1,9
	lfpdux	19,1,9
	lfpdux	18,1,9
	lfpdux	17,1,9
	lfpdux	16,1,9
	lfpdux	15,1,9
	lfpdux	14,1,9

	addi	1,1,16		# restore stack pointer
	blr			# return

#if 0

#	filter for i16complex

.global	_ZN5LOFAR4RTCP7_filterISt7complexIsEEEvjPKfPKT_PS2_IfEi
_ZN5LOFAR4RTCP7_filterISt7complexIsEEEvjPKfPKT_PS2_IfEi:
#	filters all samples for one station, one polarization

#	arguments:
#	r3 :	  nrChannels
#	r4 :	  pointer to weights line (const fcomplex[16])
#	r5 :	  pointer to first sample (const i16complex[16*r7])
#	r6 :	  pointer to result (fcomplex *)
#	r7 :	  number of samples / 16

#	internally used:
#	r9 :	  8
#	r10 :	  2048
#	r11 :	  ptr to constant table
#	r28-r31 : prefetched samples
#	f0-f15 :  delay line (real in primary, imaginary in secondary unit)
#	f16-f23 : weights (these are real values alternately stored in primary
#			   and secondary units)
#	f24-f29 : sums

#	The implementation works on 5 or 6 time samples concurrently, to avoid
#	stalls in the double hummer.  This unfortunately leads to totally
#	incomprehensible code.  The loop processes 16 samples at a time.
#	The input is converted from int16complex to dcomplex by black magic,
#	making the code even harder to understand.

	mtctr	7		# set number of iterations

	li	9,-DCOMPLEX_SIZE
	stfpdux 14,1,9		# save call-saved registers
	stfpdux 15,1,9
	stfpdux 16,1,9
	stfpdux 17,1,9
	stfpdux 18,1,9
	stfpdux 19,1,9
	stfpdux 20,1,9
	stfpdux 21,1,9
	stfpdux 22,1,9
	stfpdux 23,1,9
	stfpdux 24,1,9
	stfpdux 25,1,9
	stfpdux 26,1,9
	stfpdux 27,1,9
	stfpdux 28,1,9
	stfpdux 29,1,9
	stfpdux 30,1,9
	stfpdux 31,1,9

	subi	1,1,16
	stmw	28,0(1)	# save r28 ... r31

	lis	11,_ZN5LOFAR4RTCP13_FIR_fp_tableE@ha
	li	9,FCOMPLEX_SIZE
	la	11,_ZN5LOFAR4RTCP13_FIR_fp_tableE@l(11)
	slwi	10,3,3	# I16COMPLEX_SIZE*nrChannels*NR_POLARIZATIONS
	subi	6,6,FCOMPLEX_SIZE


	# convert 15 i16complex numbers to fcomplex
	lwzx	29,0,5		# fetch FIR history samples
	lwzux	30,5,10
	lwzux	31,5,10
	lwzux	28,5,10

	rlwinm	12,29,2,14,29
	lfssx	1,11,12
	rlwinm	12,29,18,14,29
	lfsx	1,11,12
	lwzux	29,5,10

	rlwinm	12,30,2,14,29
	lfssx	2,11,12
	rlwinm	12,30,18,14,29
	lfsx	2,11,12
	lwzux	30,5,10

	rlwinm	12,31,2,14,29
	lfssx	3,11,12
	rlwinm	12,31,18,14,29
	lfsx	3,11,12
	lwzux	31,5,10

	rlwinm	12,28,2,14,29
	lfssx	4,11,12
	rlwinm	12,28,18,14,29
	lfsx	4,11,12
	lwzux	28,5,10

	rlwinm	12,29,2,14,29
	lfssx	5,11,12
	rlwinm	12,29,18,14,29
	lfsx	5,11,12
	lwzux	29,5,10

	rlwinm	12,30,2,14,29
	lfssx	6,11,12
	rlwinm	12,30,18,14,29
	lfsx	6,11,12
	lwzux	30,5,10

	rlwinm	12,31,2,14,29
	lfssx	7,11,12
	rlwinm	12,31,18,14,29
	lfsx	7,11,12
	lwzux	31,5,10

	rlwinm	12,28,2,14,29
	lfssx	8,11,12
	rlwinm	12,28,18,14,29
	lfsx	8,11,12
	lwzux	28,5,10

	rlwinm	12,29,2,14,29
	lfssx	9,11,12
	rlwinm	12,29,18,14,29
	lfsx	9,11,12
	lwzux	29,5,10

	rlwinm	12,30,2,14,29
	lfssx	10,11,12
	rlwinm	12,30,18,14,29
	lfsx	10,11,12
	lwzux	30,5,10

	rlwinm	12,31,2,14,29
	lfssx	11,11,12
	rlwinm	12,31,18,14,29
	lfsx	11,11,12
	lwzux	31,5,10

	rlwinm	12,28,2,14,29
	lfssx	12,11,12
	rlwinm	12,28,18,14,29
	lfsx	12,11,12
	lwzux	28,5,10 # prefetch samples before entering loop

	rlwinm	12,29,2,14,29
	lfssx	13,11,12
	rlwinm	12,29,18,14,29
	lfsx	13,11,12
	lwzux	29,5,10

	rlwinm	12,30,2,14,29
	lfssx	14,11,12
	rlwinm	12,30,18,14,29
	lfsx	14,11,12
	lwzux	30,5,10

	rlwinm	12,31,2,14,29
	lfssx	15,11,12
	rlwinm	12,31,18,14,29
	lfsx	15,11,12
	lwzux	31,5,10

				; lfpsx	  16,0,4	# load weights line
				; lfpsux  17,4,9
				; lfpsux  18,4,9
				; lfpsux  19,4,9
				; lfpsux  20,4,9
				; lfpsux  21,4,9
				; lfpsux  22,4,9
				; lfpsux  23,4,9

	# essentially part of loop, but interleaved with
	# the stores in the tail of the loop
	fxpmul	 24,20,8
	fxsmul	 25,20,8
	fxpmul	 26,21,8
	fxsmul	 27,21,8
	fxpmul	 28,22,8
	fxsmul	 29,22,8
	fxpmul	 30,23,8
	fxsmul	 31,23,8

0:
	# time steps 0-7

	# load 0

	fxcsmadd 24,23,1,24	; rlwinm 12,28,2,14,29
	# load 1
	fxcpmadd 25,20,9,25	; lfssx  0,11,12
	fxcsmadd 26,20,9,26	; rlwinm 12,28,18,14,29
	fxcpmadd 27,21,9,27	; lfsx	 0,11,12
	fxcsmadd 28,21,9,28	; lwzux	 28,5,10
	fxcpmadd 29,22,9,29
	fxcsmadd 30,22,9,30
	fxcpmadd 31,23,9,31

	fxcpmadd 24,23,2,24	; rlwinm 12,29,2,14,29
	fxcsmadd 25,23,2,25	; lfssx  1,11,12
	# load 2
	fxcpmadd 26,20,10,26	; rlwinm 12,29,18,14,29
	fxcsmadd 27,20,10,27	; lfsx	 1,11,12
	fxcpmadd 28,21,10,28	; lwzux	 29,5,10
	fxcsmadd 29,21,10,29
	fxcpmadd 30,22,10,30
	fxcsmadd 31,22,10,31

	fxcsmadd 24,22,3,24	; rlwinm 12,30,2,14,29
	fxcpmadd 25,23,3,25	; lfssx  2,11,12
	fxcsmadd 26,23,3,26	; rlwinm 12,30,18,14,29
	# load 3
	fxcpmadd 27,20,11,27	; lfsx	 2,11,12
	fxcsmadd 28,20,11,28	; lwzux	 30,5,10
	fxcpmadd 29,21,11,29
	fxcsmadd 30,21,11,30
	fxcpmadd 31,22,11,31

	fxcpmadd 24,22,4,24	; rlwinm 12,31,2,14,29
	fxcsmadd 25,22,4,25	; lfssx  3,11,12
	fxcpmadd 26,23,4,26	; rlwinm 12,31,18,14,29
	fxcsmadd 27,23,4,27	; lfsx   3,11,12
	# load 4
	fxcpmadd 28,20,12,28	; lwzux	 31,5,10
	fxcsmadd 29,20,12,29
	fxcpmadd 30,21,12,30
	fxcsmadd 31,21,12,31

	fxcsmadd 24,21,5,24
	fxcpmadd 25,22,5,25
	fxcsmadd 26,22,5,26
	fxcpmadd 27,23,5,27
	fxcsmadd 28,23,5,28
	# load 5
	fxcpmadd 29,20,13,29
	fxcsmadd 30,20,13,30
	fxcpmadd 31,21,13,31

	fxcpmadd 24,21,6,24
	fxcsmadd 25,21,6,25
	fxcpmadd 26,22,6,26
	fxcsmadd 27,22,6,27
	fxcpmadd 28,23,6,28
	fxcsmadd 29,23,6,29
	# load 6
	fxcpmadd 30,20,14,30
	fxcsmadd 31,20,14,31

	fxcsmadd 24,20,7,24
	fxcpmadd 25,21,7,25
	fxcsmadd 26,21,7,26
	fxcpmadd 27,22,7,27	; rlwinm 12,28,2,14,29
	fxcsmadd 28,22,7,28	; lfssx	 4,11,12
	fxcpmadd 29,23,7,29
	fxcsmadd 30,23,7,30
	# load 7
	fxcpmadd 31,20,15,31

	# after loads
	fxcpmadd 24,16,0,24	; rlwinm 12,28,18,14,29
	fxcsmadd 25,16,0,25	; lfsx   4,11,12
	fxcpmadd 26,17,0,26
	fxcsmadd 27,17,0,27	; lwzux	 28,5,10
	fxcpmadd 28,18,0,28
	fxcsmadd 29,18,0,29
	fxcpmadd 30,19,0,30
	fxcsmadd 31,19,0,31	; rlwinm 12,29,2,14,29

	fxcsmadd 24,19,9,24	; lfssx  5,11,12
	fxcpmadd 25,16,1,25
	fxcsmadd 26,16,1,26
	fxcpmadd 27,17,1,27
	fxcsmadd 28,17,1,28	; rlwinm 12,29,18,14,29
	fxcpmadd 29,18,1,29	; lfsx   5,11,12
	fxcsmadd 30,18,1,30
	fxcpmadd 31,19,1,31	; lwzux	 29,5,10

	fxcpmadd 24,19,10,24
	fxcsmadd 25,19,10,25
	fxcpmadd 26,16,2,26
	fxcsmadd 27,16,2,27	; rlwinm 12,30,2,14,29
	fxcpmadd 28,17,2,28	; lfssx  6,11,12
	fxcsmadd 29,17,2,29
	fxcpmadd 30,18,2,30
	fxcsmadd 31,18,2,31

	fxcsmadd 24,18,11,24	; rlwinm 12,30,18,14,29
	fxcpmadd 25,19,11,25	; lfsx   6,11,12
	fxcsmadd 26,19,11,26
	fxcpmadd 27,16,3,27	; lwzux	 30,5,10
	fxcsmadd 28,16,3,28
	fxcpmadd 29,17,3,29
	fxcsmadd 30,17,3,30
	fxcpmadd 31,18,3,31	; rlwinm 12,31,2,14,29
	
	fxcpmadd 24,18,12,24	; lfssx  7,11,12
	fxcsmadd 25,18,12,25
	fxcpmadd 26,19,12,26
	fxcsmadd 27,19,12,27
	fxcpmadd 28,16,4,28	; rlwinm 12,31,18,14,29
	fxcsmadd 29,16,4,29	; lfsx   7,11,12
	fxcpmadd 30,17,4,30
	fxcsmadd 31,17,4,31	; lwzux	 31,5,10

	fxcsmadd 24,17,13,24
	fxcpmadd 25,18,13,25
	fxcsmadd 26,18,13,26
	fxcpmadd 27,19,13,27
	fxcsmadd 28,19,13,28
	fxcpmadd 29,16,5,29
	fxcsmadd 30,16,5,30
	fxcpmadd 31,17,5,31	; rlwinm 12,28,2,14,29

	fxcpmadd 24,17,14,24	; lfssx  8,11,12
	fxcsmadd 25,17,14,25
	fxcpmadd 26,18,14,26
	fxcsmadd 27,18,14,27
	fxcpmadd 28,19,14,28	; rlwinm 12,28,18,14,29
	fxcsmadd 29,19,14,29	; lfsx   8,11,12
	fxcpmadd 30,16,6,30
	fxcsmadd 31,16,6,31	; lwzux	 28,5,10

	fxcsmadd 24,16,15,24
	fxcpmadd 25,17,15,25
	fxcsmadd 26,17,15,26
	fxcpmadd 27,18,15,27
	fxcsmadd 28,18,15,28
	fxcpmadd 29,19,15,29	; stfpsux 24,6,9
	fxcsmadd 30,19,15,30	; stfpsux 25,6,9
	fxcpmadd 31,16,7,31	; stfpsux 26,6,9

	
	# time steps 8-15
	fxpmul	 24,20,0
	fxsmul	 25,20,0
	fxpmul	 26,21,0	; stfpsux 27,6,9
	fxsmul	 27,21,0	; stfpsux 28,6,9
	fxpmul	 28,22,0	; stfpsux 29,6,9
	fxsmul	 29,22,0	; stfpsux 30,6,9
	fxpmul	 30,23,0	; stfpsux 31,6,9
	fxsmul	 31,23,0

	# already loaded 8

	fxcsmadd 24,23,9,24
	# load 9
	fxcpmadd 25,20,1,25
	fxcsmadd 26,20,1,26
	fxcpmadd 27,21,1,27
	fxcsmadd 28,21,1,28
	fxcpmadd 29,22,1,29
	fxcsmadd 30,22,1,30
	fxcpmadd 31,23,1,31

	fxcpmadd 24,23,10,24
	fxcsmadd 25,23,10,25
	# load 10
	fxcpmadd 26,20,2,26
	fxcsmadd 27,20,2,27
	fxcpmadd 28,21,2,28
	fxcsmadd 29,21,2,29
	fxcpmadd 30,22,2,30
	fxcsmadd 31,22,2,31	; rlwinm 12,29,2,14,29

	fxcsmadd 24,22,11,24	; lfssx  9,11,12
	fxcpmadd 25,23,11,25
	fxcsmadd 26,23,11,26
	# load 11
	fxcpmadd 27,20,3,27
	fxcsmadd 28,20,3,28	; rlwinm 12,29,18,14,29
	fxcpmadd 29,21,3,29	; lfsx   9,11,12
	fxcsmadd 30,21,3,30
	fxcpmadd 31,22,3,31	; lwzux	 29,5,10

	fxcpmadd 24,22,12,24
	fxcsmadd 25,22,12,25
	fxcpmadd 26,23,12,26
	fxcsmadd 27,23,12,27	; rlwinm 12,30,2,14,29
	# load 12
	fxcpmadd 28,20,4,28	; lfssx  10,11,12
	fxcsmadd 29,20,4,29
	fxcpmadd 30,21,4,30
	fxcsmadd 31,21,4,31

	fxcsmadd 24,21,13,24	; rlwinm 12,30,18,14,29
	fxcpmadd 25,22,13,25	; lfsx   10,11,12
	fxcsmadd 26,22,13,26
	fxcpmadd 27,23,13,27	; lwzux	 30,5,10
	fxcsmadd 28,23,13,28
	# load 13
	fxcpmadd 29,20,5,29
	fxcsmadd 30,20,5,30
	fxcpmadd 31,21,5,31	; rlwinm 12,31,2,14,29

	fxcpmadd 24,21,14,24	; lfssx  11,11,12
	fxcsmadd 25,21,14,25
	fxcpmadd 26,22,14,26
	fxcsmadd 27,22,14,27
	fxcpmadd 28,23,14,28	; rlwinm 12,31,18,14,29
	fxcsmadd 29,23,14,29	; lfsx   11,11,12
	# load 14
	fxcpmadd 30,20,6,30
	fxcsmadd 31,20,6,31	; lwzux	 31,5,10

	fxcsmadd 24,20,15,24
	fxcpmadd 25,21,15,25
	fxcsmadd 26,21,15,26
	fxcpmadd 27,22,15,27	; rlwinm 12,28,2,14,29
	fxcsmadd 28,22,15,28	; lfssx  12,11,12
	fxcpmadd 29,23,15,29
	fxcsmadd 30,23,15,30
	# load 15
	fxcpmadd 31,20,7,31

	# after loads

	fxcpmadd 24,16,8,24	; rlwinm 12,28,18,14,29
	fxcsmadd 25,16,8,25	; lfsx   12,11,12
	fxcpmadd 26,17,8,26
	fxcsmadd 27,17,8,27	; lwzux	 28,5,10
	fxcpmadd 28,18,8,28
	fxcsmadd 29,18,8,29
	fxcpmadd 30,19,8,30
	fxcsmadd 31,19,8,31	; rlwinm 12,29,2,14,29

	fxcsmadd 24,19,1,24	; lfssx  13,11,12
	fxcpmadd 25,16,9,25
	fxcsmadd 26,16,9,26
	fxcpmadd 27,17,9,27
	fxcsmadd 28,17,9,28	; rlwinm 12,29,18,14,29
	fxcpmadd 29,18,9,29	; lfsx   13,11,12
	fxcsmadd 30,18,9,30
	fxcpmadd 31,19,9,31	; lwzux	 29,5,10

	fxcpmadd 24,19,2,24
	fxcsmadd 25,19,2,25
	fxcpmadd 26,16,10,26
	fxcsmadd 27,16,10,27	; rlwinm 12,30,2,14,29
	fxcpmadd 28,17,10,28	; lfssx  14,11,12
	fxcsmadd 29,17,10,29
	fxcpmadd 30,18,10,30
	fxcsmadd 31,18,10,31

	fxcsmadd 24,18,3,24	; rlwinm 12,30,18,14,29
	fxcpmadd 25,19,3,25	; lfsx   14,11,12
	fxcsmadd 26,19,3,26
	fxcpmadd 27,16,11,27	; lwzux	 30,5,10
	fxcsmadd 28,16,11,28
	fxcpmadd 29,17,11,29
	fxcsmadd 30,17,11,30
	fxcpmadd 31,18,11,31	; rlwinm 12,31,2,14,29

	fxcpmadd 24,18,4,24	; lfssx  15,11,12
	fxcsmadd 25,18,4,25
	fxcpmadd 26,19,4,26
	fxcsmadd 27,19,4,27
	fxcpmadd 28,16,12,28	; rlwinm 12,31,18,14,29
	fxcsmadd 29,16,12,29	; lfsx   15,11,12
	fxcpmadd 30,17,12,30
	fxcsmadd 31,17,12,31	; lwzux	 31,5,10
	
	fxcsmadd 24,17,5,24
	fxcpmadd 25,18,5,25
	fxcsmadd 26,18,5,26
	fxcpmadd 27,19,5,27
	fxcsmadd 28,19,5,28
	fxcpmadd 29,16,13,29
	fxcsmadd 30,16,13,30
	fxcpmadd 31,17,13,31

	fxcpmadd 24,17,6,24
	fxcsmadd 25,17,6,25
	fxcpmadd 26,18,6,26
	fxcsmadd 27,18,6,27
	fxcpmadd 28,19,6,28
	fxcsmadd 29,19,6,29
	fxcpmadd 30,16,14,30
	fxcsmadd 31,16,14,31

	fxcsmadd 24,16,7,24
	fxcpmadd 25,17,7,25
	fxcsmadd 26,17,7,26
	fxcpmadd 27,18,7,27
	fxcsmadd 28,18,7,28
	fxcpmadd 29,19,7,29	; stfpsux 24,6,9
	fxcsmadd 30,19,7,30	; stfpsux 25,6,9
	fxcpmadd 31,16,15,31	; stfpsux 26,6,9

	fxpmul	 24,20,8	# part of next loop
	fxsmul	 25,20,8
	fxpmul	 26,21,8	; stfpsux 27,6,9
	fxsmul	 27,21,8	; stfpsux 28,6,9
	fxpmul	 28,22,8	; stfpsux 29,6,9
	fxsmul	 29,22,8	; stfpsux 30,6,9
	fxpmul	 30,23,8	; stfpsux 31,6,9
	fxsmul	 31,23,8

	bdnz	 0b

	li	9,DCOMPLEX_SIZE	# restore call-saved registers
	lmw	28,0(1)	# restore r28 ... r31

	#addi	1,1,16
	lfpdux	31,1,9
	lfpdux	30,1,9
	lfpdux	29,1,9
	lfpdux	28,1,9
	lfpdux	27,1,9
	lfpdux	26,1,9
	lfpdux	25,1,9
	lfpdux	24,1,9
	lfpdux	23,1,9
	lfpdux	22,1,9
	lfpdux	21,1,9
	lfpdux	20,1,9
	lfpdux	19,1,9
	lfpdux	18,1,9
	lfpdux	17,1,9
	lfpdux	16,1,9
	lfpdux	15,1,9
	lfpdux	14,1,9

	addi	1,1,16		# restore stack pointer
	blr			# return

#else

#	filter for i16complex

.align	5
sub_value:
	.long	 0x43300000,0x00008000,0x43300000,0x00008000


.global	_ZN5LOFAR4RTCP7_filterISt7complexIsEEEvjPKfPKT_PS2_IfEi
_ZN5LOFAR4RTCP7_filterISt7complexIsEEEvjPKfPKT_PS2_IfEi:
#	filters all samples for one station, one polarization

#	arguments:
#	r3 :	  nrChannels
#	r4 :	  pointer to weights line (const fcomplex[16])
#	r5 :	  pointer to first sample (const i16complex[16*r7])
#	r6 :	  pointer to result (fcomplex *)
#	r7 :	  number of samples / 16

#	internally used:
#	r9 :	  8
#	r10 :	  2048
#	r11 :	  8
#	r12 :	  0x00800080
#	r28-r31 : prefetched samples
#	f0-f15 :  delay line (real in primary, imaginary in secondary unit)
#	f16-f23 : weights (these are real values alternately stored in primary
#			   and secondary units)
#	f24-f29 : sums
#	f31 :	  sub_value

#	The implementation works on 5 or 6 time samples concurrently, to avoid
#	stalls in the double hummer.  This unfortunately leads to totally
#	incomprehensible code.  The loop processes 16 samples at a time.
#	The input is converted from int16complex to dcomplex by black magic,
#	making the code even harder to understand.

	mtctr	7		# set number of iterations

	li	9,-DCOMPLEX_SIZE
	stfpdux 14,1,9		# save call-saved registers
	stfpdux 15,1,9
	stfpdux 16,1,9
	stfpdux 17,1,9
	stfpdux 18,1,9
	stfpdux 19,1,9
	stfpdux 20,1,9
	stfpdux 21,1,9
	stfpdux 22,1,9
	stfpdux 23,1,9
	stfpdux 24,1,9
	stfpdux 25,1,9
	stfpdux 26,1,9
	stfpdux 27,1,9
	stfpdux 28,1,9
	stfpdux 29,1,9
	stfpdux 30,1,9
	stfpdux 31,1,9

	subi	1,1,48
	stmw	28,16(1)	# save r28 ... r31

	lis	8,sub_value@ha	# load sub_values
	lis	12,0x0080
	la	8,sub_value@l(8)
	lfpdx	31,0,8


	# Avoid that int->fp conversion area is in the same cache way as where
	# the samples come from.  There are two places that can be used where
	# the conversion can take place: at 0(sp) or 32(sp).  r8, r7, and r11
	# contain offsets (w.r.t. sp) to this area.  Note that 16(sp) stores
	# r28 ... r31

	xor	8,1,5	
	andi.	0,8,0x01E0
	li	9,FCOMPLEX_SIZE
	mfcr	8
	slwi	10,3,3	# I16COMPLEX_SIZE*nrChannels*NR_POLARIZATIONS
	rlwinm	8,8,8,26,26
	ori	12,12,0x0080
	sub	6,6,9

	addi	7,8,14
	addi	11,8,6
	stfpdx	31,8,1		# initialize int->fp conversion area

#if 0
	lfpsx	1,0,3		# load delay line
	lfpsux	2,3,9
	lfpsux	3,3,9
	lfpsux	4,3,9
	lfpsux	5,3,9
	lfpsux	6,3,9
	lfpsux	7,3,9
	lfpsux	8,3,9
	lfpsux	9,3,9
	lfpsux	10,3,9
	lfpsux	11,3,9
	lfpsux	12,3,9
	lfpsux	13,3,9
	lfpsux	14,3,9
	lfpsux	15,3,9
#endif

	# convert 15 i16complex numbers to fcomplex
	lwzx	29,0,5		# fetch FIR history samples
	lwzux	30,5,10
	lwzux	31,5,10
	lwzux	28,5,10

	xor	29,29,12
	sthbrx	29,7,1
	srawi	29,29,16
	sthbrx	29,11,1
	lfpdx	1,8,1

	lwzux	29,5,10

	xor	30,30,12
	sthbrx	30,7,1
	srawi	30,30,16
	sthbrx	30,11,1
	lfpdx	2,8,1

	lwzux	30,5,10

	xor	31,31,12
	sthbrx	31,7,1
	srawi	31,31,16
	sthbrx	31,11,1
	lfpdx	3,8,1

	lwzux	31,5,10

	xor	28,28,12
	sthbrx	28,7,1
	srawi	28,28,16
	sthbrx	28,11,1
	lfpdx	4,8,1

	lwzux	28,5,10

	xor	29,29,12
	sthbrx	29,7,1
	srawi	29,29,16
	sthbrx	29,11,1
	lfpdx	5,8,1

	lwzux	29,5,10

	xor	30,30,12
	sthbrx	30,7,1
	srawi	30,30,16
	sthbrx	30,11,1
	lfpdx	6,8,1

	lwzux	30,5,10

	xor	31,31,12
	sthbrx	31,7,1
	srawi	31,31,16
	sthbrx	31,11,1
	lfpdx	7,8,1

	lwzux	31,5,10

	xor	28,28,12
	sthbrx	28,7,1
	srawi	28,28,16
	sthbrx	28,11,1
	lfpdx	8,8,1

	lwzux	28,5,10

	xor	29,29,12
	sthbrx	29,7,1
	srawi	29,29,16
	sthbrx	29,11,1
	lfpdx	9,8,1

	lwzux	29,5,10

	xor	30,30,12
	sthbrx	30,7,1
	srawi	30,30,16
	sthbrx	30,11,1
	lfpdx	10,8,1

	lwzux	30,5,10

	xor	31,31,12
	sthbrx	31,7,1
	srawi	31,31,16
	sthbrx	31,11,1
	lfpdx	11,8,1

	lwzux	31,5,10

	xor	28,28,12
	sthbrx	28,7,1
	srawi	28,28,16
	sthbrx	28,11,1
	lfpdx	12,8,1

	xor	29,29,12
	sthbrx	29,7,1
	srawi	29,29,16
	sthbrx	29,11,1
	lfpdx	13,8,1

	xor	30,30,12
	sthbrx	30,7,1
	srawi	30,30,16
	sthbrx	30,11,1
	lfpdx	14,8,1

	xor	31,31,12
	sthbrx	31,7,1
	srawi	31,31,16
	sthbrx	31,11,1
	lfpdx	15,8,1

	fpsub	1,1,31		; lfpsx	  16,0,4	# load weights line
	fpsub	2,2,31		; lfpsux  17,4,9
	fpsub	3,3,31		; lfpsux  18,4,9
	fpsub	4,4,31		; lfpsux  19,4,9
	fpsub	5,5,31		; lfpsux  20,4,9
	fpsub	6,6,31		; lfpsux  21,4,9
	fpsub	7,7,31		; lfpsux  22,4,9
	fpsub	8,8,31		; lfpsux  23,4,9
	fpsub	9,9,31		; lwzux	28,5,10 # prefetch samples before entering loop
	fpsub	10,10,31	; lwzux	29,5,10
	fpsub	11,11,31	; lwzux	30,5,10
	fpsub	12,12,31	; lwzux	31,5,10
	fpsub	13,13,31
	fpsub	14,14,31
	fpsub	15,15,31

	# essentially part of loop, but interleaved with
	# the stores in the tail of the loop
	fxsmul	 24,16,15
	fxpmul	 25,17,15
	fxsmul	 26,17,15
	fxpmul	 27,18,15
	fxsmul	 28,18,15
	fxpmul	 29,19,15

0:
	# time steps 0-5

	fxcsmadd 24,23,1,24	; xor	 28,28,12
	fxcpmadd 25,20,9,25	; sthbrx 28,7,1
	fxcsmadd 26,20,9,26	; srawi	 28,28,16
	fxcpmadd 27,21,9,27	; sthbrx 28,11,1
	fxcsmadd 28,21,9,28
	fxcpmadd 29,22,9,29

	fxcpmadd 24,23,2,24
	fxcsmadd 25,23,2,25	; lfpdx	 30,8,1
	fxcpmadd 26,20,10,26
	fxcsmadd 27,20,10,27
	fxcpmadd 28,21,10,28
	fxcsmadd 29,21,10,29	; lwzux	 28,5,10

	fxcsmadd 24,22,3,24
	fxcpmadd 25,23,3,25
	fxcsmadd 26,23,3,26
	fxcpmadd 27,20,11,27	; xor	 29,29,12
	fxcsmadd 28,20,11,28	; sthbrx 29,7,1
	fxcpmadd 29,21,11,29	; srawi	 29,29,16

	fxcpmadd 24,22,4,24	; sthbrx 29,11,1
	fxcsmadd 25,22,4,25
	fxcpmadd 26,23,4,26
	fxcsmadd 27,23,4,27
	fpsub	0,30,31	
	fxcpmadd 28,20,12,28	; lfpdx	 30,8,1
	fxcsmadd 29,20,12,29

	fxcsmadd 24,21,5,24
	fxcpmadd 25,22,5,25
	fxcsmadd 26,22,5,26	; lwzux	 29,5,10
	fxcpmadd 27,23,5,27
	fxcsmadd 28,23,5,28
	fxcpmadd 29,20,13,29

	fxcpmadd 24,21,6,24	; xor	 30,30,12
	fxcsmadd 25,21,6,25	; sthbrx 30,7,1
	fxcpmadd 26,22,6,26	; srawi	 30,30,16
	fxcsmadd 27,22,6,27	; sthbrx 30,11,1
	fxcpmadd 28,23,6,28
	fxcsmadd 29,23,6,29

	fxcsmadd 24,20,7,24
	fxcpmadd 25,21,7,25
	fpsub	1,30,31	
	fxcsmadd 26,21,7,26	; lfpdx	 30,8,1
	fxcpmadd 27,22,7,27
	fxcsmadd 28,22,7,28
	fxcpmadd 29,23,7,29	; lwzux	 30,5,10

	fxcpmadd 24,20,8,24
	fxcsmadd 25,20,8,25
	fxcpmadd 26,21,8,26
	fxcsmadd 27,21,8,27
	fxcpmadd 28,22,8,28
	fxcsmadd 29,22,8,29	; xor	 31,31,12

	fxcpmadd 24,17,14,24	; sthbrx 31,7,1
	fxcsmadd 25,17,14,25
	fxcpmadd 26,18,14,26
	fxcsmadd 27,18,14,27	; srawi	 31,31,16
	fxcpmadd 28,19,14,28	; sthbrx 31,11,1
	fxcsmadd 29,19,14,29

	fxcpmadd 24,16,0,24
	fxcsmadd 25,16,0,25
	fpsub	2,30,31	
	fxcpmadd 26,17,0,26	; lfpdx	 30,8,1
	fxcsmadd 27,17,0,27
	fxcpmadd 28,18,0,28
	fxcsmadd 29,18,0,29	; lwzux	 31,5,10

	fpsub	3,30,31		; xor	 28,28,12
	fxcsmadd 24,19,9,24	; sthbrx 28,7,1
	fxcpmadd 25,16,1,25
	fxcsmadd 26,16,1,26
	fxcpmadd 27,17,1,27	; srawi	 28,28,16
	fxcsmadd 28,17,1,28	; sthbrx 28,11,1
	fxcpmadd 29,18,1,29

	fxcpmadd 24,19,10,24
	fxcsmadd 25,19,10,25
	fxcpmadd 26,16,2,26	; lfpdx	 30,8,1
	fxcsmadd 27,16,2,27
	fxcpmadd 28,17,2,28
	fxcsmadd 29,17,2,29	; lwzux	 28,5,10

	fpsub	4,30,31		; xor	 29,29,12
	fxcsmadd 24,18,11,24	; sthbrx 29,7,1
	fxcpmadd 25,19,11,25
	fxcsmadd 26,19,11,26
	fxcpmadd 27,16,3,27	; srawi	 29,29,16
	fxcsmadd 28,16,3,28	; sthbrx 29,11,1
	fxcpmadd 29,17,3,29

	fxcpmadd 24,18,12,24
	fxcsmadd 25,18,12,25
	fxcpmadd 26,19,12,26	; lfpdx	 30,8,1
	fxcsmadd 27,19,12,27
	fxcpmadd 28,16,4,28
	fxcsmadd 29,16,4,29	; lwzux	 29,5,10

	fpsub	5,30,31
	fxcsmadd 24,17,13,24
	fxcpmadd 25,18,13,25
	fxcsmadd 26,18,13,26
	fxcpmadd 27,19,13,27
	fxcsmadd 28,19,13,28
	fxcpmadd 29,16,5,29
		; stfpsux 24,6,9
		; stfpsux 25,6,9


	# time steps 6-10

	fxsmul	 24,21,11	; stfpsux 26,6,9
	fxpmul	 25,22,11	; stfpsux 27,6,9
	fxsmul	 26,22,11	; stfpsux 28,6,9
	fxpmul	 27,23,11	; stfpsux 29,6,9
	fxsmul	 28,23,11

	fxcpmadd 24,17,4,24
	fxcsmadd 25,17,4,25
	fxcpmadd 26,18,4,26
	fxcsmadd 27,18,4,27	; xor	 30,30,12
	fxcpmadd 28,19,4,28	; sthbrx 30,7,1
	
	fxcsmadd 24,20,13,24
	fxcpmadd 25,21,13,25
	fxcsmadd 26,21,13,26	; srawi	 30,30,16
	fxcpmadd 27,22,13,27	; sthbrx 30,11,1
	fxcsmadd 28,22,13,28

	fxcsmadd 24,16,5,24
	fxcpmadd 25,17,5,25	; lfpdx	 30,8,1
	fxcsmadd 26,17,5,26
	fxcpmadd 27,18,5,27	; lwzux	 30,5,10
	fxcsmadd 28,18,5,28
	
	fxcpmadd 24,20,14,24	; xor	 31,31,12
	fpsub	6,30,31		; sthbrx 31,7,1
	fxcsmadd 25,20,14,25
	fxcpmadd 26,21,14,26
	fxcsmadd 27,21,14,27	; srawi	 31,31,16
	fxcpmadd 28,22,14,28	; sthbrx 31,11,1

	fxcpmadd 24,16,6,24
	fxcsmadd 25,16,6,25
	fxcpmadd 26,17,6,26	; lfpdx	 30,8,1
	fxcsmadd 27,17,6,27
	fxcpmadd 28,18,6,28	; lwzux	 31,5,10

	fxcsmadd 24,23,7,24
	fxcpmadd 25,20,15,25	; xor	 28,28,12
	fpsub	7,30,31		; sthbrx 28,7,1
	fxcsmadd 26,20,15,26
	fxcpmadd 27,21,15,27
	fxcsmadd 28,21,15,28	; srawi	 28,28,16

	fxcsmadd 24,19,15,24	; sthbrx 28,11,1
	fxcpmadd 25,16,7,25
	fxcsmadd 26,16,7,26
	fxcpmadd 27,17,7,27	; lfpdx	 30,8,1
	fxcsmadd 28,17,7,28

	fxcpmadd 24,23,8,24	; lwzux	 28,5,10
	fxcsmadd 25,23,8,25
	fxcpmadd 26,20,0,26	; xor	 29,29,12
	fpsub	8,30,31		; sthbrx 29,7,1
	fxcsmadd 27,20,0,27
	fxcpmadd 28,21,0,28

	fxcpmadd 24,19,0,24	; srawi	 29,29,16
	fxcsmadd 25,19,0,25	; sthbrx 29,11,1
	fxcpmadd 26,16,8,26
	fxcsmadd 27,16,8,27
	fxcpmadd 28,17,8,28	; lfpdx	 30,8,1

	fxcsmadd 24,17,3,24
	fxcpmadd 25,18,3,25
	fxcsmadd 26,18,3,26
	fxcpmadd 27,19,3,27
	fxcsmadd 28,19,3,28

	fxcsmadd 24,22,9,24
	fxcpmadd 25,23,9,25	; lwzux	 29,5,10
	fxcsmadd 26,23,9,26
	fxcpmadd 27,20,1,27	; xor	 30,30,12
	fpsub	9,30,31		; sthbrx 30,7,1
	fxcsmadd 28,20,1,28

	fxcsmadd 24,18,1,24
	fxcpmadd 25,19,1,25	; srawi	 30,30,16
	fxcsmadd 26,19,1,26	; sthbrx 30,11,1
	fxcpmadd 27,16,9,27
	fxcsmadd 28,16,9,28

	fxcpmadd 24,21,12,24
	fxcsmadd 25,21,12,25
	fxcpmadd 26,22,12,26
	fxcsmadd 27,22,12,27
	fxcpmadd 28,23,12,28

	fxcpmadd 24,22,10,24	; lfpdx	 30,8,1
	fxcsmadd 25,22,10,25
	fxcpmadd 26,23,10,26	; lwzux	 30,5,10
	fxcsmadd 27,23,10,27
	fxcpmadd 28,20,2,28
	fpsub	10,30,31

	fxcpmadd 24,18,2,24
	fxcsmadd 25,18,2,25
	fxcpmadd 26,19,2,26
	fxcsmadd 27,19,2,27
	fxcpmadd 28,16,10,28


	# time steps 11-15
				; stfpsux 24,6,9
	fxsmul	 24,21,0	; stfpsux 25,6,9
	fxpmul	 25,22,0	; stfpsux 26,6,9
	fxsmul	 26,22,0	; stfpsux 27,6,9
	fxpmul	 27,23,0	; stfpsux 28,6,9
	fxsmul	 28,23,0

	fxcpmadd 24,17,9,24
	fxcsmadd 25,17,9,25
	fxcpmadd 26,18,9,26
	fxcsmadd 27,18,9,27	; xor	 31,31,12
	fxcpmadd 28,19,9,28	; sthbrx 31,7,1
	
	fxcsmadd 24,20,2,24
	fxcpmadd 25,21,2,25
	fxcsmadd 26,21,2,26	; srawi	 31,31,16
	fxcpmadd 27,22,2,27	; sthbrx 31,11,1
	fxcsmadd 28,22,2,28

	fxcsmadd 24,16,10,24
	fxcpmadd 25,17,10,25	; lfpdx	 30,8,1
	fxcsmadd 26,17,10,26
	fxcpmadd 27,18,10,27	; lwzux	 31,5,10
	fxcsmadd 28,18,10,28

	fxcpmadd 24,20,3,24	; xor	 28,28,12
	fpsub	11,30,31	; sthbrx 28,7,1
	fxcsmadd 25,20,3,25
	fxcpmadd 26,21,3,26
	fxcsmadd 27,21,3,27	; srawi	 28,28,16
	fxcpmadd 28,22,3,28	; sthbrx 28,11,1
	
	fxcpmadd 24,16,11,24
	fxcsmadd 25,16,11,25
	fxcpmadd 26,17,11,26	; lfpdx	 30,8,1
	fxcsmadd 27,17,11,27
	fxcpmadd 28,18,11,28	; lwzux	 28,5,10

	fxcsmadd 24,23,12,24
	fxcpmadd 25,20,4,25	; xor	 29,29,12
	fpsub	12,30,31	; sthbrx 29,7,1
	fxcsmadd 26,20,4,26
	fxcpmadd 27,21,4,27
	fxcsmadd 28,21,4,28	; srawi	 29,29,16

	fxcsmadd 24,19,4,24	; sthbrx 29,11,1
	fxcpmadd 25,16,12,25
	fxcsmadd 26,16,12,26
	fxcpmadd 27,17,12,27	; lfpdx	 30,8,1
	fxcsmadd 28,17,12,28

	fxcpmadd 24,23,13,24	; lwzux	 29,5,10
	fxcsmadd 25,23,13,25
	fxcpmadd 26,20,5,26	; xor	 30,30,12
	fpsub	13,30,31	; sthbrx 30,7,1
	fxcsmadd 27,20,5,27
	fxcpmadd 28,21,5,28
	
	fxcsmadd 24,17,8,24
	fxcpmadd 25,18,8,25
	fxcsmadd 26,18,8,26
	fxcpmadd 27,19,8,27
	fxcsmadd 28,19,8,28

	fxcpmadd 24,19,5,24	; srawi	 30,30,16
	fxcsmadd 25,19,5,25	; sthbrx 30,11,1
	fxcpmadd 26,16,13,26
	fxcsmadd 27,16,13,27
	fxcpmadd 28,17,13,28	; lfpdx	 30,8,1

	fxcsmadd 24,22,14,24
	fxcpmadd 25,23,14,25	; lwzux	 30,5,10
	fxcsmadd 26,23,14,26
	fxcpmadd 27,20,6,27	; xor	 31,31,12
	fpsub	14,30,31	; sthbrx 31,7,1
	fxcsmadd 28,20,6,28
	
	fxcsmadd 24,18,6,24
	fxcpmadd 25,19,6,25	; srawi	 31,31,16
	fxcsmadd 26,19,6,26	; sthbrx 31,11,1
	fxcpmadd 27,16,14,27
	fxcsmadd 28,16,14,28

	fxcpmadd 24,21,1,24
	fxcsmadd 25,21,1,25
	fxcpmadd 26,22,1,26
	fxcsmadd 27,22,1,27
	fxcpmadd 28,23,1,28

	fxcpmadd 24,22,15,24	; lfpdx	 30,8,1
	fxcsmadd 25,22,15,25
	fxcpmadd 26,23,15,26	; lwzux	 31,5,10
	fxcsmadd 27,23,15,27
	fxcpmadd 28,20,7,28
	fpsub	15,30,31

	fxcpmadd 24,18,7,24
	fxcsmadd 25,18,7,25
	fxcpmadd 26,19,7,26
	fxcsmadd 27,19,7,27
	fxcpmadd 28,16,15,28

	fxpmul	 29,19,15	; stfpsux 24,6,9
	fxsmul	 24,16,15	; stfpsux 25,6,9
	fxpmul	 25,17,15	; stfpsux 26,6,9
	fxsmul	 26,17,15	; stfpsux 27,6,9
	fxpmul	 27,18,15	; stfpsux 28,6,9
	fxsmul	 28,18,15

	bdnz	 0b

#if 0
	addi	3,3,-120	# store delay line
	stfpsux	1,3,9
	stfpsux	2,3,9
	stfpsux	3,3,9
	stfpsux	4,3,9
	stfpsux	5,3,9
	stfpsux	6,3,9
	stfpsux	7,3,9
	stfpsux	8,3,9
	stfpsux	9,3,9
	stfpsux	10,3,9
	stfpsux	11,3,9
	stfpsux	12,3,9
	stfpsux	13,3,9
	stfpsux	14,3,9
	stfpsux	15,3,9
#endif

	li	9,DCOMPLEX_SIZE	# restore call-saved registers
	lmw	28,16(1)	# restore r28 ... r31

	addi	1,1,48
	lfpdx	31,0,1
	lfpdux	30,1,9
	lfpdux	29,1,9
	lfpdux	28,1,9
	lfpdux	27,1,9
	lfpdux	26,1,9
	lfpdux	25,1,9
	lfpdux	24,1,9
	lfpdux	23,1,9
	lfpdux	22,1,9
	lfpdux	21,1,9
	lfpdux	20,1,9
	lfpdux	19,1,9
	lfpdux	18,1,9
	lfpdux	17,1,9
	lfpdux	16,1,9
	lfpdux	15,1,9
	lfpdux	14,1,9

	addi	1,1,16		# restore stack pointer
	blr			# return
#endif


.align 5

#if defined HAVE_BGP
zero:	.long	0,0
#endif

.global _memzero
_memzero:
#if defined HAVE_BGP
	lis	5,zero@ha	; srwi	4,4,7
	addi	5,5,zero@l	; mtctr	4
	lfpsx	0,0,5		; li	8,16
	subi	3,3,16

0:	stfpdux	0,3,8
	stfpdux	0,3,8
	stfpdux	0,3,8
	stfpdux	0,3,8
	stfpdux	0,3,8
	stfpdux	0,3,8
	stfpdux	0,3,8
	stfpdux	0,3,8
	bdnz	0b
#else
	srwi	4,4,7
	mtctr	4
	li	4,32
	li	5,64
	li	6,96
	li	7,128

0:	dcbz	0,3
	dcbz	3,4
	dcbz	3,5
	dcbz	3,6
	add	3,3,7
	bdnz	0b
#endif
	blr


.align	5
.global	_fast_memcpy
_fast_memcpy:

	li	8,-16
	stfpdux 14,1,8
	stfpdux 15,1,8
	stfpdux 16,1,8
	stfpdux 17,1,8
	stfpdux 18,1,8
	stfpdux 19,1,8
	stfpdux 20,1,8
	stfpdux 21,1,8
	stfpdux 22,1,8
	stfpdux 23,1,8
	stfpdux 24,1,8
	stfpdux 25,1,8
	stfpdux 26,1,8
	stfpdux 27,1,8
	stfpdux 28,1,8
	stfpdux 29,1,8
	stfpdux 30,1,8
	stfpdux 31,1,8

	srwi	5,5,9
	mtctr	5
	li	8,16

	sub	3,3,8
	sub	4,4,8

0:
	lfpdux	0,4,8
	lfpdux	1,4,8
	lfpdux	2,4,8
	lfpdux	3,4,8
	lfpdux	4,4,8
	lfpdux	5,4,8
	lfpdux	6,4,8
	lfpdux	7,4,8
	lfpdux	8,4,8
	lfpdux	9,4,8
	lfpdux	10,4,8
	lfpdux	11,4,8
	lfpdux	12,4,8
	lfpdux	13,4,8
	lfpdux	14,4,8
	lfpdux	15,4,8
	lfpdux	16,4,8
	lfpdux	17,4,8
	lfpdux	18,4,8
	lfpdux	19,4,8
	lfpdux	20,4,8
	lfpdux	21,4,8
	lfpdux	22,4,8
	lfpdux	23,4,8
	lfpdux	24,4,8
	lfpdux	25,4,8
	lfpdux	26,4,8
	lfpdux	27,4,8
	lfpdux	28,4,8
	lfpdux	29,4,8
	lfpdux	30,4,8
	lfpdux	31,4,8
	stfpdux	0,3,8
	stfpdux	1,3,8
	stfpdux	2,3,8
	stfpdux	3,3,8
	stfpdux	4,3,8
	stfpdux	5,3,8
	stfpdux	6,3,8
	stfpdux	7,3,8
	stfpdux	8,3,8
	stfpdux	9,3,8
	stfpdux	10,3,8
	stfpdux	11,3,8
	stfpdux	12,3,8
	stfpdux	13,3,8
	stfpdux	14,3,8
	stfpdux	15,3,8
	stfpdux	16,3,8
	stfpdux	17,3,8
	stfpdux	18,3,8
	stfpdux	19,3,8
	stfpdux	20,3,8
	stfpdux	21,3,8
	stfpdux	22,3,8
	stfpdux	23,3,8
	stfpdux	24,3,8
	stfpdux	25,3,8
	stfpdux	26,3,8
	stfpdux	27,3,8
	stfpdux	28,3,8
	stfpdux	29,3,8
	stfpdux	30,3,8
	stfpdux	31,3,8
	bdnz	0b

	lfpdx	31,0,1
	lfpdux	30,1,8
	lfpdux	29,1,8
	lfpdux	28,1,8
	lfpdux	27,1,8
	lfpdux	26,1,8
	lfpdux	25,1,8
	lfpdux	24,1,8
	lfpdux	23,1,8
	lfpdux	22,1,8
	lfpdux	21,1,8
	lfpdux	20,1,8
	lfpdux	19,1,8
	lfpdux	18,1,8
	lfpdux	17,1,8
	lfpdux	16,1,8
	lfpdux	15,1,8
	lfpdux	14,1,8
	addi	1,1,16
	blr


	# little endian i16complex -> float

.global	_ZN5LOFAR4RTCP8_convertISt7complexIsEEEvPS2_IfEPKT_j
_ZN5LOFAR4RTCP8_convertISt7complexIsEEEvPS2_IfEPKT_j:

	stwu	1,-32(1)

	lis	8,sub_value@ha	# load sub_values
	li	9,sub_value@l
	lfpdx	0,8,9

	lis	12,0x0080
	ori	12,12,0x0080

	srwi	5,5,1
	subi	5,5,2
	mtctr	5

	subi	3,3,8

	li	8,8
	li	9,4
	li	10,14
	li	11,6
	addi	5,1,16

	stfpdx	0,0,1		# initialize int->fp conversion area
	stfpdx	0,0,5

	lwz	6,0(4)

	lwzux	7,4,9
	xor	6,6,12
	sthbrx	6,10,1
	srwi	6,6,16
	sthbrx	6,11,1

	lwzux	6,4,9
	xor	7,7,12
	sthbrx	7,10,5
	srwi	7,7,16
	sthbrx	7,11,5
	lfpdx	1,0,1

	lwzux	7,4,9
	xor	6,6,12
	sthbrx	6,10,1
	srwi	6,6,16
	sthbrx	6,11,1
	fpsub	1,1,0
	lfpdx	2,0,5

0:
	lwzux	6,4,9
	xor	7,7,12
	sthbrx	7,10,5
	srwi	7,7,16
	sthbrx	7,11,5
	fpsub	2,2,0
	stfpsux	1,3,8
	lfpdx	1,0,1

	lwzux	7,4,9
	xor	6,6,12
	sthbrx	6,10,1
	srwi	6,6,16
	sthbrx	6,11,1
	fpsub	1,1,0
	stfpsux	2,3,8
	lfpdx	2,0,5

	bdnz	0b

	xor	7,7,12
	sthbrx	7,10,5
	srwi	7,7,16
	sthbrx	7,11,5
	fpsub	2,2,0
	stfpsux	1,3,8
	lfpdx	1,0,1

	fpsub	1,1,0
	stfpsux	2,3,8
	lfpdx	2,0,5

	fpsub	2,2,0
	stfpsux	1,3,8

	stfpsux	2,3,8

	addi	1,1,32
	blr


	# i8complex -> float

.global	_ZN5LOFAR4RTCP8_convertISt7complexIaEEEvPS2_IfEPKT_j
_ZN5LOFAR4RTCP8_convertISt7complexIaEEEvPS2_IfEPKT_j:

	lis	11,_ZN5LOFAR4RTCP13_FIR_fp_tableE@ha
	la	11,_ZN5LOFAR4RTCP13_FIR_fp_tableE@l(11)

	srwi	5,5,2
	subi	5,5,2
	mtctr	5

	subi	3,3,8

	li	10,8
	li	9,2

	lhz	5,0(4)
	lhzux	6,4,9
	lhzux	7,4,9
	lhzux	8,4,9

	slwi	5,5,3
	lfpsx	0,11,5
	lhzux	5,4,9

	slwi	6,6,3
	lfpsx	1,11,6
	lhzux	6,4,9

	slwi	7,7,3
	lfpsx	2,11,7
	lhzux	7,4,9

	slwi	8,8,3
	lfpsx	3,11,8
	lhzux	8,4,9

0:
	slwi	5,5,3
	stfpsux	0,3,10
	lfpsx	0,11,5
	lhzux	5,4,9

	slwi	6,6,3
	stfpsux	1,3,10
	lfpsx	1,11,6
	lhzux	6,4,9

	slwi	7,7,3
	stfpsux	2,3,10
	lfpsx	2,11,7
	lhzux	7,4,9

	slwi	8,8,3
	stfpsux	3,3,10
	lfpsx	3,11,8
	lhzux	8,4,9

	bdnz	0b

	slwi	5,5,3
	stfpsux	0,3,10
	lfpsx	0,11,5

	slwi	6,6,3
	stfpsux	1,3,10
	lfpsx	1,11,6

	slwi	7,7,3
	stfpsux	2,3,10
	lfpsx	2,11,7

	slwi	8,8,3
	stfpsux	3,3,10
	lfpsx	3,11,8

	stfpsux	0,3,10
	stfpsux	1,3,10
	stfpsux	2,3,10
	stfpsux	3,3,10

	blr


	# i4complex -> float

.global	_ZN5LOFAR4RTCP8_convertINS_5TYPES9i4complexEEEvPSt7complexIfEPKT_j
_ZN5LOFAR4RTCP8_convertINS_5TYPES9i4complexEEEvPSt7complexIfEPKT_j:

	lis     11,_ZN5LOFAR4RTCP13_FIR_fp_tableE@ha
	la      11,_ZN5LOFAR4RTCP13_FIR_fp_tableE@l(11)

	srwi	5,5,2
	subi	5,5,2
	mtctr	5

	subi	3,3,8

	li	10,8
	li	9,1

	lbz	5,0(4)
	lbzux	6,4,9
	lbzux	7,4,9
	lbzux	8,4,9

	slwi	5,5,3
	lfpsx	0,11,5
	lbzux	5,4,9

	slwi	6,6,3
	lfpsx	1,11,6
	lbzux	6,4,9

	slwi	7,7,3
	lfpsx	2,11,7
	lbzux	7,4,9

	slwi	8,8,3
	lfpsx	3,11,8
	lbzux	8,4,9

0:
	slwi	5,5,3
	stfpsux	0,3,10
	lfpsx	0,11,5
	lbzux	5,4,9

	slwi	6,6,3
	stfpsux	1,3,10
	lfpsx	1,11,6
	lbzux	6,4,9

	slwi	7,7,3
	stfpsux	2,3,10
	lfpsx	2,11,7
	lbzux	7,4,9

	slwi	8,8,3
	stfpsux	3,3,10
	lfpsx	3,11,8
	lbzux	8,4,9

	bdnz	0b

	slwi	5,5,3
	stfpsux	0,3,10
	lfpsx	0,11,5

	slwi	6,6,3
	stfpsux	1,3,10
	lfpsx	1,11,6

	slwi	7,7,3
	stfpsux	2,3,10
	lfpsx	2,11,7

	slwi	8,8,3
	stfpsux	3,3,10
	lfpsx	3,11,8

	stfpsux	0,3,10
	stfpsux	1,3,10
	stfpsux	2,3,10
	stfpsux	3,3,10

	blr


.global	_apply_single_channel_delays
_apply_single_channel_delays:

	mtctr	4

	lfpdx	8,0,5
	lfpdx	9,0,6
	subi	4,3,8
	subi	3,3,8
	li	8,8

0:	lfpsux	0,3,8
	lfpsux	1,3,8
	
	fxpmul	4,0,8
	fxpmul	5,1,8
	fxpmul	10,8,9
	fxcxnpma 0,0,8,4
	fxcxnpma 1,1,8,5
	fxcxnpma 8,8,9,10

	stfpsux	0,4,8
	stfpsux	1,4,8

	bdnz	0b	

	blr

#endif
