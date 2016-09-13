	mtctr	9

#if NR_STATIONS >= 1
	mr	26,5
#endif
#if NR_STATIONS >= 2
	add	27,26,6
#endif
#if NR_STATIONS >= 3
	add	28,27,6
#endif
#if NR_STATIONS >= 4
	add	29,28,6
#endif
#if NR_STATIONS >= 5
	add	30,29,6
#endif
#if NR_STATIONS >= 6
	add	31,30,6
#endif

#if NR_STATIONS >= 2
	addi	11,8,-(NR_BEAMS-1)*8
#endif

#if NR_STATIONS >= 1 && NR_BEAMS >= 1
	lfpsx	0,0,7
#endif
#if NR_STATIONS >= 1 && NR_BEAMS >= 2
	lfpsux	1,7,0
#endif
#if NR_STATIONS >= 1 && NR_BEAMS >= 3
	lfpsux	2,7,0
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 1
	lfpsux	3,7,11
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 2
	lfpsux	4,7,0
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 3
	lfpsux	5,7,0
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 1
	lfpsux	6,7,11
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 2
	lfpsux	7,7,0
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 3
	lfpsux	8,7,0
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 1
	lfpsux	9,7,11
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 2
	lfpsux	10,7,0
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 3
	lfpsux	11,7,0
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 1
	lfpsux	12,7,11
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 2
	lfpsux	13,7,0
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 3
	lfpsux	14,7,0
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 1
	lfpsux	15,7,11
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 2
	lfpsux	16,7,0
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 3
	lfpsux	17,7,0
#endif

	# phase 0

#if !defined FIRST_TIME && NR_BEAMS >= 1
	mr 20,21
	; lfpsx  24,0,20
#endif
#if !defined FIRST_TIME && NR_BEAMS >= 2
	mr 22,23
	; lfpsx  25,0,22
#endif
#if !defined FIRST_TIME && NR_BEAMS >= 3
	mr 24,25
	; lfpsx  26,0,24
#endif

#if NR_STATIONS >= 1
	; lfpsx  31,0,26	# stat 0, time 2
#endif
#if NR_STATIONS >= 3
	; lfpsx	 29,0,28	# stat 2, time 2
#endif

	# phase 1

#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxpmul	 24,31,0	# stat 0, beam 0, time 2
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxcpmadd 24,31,0,24	# stat 0, beam 0, time 2
#endif
#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxpmul	 25,31,1	# stat 0, beam 1, time 2
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxcpmadd 25,31,1,25	# stat 0, beam 1, time 2
#endif
#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxpmul	 26,31,2	# stat 0, beam 2, time 2
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxcpmadd 26,31,2,26	# stat 0, beam 2, time 2
#endif

#if NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxcxnpma 24,31,0,24	# stat 0, beam 0, time 2
#endif
#if NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxcxnpma 25,31,1,25	# stat 0, beam 1, time 2
#endif
#if NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxcxnpma 26,31,2,26	# stat 0, beam 2, time 2
#endif

#if NR_STATIONS >= 5
	; lfpsx  31,0,30	# stat 4, time 2
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 1
	; lfpsux 27,20,0	# beam 0, time 3
#endif
#if !defined FIRST_TIME && NR_BEAMS >= 2
	; lfpsux 28,22,0	# beam 1, time 3
#endif
#if NR_STATIONS >= 1
	; lfpsux 21,26,0	# stat 0, time 3
#endif

#if NR_STATIONS >= 2
	; lfpsx  22,0,27	# stat 1, time 2
#endif

#if NR_STATIONS >= 3 && NR_BEAMS >= 1
	fxcpmadd 24,29,6,24	# stat 2, beam 0, time 2
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 2
	fxcpmadd 25,29,7,25	# stat 2, beam 1, time 2
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 3
	fxcpmadd 26,29,8,26	# stat 2, beam 2, time 2
#endif

#if NR_STATIONS >= 3 && NR_BEAMS >= 1
	fxcxnpma 24,29,6,24	# stat 2, beam 0, time 2
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 2
	fxcxnpma 25,29,7,25	# stat 2, beam 1, time 2
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 3
	fxcxnpma 26,29,8,26	# stat 2, beam 2, time 2
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 3
	; lfpsux 29,24,0	# beam 2, time 3
#endif

#if NR_STATIONS >= 5 && NR_BEAMS >= 1
	fxcpmadd 24,31,12,24	# stat 4, beam 0, time 2
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 2
	fxcpmadd 25,31,13,25	# stat 4, beam 1, time 2
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 3
	fxcpmadd 26,31,14,26	# stat 4, beam 2, time 2
#endif

#if NR_STATIONS >= 3
	; lfpsux 30,28,0	# stat 2, time 3
#endif

#if NR_STATIONS >= 5 && NR_BEAMS >= 1
	fxcxnpma 24,31,12,24	# stat 4, beam 0, time 2
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 2
	fxcxnpma 25,31,13,25	# stat 4, beam 1, time 2
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 3
	fxcxnpma 26,31,14,26	# stat 4, beam 2, time 2
#endif

#if NR_STATIONS >= 4
	; lfpsx  31,0,29	# stat 3, time 2
#endif

	# phase 2

#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxpmul	 27,21,0	# stat 0, beam 0, time 3
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxcpmadd 27,21,0,27	# stat 0, beam 0, time 3
#endif
#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxpmul	 28,21,1	# stat 0, beam 1, time 3
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxcpmadd 28,21,1,28	# stat 0, beam 1, time 3
#endif
#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxpmul	 29,21,2	# stat 0, beam 2, time 3
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxcpmadd 29,21,2,29	# stat 0, beam 2, time 3
#endif

#if NR_STATIONS >= 2 && NR_BEAMS >= 1
	fxcpmadd 24,22,3,24	# stat 1, beam 0, time 2
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 2
	fxcpmadd 25,22,4,25	# stat 1, beam 1, time 2
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 3
	fxcpmadd 26,22,5,26	# stat 1, beam 2, time 2
#endif

#if NR_STATIONS >= 5
	; lfpsux 18,30,0	# stat 4, time 3
#endif
#if NR_STATIONS >= 6
	; lfpsx  19,0,31	# stat 5, time 2
#endif

#if NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxcxnpma 27,21,0,27	# stat 0, beam 0, time 3
#endif
#if NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxcxnpma 28,21,1,28	# stat 0, beam 1, time 3
#endif
#if NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxcxnpma 29,21,2,29	# stat 0, beam 2, time 3
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 1
	; lfpsux 21,20,0	# beam 0, time 0
#endif

#if NR_STATIONS >= 2 && NR_BEAMS >= 1
	fxcxnpma 24,22,3,24	# stat 1, beam 0, time 2
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 2
	fxcxnpma 25,22,4,25	# stat 1, beam 1, time 2
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 3
	fxcxnpma 26,22,5,26	# stat 1, beam 2, time 2
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 2
	; lfpsux 22,22,0	# beam 1, time 0
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 3
	; lfpsux 23,24,0	# beam 2, time 0
#endif

#if NR_STATIONS >= 3 && NR_BEAMS >= 1
	fxcpmadd 27,30,6,27	# stat 2, beam 0, time 3
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 2
	fxcpmadd 28,30,7,28	# stat 2, beam 1, time 3
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 3
	fxcpmadd 29,30,8,29	# stat 2, beam 2, time 3
#endif

#if NR_STATIONS >= 4 && NR_BEAMS >= 1
	fxcpmadd 24,31,9,24	# stat 3, beam 0, time 2
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 2
	fxcpmadd 25,31,10,25	# stat 3, beam 1, time 2
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 3
	fxcpmadd 26,31,11,26	# stat 3, beam 2, time 2
#endif

#if NR_STATIONS >= 3 && NR_BEAMS >= 1
	fxcxnpma 27,30,6,27	# stat 2, beam 0, time 3
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 2
	fxcxnpma 28,30,7,28	# stat 2, beam 1, time 3
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 3
	fxcxnpma 29,30,8,29	# stat 2, beam 2, time 3
#endif

#if NR_STATIONS >= 2
	; lfpsux 30,27,0	# stat 1, time 3
#endif

#if NR_STATIONS >= 4 && NR_BEAMS >= 1
	fxcxnpma 24,31,9,24	# stat 3, beam 0, time 2
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 2
	fxcxnpma 25,31,10,25	# stat 3, beam 1, time 2
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 3
	fxcxnpma 26,31,11,26	# stat 3, beam 2, time 2
#endif

#if NR_STATIONS >= 1
	; lfpsux 31,26,0	# stat 0, time 0
#endif

#if NR_STATIONS >= 5 && NR_BEAMS >= 1
	fxcpmadd 27,18,12,27	# stat 4, beam 0, time 3
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 2
	fxcpmadd 28,18,13,28	# stat 4, beam 1, time 3
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 3
	fxcpmadd 29,18,14,29	# stat 4, beam 2, time 3
#endif

#if NR_STATIONS >= 6 && NR_BEAMS >= 1
	fxcpmadd 24,19,15,24	# stat 5, beam 0, time 2
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 2
	fxcpmadd 25,19,16,25	# stat 5, beam 1, time 2
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 3
	fxcpmadd 26,19,17,26	# stat 5, beam 2, time 2
#endif

#if NR_STATIONS >= 3
	; lfpsux 20,28,0	# stat 2, time 0
#endif

#if NR_STATIONS >= 5 && NR_BEAMS >= 1
	fxcxnpma 27,18,12,27	# stat 4, beam 0, time 3
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 2
	fxcxnpma 28,18,13,28	# stat 4, beam 1, time 3
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 3
	fxcxnpma 29,18,14,29	# stat 4, beam 2, time 3
#endif

#if NR_STATIONS >= 6 && NR_BEAMS >= 1
	fxcxnpma 24,19,15,24	# stat 5, beam 0, time 2
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 2
	fxcxnpma 25,19,16,25	# stat 5, beam 1, time 2
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 3
	fxcxnpma 26,19,17,26	# stat 5, beam 2, time 2
#endif

#if NR_STATIONS >= 4
	; lfpsux 19,29,0	# stat 3, time 3
#endif

	# phase 3

#if NR_STATIONS >= 2 && NR_BEAMS >= 1
	fxcpmadd 27,30,3,27	# stat 1, beam 0, time 3
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 2
	fxcpmadd 28,30,4,28	# stat 1, beam 1, time 3
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 3
	fxcpmadd 29,30,5,29	# stat 1, beam 2, time 3
#endif

#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxpmul	 21,31,0	# stat 0, beam 0, time 0
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxcpmadd 21,31,0,21	# stat 0, beam 0, time 0
#endif
#if NR_BEAMS >= 1
	; stfpsx  24,0,21	# beam 0, time 2
#endif
#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxpmul	 22,31,1	# stat 0, beam 1, time 0
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxcpmadd 22,31,1,22	# stat 0, beam 1, time 0
#endif
#if NR_BEAMS >= 2
	; stfpsx  25,0,23	# beam 1, time 2
#endif
#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxpmul	 23,31,2	# stat 0, beam 2, time 0
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxcpmadd 23,31,2,23	# stat 0, beam 2, time 0
#endif
#if NR_BEAMS >= 3
	; stfpsx  26,0,25	# beam 2, time 2
#endif

#if NR_STATIONS >= 2 && NR_BEAMS >= 1
	fxcxnpma 27,30,3,27	# stat 1, beam 0, time 3
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 2
	fxcxnpma 28,30,4,28	# stat 1, beam 1, time 3
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 3
	fxcxnpma 29,30,5,29	# stat 1, beam 2, time 3
#endif

#if NR_STATIONS >= 6
	; lfpsux 30,31,0	# stat 5, time 3
#endif

#if NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxcxnpma 21,31,0,21	# stat 0, beam 0, time 0
#endif
#if NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxcxnpma 22,31,1,22	# stat 0, beam 1, time 0
#endif
#if NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxcxnpma 23,31,2,23	# stat 0, beam 2, time 0
#endif

#if NR_STATIONS >= 5
	; lfpsux 31,30,0	# stat 4, time 0
#endif

#if NR_STATIONS >= 4 && NR_BEAMS >= 1
	fxcpmadd 27,19,9,27	# stat 3, beam 0, time 3
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 2
	fxcpmadd 28,19,10,28	# stat 3, beam 1, time 3
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 3
	fxcpmadd 29,19,11,29	# stat 3, beam 2, time 3
#endif

#if NR_STATIONS >= 1
	; lfpsux 24,26,0	# stat 0, time 1
#endif
#if NR_STATIONS >= 2
	; lfpsux 25,27,0	# stat 1, time 0
#endif

#if NR_STATIONS >= 3 && NR_BEAMS >= 1
	fxcpmadd 21,20,6,21	# stat 2, beam 0, time 0
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 2
	fxcpmadd 22,20,7,22	# stat 2, beam 1, time 0
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 3
	fxcpmadd 23,20,8,23	# stat 2, beam 2, time 0
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 1
	; lfpsux 18,20,0	# beam 0, time 1
#endif

#if NR_STATIONS >= 4 && NR_BEAMS >= 1
	fxcxnpma 27,19,9,27	# stat 3, beam 0, time 3
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 2
	fxcxnpma 28,19,10,28	# stat 3, beam 1, time 3
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 3
	fxcxnpma 29,19,11,29	# stat 3, beam 2, time 3
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 2
	; lfpsux 19,22,0	# beam 1, time 1
#endif

#if NR_STATIONS >= 3 && NR_BEAMS >= 1
	fxcxnpma 21,20,6,21	# stat 2, beam 0, time 0
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 2
	fxcxnpma 22,20,7,22	# stat 2, beam 1, time 0
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 3
	fxcxnpma 23,20,8,23	# stat 2, beam 2, time 0
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 3
	; lfpsux 20,24,0	# beam 2, time 1
#endif

#if NR_STATIONS >= 6 && NR_BEAMS >= 1
	fxcpmadd 27,30,15,27	# stat 5, beam 0, time 3
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 2
	fxcpmadd 28,30,16,28	# stat 5, beam 1, time 3
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 3
	fxcpmadd 29,30,17,29	# stat 5, beam 2, time 3
#endif

#if NR_STATIONS >= 5 && NR_BEAMS >= 1
	fxcpmadd 21,31,12,21	# stat 4, beam 0, time 0
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 2
	fxcpmadd 22,31,13,22	# stat 4, beam 1, time 0
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 3
	fxcpmadd 23,31,14,23	# stat 4, beam 2, time 0
#endif

#if NR_STATIONS >= 6 && NR_BEAMS >= 1
	fxcxnpma 27,30,15,27	# stat 5, beam 0, time 3
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 2
	fxcxnpma 28,30,16,28	# stat 5, beam 1, time 3
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 3
	fxcxnpma 29,30,17,29	# stat 5, beam 2, time 3
#endif

#if NR_STATIONS >= 3
	; lfpsux 30,28,0	# stat 2, time 1
#endif

#if NR_STATIONS >= 5 && NR_BEAMS >= 1
	fxcxnpma 21,31,12,21	# stat 4, beam 0, time 0
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 2
	fxcxnpma 22,31,13,22	# stat 4, beam 1, time 0
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 3
	fxcxnpma 23,31,14,23	# stat 4, beam 2, time 0
#endif

#if NR_STATIONS >= 4
	; lfpsux 31,29,0	# stat 3, time 0
#endif

	li	7,64

0:	# phase 0

#if NR_BEAMS >= 1
	; stfpsux 27,21,0	# beam 0, time 3
#endif

#if NR_STATIONS >= 5
	; lfpsux 27,30,0	# stat 4, time 1

#endif

#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxpmul	 18,24,0	# stat 0, beam 0, time 1
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxcpmadd 18,24,0,18	# stat 0, beam 0, time 1
#endif
#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxpmul	 19,24,1	# stat 0, beam 1, time 1
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxcpmadd 19,24,1,19	# stat 0, beam 1, time 1
#endif
#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxpmul	 20,24,2	# stat 0, beam 2, time 1
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxcpmadd 20,24,2,20	# stat 0, beam 2, time 1
#endif

#if NR_BEAMS >= 2
	; stfpsux 28,23,0	# beam 1, time 3
#endif

#if NR_STATIONS >= 6
	; lfpsux 28,31,0	# stat 5, time 0
#endif

#if NR_STATIONS >= 2 && NR_BEAMS >= 1
	fxcpmadd 21,25,3,21	# stat 1, beam 0, time 0
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 2
	fxcpmadd 22,25,4,22	# stat 1, beam 1, time 0
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 3
	fxcpmadd 23,25,5,23	# stat 1, beam 2, time 0
#endif

#if NR_BEAMS >= 3
	; stfpsux 29,25,0	# beam 2, time 3
#endif

#if NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxcxnpma 18,24,0,18	# stat 0, beam 0, time 1
#endif
#if NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxcxnpma 19,24,1,19	# stat 0, beam 1, time 1
#endif
#if NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxcxnpma 20,24,2,20	# stat 0, beam 2, time 1
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 1
	; lfpsux 24,20,0	# beam 0, time 2
#endif

#if NR_STATIONS >= 2 && NR_BEAMS >= 1
	fxcxnpma 21,25,3,21	# stat 1, beam 0, time 0
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 2
	fxcxnpma 22,25,4,22	# stat 1, beam 1, time 0
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 3
	fxcxnpma 23,25,5,23	# stat 1, beam 2, time 0
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 2
	; lfpsux 25,22,0	# beam 1, time 2
#endif

#if NR_STATIONS >= 3 && NR_BEAMS >= 1
	fxcpmadd 18,30,6,18	# stat 2, beam 0, time 1
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 2
	fxcpmadd 19,30,7,19	# stat 2, beam 1, time 1
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 3
	fxcpmadd 20,30,8,20	# stat 2, beam 2, time 1
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 3
	; lfpsux 26,24,0	# beam 2, time 2
#endif

#if NR_STATIONS >= 1
	; dcbt	 26,7		# stat 0
#endif

#if NR_STATIONS >= 4 && NR_BEAMS >= 1
	fxcpmadd 21,31,9,21	# stat 3, beam 0, time 0
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 2
	fxcpmadd 22,31,10,22	# stat 3, beam 1, time 0
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 3
	fxcpmadd 23,31,11,23	# stat 3, beam 2, time 0
#endif

#if NR_STATIONS >= 3 && NR_BEAMS >= 1
	fxcxnpma 18,30,6,18	# stat 2, beam 0, time 1
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 2
	fxcxnpma 19,30,7,19	# stat 2, beam 1, time 1
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 3
	fxcxnpma 20,30,8,20	# stat 2, beam 2, time 1
#endif

#if NR_STATIONS >= 2
	; lfpsux 30,27,0	# stat 1, time 1
#endif

#if NR_STATIONS >= 4 && NR_BEAMS >= 1
	fxcxnpma 21,31,9,21	# stat 3, beam 0, time 0
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 2
	fxcxnpma 22,31,10,22	# stat 3, beam 1, time 0
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 3
	fxcxnpma 23,31,11,23	# stat 3, beam 2, time 0
#endif

#if NR_STATIONS >= 1
	; lfpsux 31,26,0	# stat 0, time 2
#endif

#if NR_STATIONS >= 5 && NR_BEAMS >= 1
	fxcpmadd 18,27,12,18	# stat 4, beam 0, time 1
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 2
	fxcpmadd 19,27,13,19	# stat 4, beam 1, time 1
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 3
	fxcpmadd 20,27,14,20	# stat 4, beam 2, time 1
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 2
	; dcbt	 22,7		# beam 1
#endif

#if NR_STATIONS >= 6 && NR_BEAMS >= 1
	fxcpmadd 21,28,15,21	# stat 5, beam 0, time 0
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 2
	fxcpmadd 22,28,16,22	# stat 5, beam 1, time 0
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 3
	fxcpmadd 23,28,17,23	# stat 5, beam 2, time 0
#endif

#if NR_STATIONS >= 3
	; lfpsux 29,28,0	# stat 2, time 2
#endif

#if NR_STATIONS >= 5 && NR_BEAMS >= 1
	fxcxnpma 18,27,12,18	# stat 4, beam 0, time 1
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 2
	fxcxnpma 19,27,13,19	# stat 4, beam 1, time 1
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 3
	fxcxnpma 20,27,14,20	# stat 4, beam 2, time 1
#endif

#if NR_STATIONS >= 3
	; dcbt	 28,7		# stat 2
#endif

#if NR_STATIONS >= 6 && NR_BEAMS >= 1
	fxcxnpma 21,28,15,21	# stat 5, beam 0, time 0
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 2
	fxcxnpma 22,28,16,22	# stat 5, beam 1, time 0
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 3
	fxcxnpma 23,28,17,23	# stat 5, beam 2, time 0
#endif

#if NR_STATIONS >= 4
	; lfpsux 28,29,0	# stat 3, time 1
#endif

	# phase 1

#if NR_STATIONS >= 2 && NR_BEAMS >= 1
	fxcpmadd 18,30,3,18	# stat 1, beam 0, time 1
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 2
	fxcpmadd 19,30,4,19	# stat 1, beam 1, time 1
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 3
	fxcpmadd 20,30,5,20	# stat 1, beam 2, time 1
#endif

#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxpmul	 24,31,0	# stat 0, beam 0, time 2
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxcpmadd 24,31,0,24	# stat 0, beam 0, time 2
#endif
#if NR_BEAMS >= 1
	; stfpsux 21,21,0	# beam 0, time 0
#endif
#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxpmul	 25,31,1	# stat 0, beam 1, time 2
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxcpmadd 25,31,1,25	# stat 0, beam 1, time 2
#endif
#if NR_STATIONS >= 1
	; lfpsux 21,26,0	# stat 0, time 3
#endif
#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxpmul	 26,31,2	# stat 0, beam 2, time 2
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxcpmadd 26,31,2,26	# stat 0, beam 2, time 2
#endif
#if NR_BEAMS >= 2
	; stfpsux 22,23,0	# beam 1, time 0
#endif

#if NR_STATIONS >= 2 && NR_BEAMS >= 1
	fxcxnpma 18,30,3,18	# stat 1, beam 0, time 1
#endif
#if NR_STATIONS >= 2
	; lfpsux 22,27,0	# stat 1, time 2
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 2
	fxcxnpma 19,30,4,19	# stat 1, beam 1, time 1
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 3
	fxcxnpma 20,30,5,20	# stat 1, beam 2, time 1
#endif

#if NR_BEAMS >= 3
	; stfpsux 23,25,0	# beam 2, time 0
#endif

#if NR_STATIONS >= 6
	; lfpsux 30,31,0	# stat 5, time 1
#endif

#if NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxcxnpma 24,31,0,24	# stat 0, beam 0, time 2
#endif
#if NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxcxnpma 25,31,1,25	# stat 0, beam 1, time 2
#endif
#if NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxcxnpma 26,31,2,26	# stat 0, beam 2, time 2
#endif

#if NR_STATIONS >= 5
	; lfpsux 31,30,0	# stat 4, time 2
#endif

#if NR_STATIONS >= 4 && NR_BEAMS >= 1
	fxcpmadd 18,28,9,18	# stat 3, beam 0, time 1
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 2
	fxcpmadd 19,28,10,19	# stat 3, beam 1, time 1
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 3
	fxcpmadd 20,28,11,20	# stat 3, beam 2, time 1
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 1
	; lfpsux 27,20,0	# beam 0, time 3
#endif

#if NR_STATIONS >= 3 && NR_BEAMS >= 1
	fxcpmadd 24,29,6,24	# stat 2, beam 0, time 2
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 2
	fxcpmadd 25,29,7,25	# stat 2, beam 1, time 2
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 3
	fxcpmadd 26,29,8,26	# stat 2, beam 2, time 2
#endif

#if NR_STATIONS >= 5
	; dcbt	 30,7		# stat 4
#endif

#if NR_STATIONS >= 4 && NR_BEAMS >= 1
	fxcxnpma 18,28,9,18	# stat 3, beam 0, time 1
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 2
	fxcxnpma 19,28,10,19	# stat 3, beam 1, time 1
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 3
	fxcxnpma 20,28,11,20	# stat 3, beam 2, time 1
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 2
	; lfpsux 28,22,0	# beam 1, time 3
#endif

#if NR_STATIONS >= 3 && NR_BEAMS >= 1
	fxcxnpma 24,29,6,24	# stat 2, beam 0, time 2
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 2
	fxcxnpma 25,29,7,25	# stat 2, beam 1, time 2
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 3
	fxcxnpma 26,29,8,26	# stat 2, beam 2, time 2
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 3
	; lfpsux 29,24,0	# beam 2, time 3
#endif

#if NR_STATIONS >= 6 && NR_BEAMS >= 1
	fxcpmadd 18,30,15,18	# stat 5, beam 0, time 1
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 2
	fxcpmadd 19,30,16,19	# stat 5, beam 1, time 1
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 3
	fxcpmadd 20,30,17,20	# stat 5, beam 2, time 1
#endif

#if NR_STATIONS >= 2
	; dcbt	 27,7		# stat 1
#endif

#if NR_STATIONS >= 5 && NR_BEAMS >= 1
	fxcpmadd 24,31,12,24	# stat 4, beam 0, time 2
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 2
	fxcpmadd 25,31,13,25	# stat 4, beam 1, time 2
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 3
	fxcpmadd 26,31,14,26	# stat 4, beam 2, time 2
#endif

#if NR_STATIONS >= 6 && NR_BEAMS >= 1
	fxcxnpma 18,30,15,18	# stat 5, beam 0, time 1
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 2
	fxcxnpma 19,30,16,19	# stat 5, beam 1, time 1
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 3
	fxcxnpma 20,30,17,20	# stat 5, beam 2, time 1
#endif

#if NR_STATIONS >= 3
	; lfpsux 30,28,0	# stat 2, time 3
#endif

#if NR_STATIONS >= 5 && NR_BEAMS >= 1
	fxcxnpma 24,31,12,24	# stat 4, beam 0, time 2
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 2
	fxcxnpma 25,31,13,25	# stat 4, beam 1, time 2
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 3
	fxcxnpma 26,31,14,26	# stat 4, beam 2, time 2
#endif

#if NR_STATIONS >= 4
	; lfpsux 31,29,0	# stat 3, time 2
#endif

	# phase 2

#if NR_BEAMS >= 1
	; stfpsux 18,21,0	# beam 0, time 1
#endif

#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxpmul	 27,21,0	# stat 0, beam 0, time 3
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxcpmadd 27,21,0,27	# stat 0, beam 0, time 3
#endif
#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxpmul	 28,21,1	# stat 0, beam 1, time 3
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxcpmadd 28,21,1,28	# stat 0, beam 1, time 3
#endif
#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxpmul	 29,21,2	# stat 0, beam 2, time 3
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxcpmadd 29,21,2,29	# stat 0, beam 2, time 3
#endif

#if NR_STATIONS >= 5
	; lfpsux 18,30,0	# stat 4, time 3
#endif

#if NR_BEAMS >= 2
	; stfpsux 19,23,0	# beam 1, time 1
#endif

#if NR_STATIONS >= 2 && NR_BEAMS >= 1
	fxcpmadd 24,22,3,24	# stat 1, beam 0, time 2
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 2
	fxcpmadd 25,22,4,25	# stat 1, beam 1, time 2
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 3
	fxcpmadd 26,22,5,26	# stat 1, beam 2, time 2
#endif

#if NR_STATIONS >= 6
	; lfpsux 19,31,0	# stat 5, time 2
#endif

#if NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxcxnpma 27,21,0,27	# stat 0, beam 0, time 3
#endif
#if NR_BEAMS >= 3
	; stfpsux 20,25,0	# beam 2, time 1
#endif
#if NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxcxnpma 28,21,1,28	# stat 0, beam 1, time 3
#endif
#if NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxcxnpma 29,21,2,29	# stat 0, beam 2, time 3
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 1
	; lfpsux 21,20,0	# beam 0, time 0
#endif

#if NR_STATIONS >= 2 && NR_BEAMS >= 1
	fxcxnpma 24,22,3,24	# stat 1, beam 0, time 2
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 2
	fxcxnpma 25,22,4,25	# stat 1, beam 1, time 2
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 3
	fxcxnpma 26,22,5,26	# stat 1, beam 2, time 2
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 2
	; lfpsux 22,22,0	# beam 1, time 0
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 3
	; lfpsux 23,24,0	# beam 2, time 0
#endif

#if NR_STATIONS >= 3 && NR_BEAMS >= 1
	fxcpmadd 27,30,6,27	# stat 2, beam 0, time 3
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 2
	fxcpmadd 28,30,7,28	# stat 2, beam 1, time 3
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 3
	fxcpmadd 29,30,8,29	# stat 2, beam 2, time 3
#endif

#if NR_STATIONS >= 4
	; dcbt	 29,7		# stat 3
#endif

#if NR_STATIONS >= 4 && NR_BEAMS >= 1
	fxcpmadd 24,31,9,24	# stat 3, beam 0, time 2
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 2
	fxcpmadd 25,31,10,25	# stat 3, beam 1, time 2
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 3
	fxcpmadd 26,31,11,26	# stat 3, beam 2, time 2
#endif

#if NR_STATIONS >= 3 && NR_BEAMS >= 1
	fxcxnpma 27,30,6,27	# stat 2, beam 0, time 3
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 2
	fxcxnpma 28,30,7,28	# stat 2, beam 1, time 3
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 3
	fxcxnpma 29,30,8,29	# stat 2, beam 2, time 3
#endif

#if NR_STATIONS >= 2
	; lfpsux 30,27,0	# stat 1, time 3
#endif

#if NR_STATIONS >= 4 && NR_BEAMS >= 1
	fxcxnpma 24,31,9,24	# stat 3, beam 0, time 2
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 2
	fxcxnpma 25,31,10,25	# stat 3, beam 1, time 2
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 3
	fxcxnpma 26,31,11,26	# stat 3, beam 2, time 2
#endif

#if NR_STATIONS >= 1
	; lfpsux 31,26,0	# stat 0, time 0
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 3
	; dcbt	 24,7		# beam 2
#endif

#if NR_STATIONS >= 5 && NR_BEAMS >= 1
	fxcpmadd 27,18,12,27	# stat 4, beam 0, time 3
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 2
	fxcpmadd 28,18,13,28	# stat 4, beam 1, time 3
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 3
	fxcpmadd 29,18,14,29	# stat 4, beam 2, time 3
#endif

#if NR_STATIONS >= 6 && NR_BEAMS >= 1
	fxcpmadd 24,19,15,24	# stat 5, beam 0, time 2
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 2
	fxcpmadd 25,19,16,25	# stat 5, beam 1, time 2
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 3
	fxcpmadd 26,19,17,26	# stat 5, beam 2, time 2
#endif

#if NR_STATIONS >= 3
	; lfpsux 20,28,0	# stat 2, time 0
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 1
	; dcbt	 20,7		# beam 0
#endif

#if NR_STATIONS >= 5 && NR_BEAMS >= 1
	fxcxnpma 27,18,12,27	# stat 4, beam 0, time 3
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 2
	fxcxnpma 28,18,13,28	# stat 4, beam 1, time 3
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 3
	fxcxnpma 29,18,14,29	# stat 4, beam 2, time 3
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 1
	fxcxnpma 24,19,15,24	# stat 5, beam 0, time 2
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 2
	fxcxnpma 25,19,16,25	# stat 5, beam 1, time 2
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 3
	fxcxnpma 26,19,17,26	# stat 5, beam 2, time 2
#endif

#if NR_STATIONS >= 4
	; lfpsux 19,29,0	# stat 3, time 3
#endif

	# phase 3

#if NR_STATIONS >= 2 && NR_BEAMS >= 1
	fxcpmadd 27,30,3,27	# stat 1, beam 0, time 3
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 2
	fxcpmadd 28,30,4,28	# stat 1, beam 1, time 3
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 3
	fxcpmadd 29,30,5,29	# stat 1, beam 2, time 3
#endif

#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxpmul	 21,31,0	# stat 0, beam 0, time 0
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxcpmadd 21,31,0,21	# stat 0, beam 0, time 0
#endif
#if NR_BEAMS >= 1
	; stfpsux 24,21,0	# beam 0, time 2
#endif
#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxpmul	 22,31,1	# stat 0, beam 1, time 0
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxcpmadd 22,31,1,22	# stat 0, beam 1, time 0
#endif
#if NR_STATIONS >= 1
	; lfpsux 24,26,0	# stat 0, time 1
#endif
#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxpmul	 23,31,2	# stat 0, beam 2, time 0
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxcpmadd 23,31,2,23	# stat 0, beam 2, time 0
#endif

#if NR_BEAMS >= 2
	; stfpsux 25,23,0	# beam 1, time 2
#endif

#if NR_STATIONS >= 2 && NR_BEAMS >= 1
	fxcxnpma 27,30,3,27	# stat 1, beam 0, time 3
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 2
	fxcxnpma 28,30,4,28	# stat 1, beam 1, time 3
#endif
#if NR_STATIONS >= 2
	; lfpsux 25,27,0	# stat 1, time 0
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 3
	fxcxnpma 29,30,5,29	# stat 1, beam 2, time 3
#endif
#if NR_STATIONS >= 6
	; lfpsux 30,31,0	# stat 5, time 3
#endif

#if NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxcxnpma 21,31,0,21	# stat 0, beam 0, time 0
#endif
#if NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxcxnpma 22,31,1,22	# stat 0, beam 1, time 0
#endif
#if NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxcxnpma 23,31,2,23	# stat 0, beam 2, time 0
#endif

#if NR_STATIONS >= 5
	; lfpsux 31,30,0	# stat 4, time 0
#endif

#if NR_STATIONS >= 4 && NR_BEAMS >= 1
	fxcpmadd 27,19,9,27	# stat 3, beam 0, time 3
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 2
	fxcpmadd 28,19,10,28	# stat 3, beam 1, time 3
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 3
	fxcpmadd 29,19,11,29	# stat 3, beam 2, time 3
#endif

#if NR_BEAMS >= 3
	; stfpsux 26,25,0	# beam 2, time 2
#endif

#if NR_STATIONS >= 3 && NR_BEAMS >= 1
	fxcpmadd 21,20,6,21	# stat 2, beam 0, time 0
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 2
	fxcpmadd 22,20,7,22	# stat 2, beam 1, time 0
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 3
	fxcpmadd 23,20,8,23	# stat 2, beam 2, time 0
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 1
	; lfpsux 18,20,0	# beam 0, time 0
#endif

#if NR_STATIONS >= 4 && NR_BEAMS >= 1
	fxcxnpma 27,19,9,27	# stat 3, beam 0, time 3
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 2
	fxcxnpma 28,19,10,28	# stat 3, beam 1, time 3
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 3
	fxcxnpma 29,19,11,29	# stat 3, beam 2, time 3
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 2
	; lfpsux 19,22,0	# beam 1, time 0
#endif

#if NR_STATIONS >= 3 && NR_BEAMS >= 1
	fxcxnpma 21,20,6,21	# stat 2, beam 0, time 0
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 2
	fxcxnpma 22,20,7,22	# stat 2, beam 1, time 0
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 3
	fxcxnpma 23,20,8,23	# stat 2, beam 2, time 0
#endif

#if !defined FIRST_TIME && NR_BEAMS >= 3
	; lfpsux 20,24,0	# beam 2, time 0
#endif

#if NR_STATIONS >= 6 && NR_BEAMS >= 1
	fxcpmadd 27,30,15,27	# stat 5, beam 0, time 3
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 2
	fxcpmadd 28,30,16,28	# stat 5, beam 1, time 3
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 3
	fxcpmadd 29,30,17,29	# stat 5, beam 2, time 3
#endif

#if NR_STATIONS >= 6
	; dcbt	 31,7		# stat 5
#endif

#if NR_STATIONS >= 5 && NR_BEAMS >= 1
	fxcpmadd 21,31,12,21	# stat 4, beam 0, time 0
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 2
	fxcpmadd 22,31,13,22	# stat 4, beam 1, time 0
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 3
	fxcpmadd 23,31,14,23	# stat 4, beam 2, time 0
#endif

#if NR_STATIONS >= 6 && NR_BEAMS >= 1
	fxcxnpma 27,30,15,27	# stat 5, beam 0, time 3
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 2
	fxcxnpma 28,30,16,28	# stat 5, beam 1, time 3
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 3
	fxcxnpma 29,30,17,29	# stat 5, beam 2, time 3
#endif

#if NR_STATIONS >= 3
	; lfpsux 30,28,0	# stat 2, time 1
#endif

#if NR_STATIONS >= 5 && NR_BEAMS >= 1
	fxcxnpma 21,31,12,21	# stat 4, beam 0, time 0
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 2
	fxcxnpma 22,31,13,22	# stat 4, beam 1, time 0
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 3
	fxcxnpma 23,31,14,23	# stat 4, beam 2, time 0
#endif

#if NR_STATIONS >= 4
	; lfpsux 31,29,0	# stat 3, time 0
#endif

	bdnz	 0b

	# epilogue
	# phase 0

#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxpmul	 18,24,0	# stat 0, beam 0, time 1
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxcpmadd 18,24,0,18	# stat 0, beam 0, time 1
#endif
#if NR_BEAMS >= 1
	; stfpsux 27,21,0	# beam 0, time 3
#endif
#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxpmul	 19,24,1	# stat 0, beam 1, time 1
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxcpmadd 19,24,1,19	# stat 0, beam 1, time 1
#endif
#if NR_BEAMS >= 2
	; stfpsux 28,23,0	# beam 1, time 3
#endif
#if defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxpmul	 20,24,2	# stat 0, beam 2, time 1
#elif !defined FIRST_TIME && NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxcpmadd 20,24,2,20	# stat 0, beam 2, time 1
#endif
#if NR_BEAMS >= 3
	; stfpsux 29,25,0	# beam 2, time 3
#endif

#if NR_STATIONS >= 2 && NR_BEAMS >= 1
	fxcpmadd 21,25,3,21	# stat 1, beam 0, time 0
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 2
	fxcpmadd 22,25,4,22	# stat 1, beam 1, time 0
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 3
	fxcpmadd 23,25,5,23	# stat 1, beam 2, time 0
#endif

#if NR_STATIONS >= 5
	; lfpsux 27,30,0	# stat 4, time 1
#endif
#if NR_STATIONS >= 6
	; lfpsux 28,31,0	# stat 5, time 0
#endif

#if NR_STATIONS >= 1 && NR_BEAMS >= 1
	fxcxnpma 18,24,0,18	# stat 0, beam 0, time 1
#endif
#if NR_STATIONS >= 1 && NR_BEAMS >= 2
	fxcxnpma 19,24,1,19	# stat 0, beam 1, time 1
#endif
#if NR_STATIONS >= 1 && NR_BEAMS >= 3
	fxcxnpma 20,24,2,20	# stat 0, beam 2, time 1
#endif

#if NR_STATIONS >= 2 && NR_BEAMS >= 1
	fxcxnpma 21,25,3,21	# stat 1, beam 0, time 0
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 2
	fxcxnpma 22,25,4,22	# stat 1, beam 1, time 0
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 3
	fxcxnpma 23,25,5,23	# stat 1, beam 2, time 0
#endif

#if NR_STATIONS >= 3 && NR_BEAMS >= 1
	fxcpmadd 18,30,6,18	# stat 2, beam 0, time 1
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 2
	fxcpmadd 19,30,7,19	# stat 2, beam 1, time 1
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 3
	fxcpmadd 20,30,8,20	# stat 2, beam 2, time 1
#endif

#if NR_STATIONS >= 4 && NR_BEAMS >= 1
	fxcpmadd 21,31,9,21	# stat 3, beam 0, time 0
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 2
	fxcpmadd 22,31,10,22	# stat 3, beam 1, time 0
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 3
	fxcpmadd 23,31,11,23	# stat 3, beam 2, time 0
#endif

#if NR_STATIONS >= 3 && NR_BEAMS >= 1
	fxcxnpma 18,30,6,18	# stat 2, beam 0, time 1
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 2
	fxcxnpma 19,30,7,19	# stat 2, beam 1, time 1
#endif
#if NR_STATIONS >= 3 && NR_BEAMS >= 3
	fxcxnpma 20,30,8,20	# stat 2, beam 2, time 1
#endif

#if NR_STATIONS >= 2
	; lfpsux 30,27,0	# stat 1, time 1
#endif

#if NR_STATIONS >= 4 && NR_BEAMS >= 1
	fxcxnpma 21,31,9,21	# stat 3, beam 0, time 0
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 2
	fxcxnpma 22,31,10,22	# stat 3, beam 1, time 0
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 3
	fxcxnpma 23,31,11,23	# stat 3, beam 2, time 0
#endif

#if NR_STATIONS >= 5 && NR_BEAMS >= 1
	fxcpmadd 18,27,12,18	# stat 4, beam 0, time 1
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 2
	fxcpmadd 19,27,13,19	# stat 4, beam 1, time 1
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 3
	fxcpmadd 20,27,14,20	# stat 4, beam 2, time 1
#endif

#if NR_STATIONS >= 6 && NR_BEAMS >= 1
	fxcpmadd 21,28,15,21	# stat 5, beam 0, time 0
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 2
	fxcpmadd 22,28,16,22	# stat 5, beam 1, time 0
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 3
	fxcpmadd 23,28,17,23	# stat 5, beam 2, time 0
#endif

#if NR_STATIONS >= 5 && NR_BEAMS >= 1
	fxcxnpma 18,27,12,18	# stat 4, beam 0, time 1
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 2
	fxcxnpma 19,27,13,19	# stat 4, beam 1, time 1
#endif
#if NR_STATIONS >= 5 && NR_BEAMS >= 3
	fxcxnpma 20,27,14,20	# stat 4, beam 2, time 1
#endif

#if NR_STATIONS >= 6 && NR_BEAMS >= 1
	fxcxnpma 21,28,15,21	# stat 5, beam 0, time 0
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 2
	fxcxnpma 22,28,16,22	# stat 5, beam 1, time 0
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 3
	fxcxnpma 23,28,17,23	# stat 5, beam 2, time 0
#endif

#if NR_STATIONS >= 4
	; lfpsux 28,29,0	# stat 3, time 1
#endif

	# phase 1

#if NR_STATIONS >= 2 && NR_BEAMS >= 1
	fxcpmadd 18,30,3,18	# stat 1, beam 0, time 1
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 2
	fxcpmadd 19,30,4,19	# stat 1, beam 1, time 1
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 3
	fxcpmadd 20,30,5,20	# stat 1, beam 2, time 1
#endif

#if NR_BEAMS >= 1
	; stfpsux 21,21,0	# beam 0, time 0
#endif
#if NR_BEAMS >= 2
	; stfpsux 22,23,0	# beam 1, time 0
#endif
#if NR_BEAMS >= 3
	; stfpsux 23,25,0	# beam 2, time 0
#endif

#if NR_STATIONS >= 2 && NR_BEAMS >= 1
	fxcxnpma 18,30,3,18	# stat 1, beam 0, time 1
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 2
	fxcxnpma 19,30,4,19	# stat 1, beam 1, time 1
#endif
#if NR_STATIONS >= 2 && NR_BEAMS >= 3
	fxcxnpma 20,30,5,20	# stat 1, beam 2, time 1
#endif

#if NR_STATIONS >= 6
	; lfpsux 30,31,0	# stat 5, time 1
#endif

#if NR_STATIONS >= 4 && NR_BEAMS >= 1
	fxcpmadd 18,28,9,18	# stat 3, beam 0, time 1
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 2
	fxcpmadd 19,28,10,19	# stat 3, beam 1, time 1
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 3
	fxcpmadd 20,28,11,20	# stat 3, beam 2, time 1
#endif

#if NR_STATIONS >= 4 && NR_BEAMS >= 1
	fxcxnpma 18,28,9,18	# stat 3, beam 0, time 1
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 2
	fxcxnpma 19,28,10,19	# stat 3, beam 1, time 1
#endif
#if NR_STATIONS >= 4 && NR_BEAMS >= 3
	fxcxnpma 20,28,11,20	# stat 3, beam 2, time 1
#endif

#if NR_STATIONS >= 6 && NR_BEAMS >= 1
	fxcpmadd 18,30,15,18	# stat 5, beam 0, time 1
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 2
	fxcpmadd 19,30,16,19	# stat 5, beam 1, time 1
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 3
	fxcpmadd 20,30,17,20	# stat 5, beam 2, time 1
#endif

#if NR_STATIONS >= 6 && NR_BEAMS >= 1
	fxcxnpma 18,30,15,18	# stat 5, beam 0, time 1
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 2
	fxcxnpma 19,30,16,19	# stat 5, beam 1, time 1
#endif
#if NR_STATIONS >= 6 && NR_BEAMS >= 3
	fxcxnpma 20,30,17,20	# stat 5, beam 2, time 1
#endif

	# phase 2

#if NR_BEAMS >= 1
	; stfpsux 18,21,0	# beam 0, time 1
#endif
#if NR_BEAMS >= 2
	; stfpsux 19,23,0	# beam 1, time 1
#endif
#if NR_BEAMS >= 3
	; stfpsux 20,25,0	# beam 2, time 1
#endif
