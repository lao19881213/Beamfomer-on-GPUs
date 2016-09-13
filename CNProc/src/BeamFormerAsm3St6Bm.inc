
	# phase 0

	lfpsx	 26,0,23	# stat 0, time 0

	lfpsx	 27,0,24	# stat 1, time 0

	lfpsx	 22,0,26	# beam 0, time 0

	# phase 1

	lfpsx	 28,0,25	# stat 2, time 0
	lfpsx	 23,0,27	# beam 1, time 0

	fxcpmadd 22,26,0,22	# stat 0, beam 0, time 0
	lfpsx	 24,0,28	# beam 2, time 0
	lfpsux	 29,23,0	# stat 0, time 1

	fxcxnpma 22,26,0,22	# stat 0, beam 0, time 0
	fxcpmadd 23,26,1,23	# stat 0, beam 1, time 0
	lfpsx	 25,0,29	# beam 3, time 0

	fxcpmadd 22,27,6,22	# stat 1, beam 0, time 0
	fxcxnpma 23,26,1,23	# stat 0, beam 1, time 0
	fxcpmadd 24,26,2,24	# stat 0, beam 2, time 0
	lfpsx	 18,0,30	# beam 4, time 0
	lfpsux	 30,24,0	# stat 1, time 1

	fxcxnpma 22,27,6,22	# stat 1, beam 0, time 0
	fxcpmadd 23,27,7,23	# stat 1, beam 1, time 0
	fxcxnpma 24,26,2,24	# stat 0, beam 2, time 0
	fxcpmadd 25,26,3,25	# stat 0, beam 3, time 0
	lfpsx	 19,0,31	# beam 5, time 0

	fxcpmadd 22,28,12,22	# stat 2, beam 0, time 0
	fxcxnpma 23,27,7,23	# stat 1, beam 1, time 0
	fxcpmadd 24,27,8,24	# stat 1, beam 2, time 0
	fxcxnpma 25,26,3,25	# stat 0, beam 3, time 0
	fxcpmadd 18,26,4,18	# stat 0, beam 4, time 0
	lfpsux	 20,26,0	# beam 0, time 1

	# phase 2

	fxcxnpma 22,28,12,22	# stat 2, beam 0, time 0
	lfpsux	 31,25,0	# stat 2, time 1
	fxcpmadd 23,28,13,23	# stat 2, beam 1, time 0
	fxcxnpma 24,27,8,24	# stat 1, beam 2, time 0
	fxcpmadd 25,27,9,25	# stat 1, beam 3, time 0
	fxcxnpma 18,26,4,18	# stat 0, beam 4, time 0
	fxcpmadd 19,26,5,19	# stat 0, beam 5, time 0
	lfpsux	 21,27,0	# beam 1, time 1

	fxcpmadd 20,29,0,20	# stat 0, beam 0, time 1
	stfpsx	 22,26,8	# beam 0, time 0
	fxcxnpma 23,28,13,23	# stat 2, beam 1, time 0
	lfpsux	 22,28,0	# beam 2, time 1
	fxcpmadd 24,28,14,24	# stat 2, beam 2, time 0
	fxcxnpma 25,27,9,25	# stat 1, beam 3, time 0
	fxcpmadd 18,27,10,18	# stat 1, beam 4, time 0
	fxcxnpma 19,26,5,19	# stat 0, beam 5, time 0
	lfpsux	 26,23,0	# stat 0, time 2

	fxcxnpma 20,29,0,20	# stat 0, beam 0, time 1
	fxcpmadd 21,29,1,21	# stat 0, beam 1, time 1
	stfpsx	 23,27,8	# beam 1, time 0
	fxcxnpma 24,28,14,24	# stat 2, beam 2, time 0
	lfpsux	 23,29,0	# beam 3, time 1
	fxcpmadd 25,28,15,25	# stat 2, beam 3, time 0
	fxcxnpma 18,27,10,18	# stat 1, beam 4, time 0
	fxcpmadd 19,27,11,19	# stat 1, beam 5, time 0

	fxcpmadd 20,30,6,20	# stat 1, beam 0, time 1
	fxcxnpma 21,29,1,21	# stat 0, beam 1, time 1
	fxcpmadd 22,29,2,22	# stat 0, beam 2, time 1
	stfpsx	 24,28,8	# beam 2, time 0
	fxcxnpma 25,28,15,25	# stat 2, beam 3, time 0
	lfpsux	 24,30,0	# beam 4, time 1
	fxcpmadd 18,28,16,18	# stat 2, beam 4, time 0
	fxcxnpma 19,27,11,19	# stat 1, beam 5, time 0
	lfpsux	 27,24,0	# stat 1, time 2

	fxcxnpma 20,30,6,20	# stat 1, beam 0, time 1
	fxcpmadd 21,30,7,21	# stat 1, beam 1, time 1
	fxcxnpma 22,29,2,22	# stat 0, beam 2, time 1
	fxcpmadd 23,29,3,23	# stat 0, beam 3, time 1
	stfpsx	 25,29,8	# beam 3, time 0
	fxcxnpma 18,28,16,18	# stat 2, beam 4, time 0
	lfpsux	 25,31,0	# beam 5, time 1
	fxcpmadd 19,28,17,19	# stat 2, beam 5, time 0

	fxcpmadd 20,31,12,20	# stat 2, beam 0, time 1
	fxcxnpma 21,30,7,21	# stat 1, beam 1, time 1
	fxcpmadd 22,30,8,22	# stat 1, beam 2, time 1
	fxcxnpma 23,29,3,23	# stat 0, beam 3, time 1
	fxcpmadd 24,29,4,24	# stat 0, beam 4, time 1
	stfpsx	 18,30,8	# beam 4, time 0
	fxcxnpma 19,28,17,19	# stat 2, beam 5, time 0
	lfpsux	 18,26,0	# beam 0, time 2

	# phase 3

	fxcxnpma 20,31,12,20	# stat 2, beam 0, time 1
	lfpsux	 28,25,0	# stat 2, time 2
	fxcpmadd 21,31,13,21	# stat 2, beam 1, time 1
	fxcxnpma 22,30,8,22	# stat 1, beam 2, time 1
	fxcpmadd 23,30,9,23	# stat 1, beam 3, time 1
	fxcxnpma 24,29,4,24	# stat 0, beam 4, time 1
	stfpsx	 19,31,8	# beam 5, time 0
	fxcpmadd 25,29,5,25	# stat 0, beam 5, time 1
	lfpsux	 19,27,0	# beam 1, time 2

	fxcpmadd 18,26,0,18	# stat 0, beam 0, time 2
	stfpsx	 20,26,8	# beam 0, time 1
	fxcxnpma 21,31,13,21	# stat 2, beam 1, time 1
	lfpsux	 20,28,0	# beam 2, time 2
	fxcpmadd 22,31,14,22	# stat 2, beam 2, time 1
	fxcxnpma 23,30,9,23	# stat 1, beam 3, time 1
	fxcpmadd 24,30,10,24	# stat 1, beam 4, time 1
	fxcxnpma 25,29,5,25	# stat 0, beam 5, time 1
	lfpsux	 29,23,0	# stat 0, time 3

	fxcxnpma 18,26,0,18	# stat 0, beam 0, time 2
	fxcpmadd 19,26,1,19	# stat 0, beam 1, time 2
	stfpsx	 21,27,8	# beam 1, time 1
	fxcxnpma 22,31,14,22	# stat 2, beam 2, time 1
	lfpsux	 21,29,0	# beam 3, time 2
	fxcpmadd 23,31,15,23	# stat 2, beam 3, time 1
	fxcxnpma 24,30,10,24	# stat 1, beam 4, time 1
	fxcpmadd 25,30,11,25	# stat 1, beam 5, time 1

	fxcpmadd 18,27,6,18	# stat 1, beam 0, time 2
	fxcxnpma 19,26,1,19	# stat 0, beam 1, time 2
	fxcpmadd 20,26,2,20	# stat 0, beam 2, time 2
	stfpsx	 22,28,8	# beam 2, time 1
	fxcxnpma 23,31,15,23	# stat 2, beam 3, time 1
	lfpsux	 22,30,0	# beam 4, time 2
	fxcpmadd 24,31,16,24	# stat 2, beam 4, time 1
	fxcxnpma 25,30,11,25	# stat 1, beam 5, time 1
	lfpsux	 30,24,0	# stat 1, time 3

	fxcxnpma 18,27,6,18	# stat 1, beam 0, time 2
	fxcpmadd 19,27,7,19	# stat 1, beam 1, time 2
	fxcxnpma 20,26,2,20	# stat 0, beam 2, time 2
	fxcpmadd 21,26,3,21	# stat 0, beam 3, time 2
	stfpsx	 23,29,8	# beam 3, time 1
	fxcxnpma 24,31,16,24	# stat 2, beam 4, time 1
	lfpsux	 23,31,0	# beam 5, time 2
	fxcpmadd 25,31,17,25	# stat 2, beam 5, time 1

	fxcpmadd 18,28,12,18	# stat 2, beam 0, time 2
	fxcxnpma 19,27,7,19	# stat 1, beam 1, time 2
	fxcpmadd 20,27,8,20	# stat 1, beam 2, time 2
	fxcxnpma 21,26,3,21	# stat 0, beam 3, time 2
	fxcpmadd 22,26,4,22	# stat 0, beam 4, time 2
	stfpsx	 24,30,8	# beam 4, time 1
	fxcxnpma 25,31,17,25	# stat 2, beam 5, time 1
	lfpsux	 24,26,0	# beam 0, time 3



0:
	# phase 0

	fxcxnpma 18,28,12,18	# stat 2, beam 0, time 2
	lfpsux	 31,25,0	# stat 2, time 3
	fxcpmadd 19,28,13,19	# stat 2, beam 1, time 2
	fxcxnpma 20,27,8,20	# stat 1, beam 2, time 2
	fxcpmadd 21,27,9,21	# stat 1, beam 3, time 2
	fxcxnpma 22,26,4,22	# stat 0, beam 4, time 2
	stfpsx	 25,31,8	# beam 5, time 1
	fxcpmadd 23,26,5,23	# stat 0, beam 5, time 2
	lfpsux	 25,27,0	# beam 1, time 3

	fxcpmadd 24,29,0,24	# stat 0, beam 0, time 3
	stfpsx	 18,26,8	# beam 0, time 2
	fxcxnpma 19,28,13,19	# stat 2, beam 1, time 2
	lfpsux	 18,28,0	# beam 2, time 3
	fxcpmadd 20,28,14,20	# stat 2, beam 2, time 2
	fxcxnpma 21,27,9,21	# stat 1, beam 3, time 2
	fxcpmadd 22,27,10,22	# stat 1, beam 4, time 2
	fxcxnpma 23,26,5,23	# stat 0, beam 5, time 2
	lfpsux	 26,23,0	# stat 0, time 0

	fxcxnpma 24,29,0,24	# stat 0, beam 0, time 3
	dcbt	 23,9
	fxcpmadd 25,29,1,25	# stat 0, beam 1, time 3
	stfpsx	 19,27,8	# beam 1, time 2
	fxcxnpma 20,28,14,20	# stat 2, beam 2, time 2
	lfpsux	 19,29,0	# beam 3, time 3
	fxcpmadd 21,28,15,21	# stat 2, beam 3, time 2
	fxcxnpma 22,27,10,22	# stat 1, beam 4, time 2
	fxcpmadd 23,27,11,23	# stat 1, beam 5, time 2

	fxcpmadd 24,30,6,24	# stat 1, beam 0, time 3
	fxcxnpma 25,29,1,25	# stat 0, beam 1, time 3
	fxcpmadd 18,29,2,18	# stat 0, beam 2, time 3
	stfpsx	 20,28,8	# beam 2, time 2
	fxcxnpma 21,28,15,21	# stat 2, beam 3, time 2
	lfpsux	 20,30,0	# beam 4, time 3
	fxcpmadd 22,28,16,22	# stat 2, beam 4, time 2
	fxcxnpma 23,27,11,23	# stat 1, beam 5, time 2
	lfpsux	 27,24,0	# stat 1, time 0

	fxcxnpma 24,30,6,24	# stat 1, beam 0, time 3
	dcbt	 24,9
	fxcpmadd 25,30,7,25	# stat 1, beam 1, time 3
	fxcxnpma 18,29,2,18	# stat 0, beam 2, time 3
	fxcpmadd 19,29,3,19	# stat 0, beam 3, time 3
	stfpsx	 21,29,8	# beam 3, time 2
	fxcxnpma 22,28,16,22	# stat 2, beam 4, time 2
	lfpsux	 21,31,0	# beam 5, time 3
	fxcpmadd 23,28,17,23	# stat 2, beam 5, time 2

	fxcpmadd 24,31,12,24	# stat 2, beam 0, time 3
	fxcxnpma 25,30,7,25	# stat 1, beam 1, time 3
	fxcpmadd 18,30,8,18	# stat 1, beam 2, time 3
	fxcxnpma 19,29,3,19	# stat 0, beam 3, time 3
	fxcpmadd 20,29,4,20	# stat 0, beam 4, time 3
	stfpsx	 22,30,8	# beam 4, time 2
	fxcxnpma 23,28,17,23	# stat 2, beam 5, time 2
	lfpsux	 22,26,0	# beam 0, time 0

	# phase 1

	fxcxnpma 24,31,12,24	# stat 2, beam 0, time 3
	lfpsux	 28,25,0	# stat 2, time 0
	fxcpmadd 25,31,13,25	# stat 2, beam 1, time 3
	dcbt	 25,9
	fxcxnpma 18,30,8,18	# stat 1, beam 2, time 3
	fxcpmadd 19,30,9,19	# stat 1, beam 3, time 3
	fxcxnpma 20,29,4,20	# stat 0, beam 4, time 3
	stfpsx	 23,31,8	# beam 5, time 2
	fxcpmadd 21,29,5,21	# stat 0, beam 5, time 3
	lfpsux	 23,27,0	# beam 1, time 0

	fxcpmadd 22,26,0,22	# stat 0, beam 0, time 0
	stfpsx	 24,26,8	# beam 0, time 3
	fxcxnpma 25,31,13,25	# stat 2, beam 1, time 3
	lfpsux	 24,28,0	# beam 2, time 0
	fxcpmadd 18,31,14,18	# stat 2, beam 2, time 3
	fxcxnpma 19,30,9,19	# stat 1, beam 3, time 3
	dcbt	 26,9
	fxcpmadd 20,30,10,20	# stat 1, beam 4, time 3
	fxcxnpma 21,29,5,21	# stat 0, beam 5, time 3
	lfpsux	 29,23,0	# stat 0, time 1

	fxcxnpma 22,26,0,22	# stat 0, beam 0, time 0
	fxcpmadd 23,26,1,23	# stat 0, beam 1, time 0
	stfpsx	 25,27,8	# beam 1, time 3
	fxcxnpma 18,31,14,18	# stat 2, beam 2, time 3
	lfpsux	 25,29,0	# beam 3, time 0
	fxcpmadd 19,31,15,19	# stat 2, beam 3, time 3
	fxcxnpma 20,30,10,20	# stat 1, beam 4, time 3
	dcbt	 27,9
	fxcpmadd 21,30,11,21	# stat 1, beam 5, time 3

	fxcpmadd 22,27,6,22	# stat 1, beam 0, time 0
	fxcxnpma 23,26,1,23	# stat 0, beam 1, time 0
	fxcpmadd 24,26,2,24	# stat 0, beam 2, time 0
	stfpsx	 18,28,8	# beam 2, time 3
	fxcxnpma 19,31,15,19	# stat 2, beam 3, time 3
	lfpsux	 18,30,0	# beam 4, time 0
	fxcpmadd 20,31,16,20	# stat 2, beam 4, time 3
	fxcxnpma 21,30,11,21	# stat 1, beam 5, time 3
	lfpsux	 30,24,0	# stat 1, time 1

	fxcxnpma 22,27,6,22	# stat 1, beam 0, time 0
	fxcpmadd 23,27,7,23	# stat 1, beam 1, time 0
	fxcxnpma 24,26,2,24	# stat 0, beam 2, time 0
	fxcpmadd 25,26,3,25	# stat 0, beam 3, time 0
	stfpsx	 19,29,8	# beam 3, time 3
	fxcxnpma 20,31,16,20	# stat 2, beam 4, time 3
	lfpsux	 19,31,0	# beam 5, time 0
	fxcpmadd 21,31,17,21	# stat 2, beam 5, time 3

	fxcpmadd 22,28,12,22	# stat 2, beam 0, time 0
	dcbt	 28,9
	fxcxnpma 23,27,7,23	# stat 1, beam 1, time 0
	fxcpmadd 24,27,8,24	# stat 1, beam 2, time 0
	fxcxnpma 25,26,3,25	# stat 0, beam 3, time 0
	fxcpmadd 18,26,4,18	# stat 0, beam 4, time 0
	stfpsx	 20,30,8	# beam 4, time 3
	fxcxnpma 21,31,17,21	# stat 2, beam 5, time 3
	lfpsux	 20,26,0	# beam 0, time 1

	# phase 2

	fxcxnpma 22,28,12,22	# stat 2, beam 0, time 0
	lfpsux	 31,25,0	# stat 2, time 1
	fxcpmadd 23,28,13,23	# stat 2, beam 1, time 0
	fxcxnpma 24,27,8,24	# stat 1, beam 2, time 0
	fxcpmadd 25,27,9,25	# stat 1, beam 3, time 0
	fxcxnpma 18,26,4,18	# stat 0, beam 4, time 0
	stfpsx	 21,31,8	# beam 5, time 3
	fxcpmadd 19,26,5,19	# stat 0, beam 5, time 0
	lfpsux	 21,27,0	# beam 1, time 1

	fxcpmadd 20,29,0,20	# stat 0, beam 0, time 1
	stfpsx	 22,26,8	# beam 0, time 0
	fxcxnpma 23,28,13,23	# stat 2, beam 1, time 0
	lfpsux	 22,28,0	# beam 2, time 1
	fxcpmadd 24,28,14,24	# stat 2, beam 2, time 0
	fxcxnpma 25,27,9,25	# stat 1, beam 3, time 0
	dcbt	 29,9
	fxcpmadd 18,27,10,18	# stat 1, beam 4, time 0
	fxcxnpma 19,26,5,19	# stat 0, beam 5, time 0
	lfpsux	 26,23,0	# stat 0, time 2

	fxcxnpma 20,29,0,20	# stat 0, beam 0, time 1
	fxcpmadd 21,29,1,21	# stat 0, beam 1, time 1
	stfpsx	 23,27,8	# beam 1, time 0
	fxcxnpma 24,28,14,24	# stat 2, beam 2, time 0
	lfpsux	 23,29,0	# beam 3, time 1
	fxcpmadd 25,28,15,25	# stat 2, beam 3, time 0
	fxcxnpma 18,27,10,18	# stat 1, beam 4, time 0
	dcbt	 30,9
	fxcpmadd 19,27,11,19	# stat 1, beam 5, time 0

	fxcpmadd 20,30,6,20	# stat 1, beam 0, time 1
	fxcxnpma 21,29,1,21	# stat 0, beam 1, time 1
	fxcpmadd 22,29,2,22	# stat 0, beam 2, time 1
	stfpsx	 24,28,8	# beam 2, time 0
	fxcxnpma 25,28,15,25	# stat 2, beam 3, time 0
	lfpsux	 24,30,0	# beam 4, time 1
	fxcpmadd 18,28,16,18	# stat 2, beam 4, time 0
	fxcxnpma 19,27,11,19	# stat 1, beam 5, time 0
	lfpsux	 27,24,0	# stat 1, time 2

	fxcxnpma 20,30,6,20	# stat 1, beam 0, time 1
	fxcpmadd 21,30,7,21	# stat 1, beam 1, time 1
	fxcxnpma 22,29,2,22	# stat 0, beam 2, time 1
	fxcpmadd 23,29,3,23	# stat 0, beam 3, time 1
	stfpsx	 25,29,8	# beam 3, time 0
	fxcxnpma 18,28,16,18	# stat 2, beam 4, time 0
	lfpsux	 25,31,0	# beam 5, time 1
	fxcpmadd 19,28,17,19	# stat 2, beam 5, time 0

	fxcpmadd 20,31,12,20	# stat 2, beam 0, time 1
	dcbt	 31,9
	fxcxnpma 21,30,7,21	# stat 1, beam 1, time 1
	fxcpmadd 22,30,8,22	# stat 1, beam 2, time 1
	fxcxnpma 23,29,3,23	# stat 0, beam 3, time 1
	fxcpmadd 24,29,4,24	# stat 0, beam 4, time 1
	stfpsx	 18,30,8	# beam 4, time 0
	fxcxnpma 19,28,17,19	# stat 2, beam 5, time 0
	lfpsux	 18,26,0	# beam 0, time 2

	# phase 3

	fxcxnpma 20,31,12,20	# stat 2, beam 0, time 1
	lfpsux	 28,25,0	# stat 2, time 2
	fxcpmadd 21,31,13,21	# stat 2, beam 1, time 1
	fxcxnpma 22,30,8,22	# stat 1, beam 2, time 1
	fxcpmadd 23,30,9,23	# stat 1, beam 3, time 1
	fxcxnpma 24,29,4,24	# stat 0, beam 4, time 1
	stfpsx	 19,31,8	# beam 5, time 0
	fxcpmadd 25,29,5,25	# stat 0, beam 5, time 1
	lfpsux	 19,27,0	# beam 1, time 2

	fxcpmadd 18,26,0,18	# stat 0, beam 0, time 2
	stfpsx	 20,26,8	# beam 0, time 1
	fxcxnpma 21,31,13,21	# stat 2, beam 1, time 1
	lfpsux	 20,28,0	# beam 2, time 2
	fxcpmadd 22,31,14,22	# stat 2, beam 2, time 1
	fxcxnpma 23,30,9,23	# stat 1, beam 3, time 1
	fxcpmadd 24,30,10,24	# stat 1, beam 4, time 1
	fxcxnpma 25,29,5,25	# stat 0, beam 5, time 1
	lfpsux	 29,23,0	# stat 0, time 3

	fxcxnpma 18,26,0,18	# stat 0, beam 0, time 2
	fxcpmadd 19,26,1,19	# stat 0, beam 1, time 2
	stfpsx	 21,27,8	# beam 1, time 1
	fxcxnpma 22,31,14,22	# stat 2, beam 2, time 1
	lfpsux	 21,29,0	# beam 3, time 2
	fxcpmadd 23,31,15,23	# stat 2, beam 3, time 1
	fxcxnpma 24,30,10,24	# stat 1, beam 4, time 1
	fxcpmadd 25,30,11,25	# stat 1, beam 5, time 1

	fxcpmadd 18,27,6,18	# stat 1, beam 0, time 2
	fxcxnpma 19,26,1,19	# stat 0, beam 1, time 2
	fxcpmadd 20,26,2,20	# stat 0, beam 2, time 2
	stfpsx	 22,28,8	# beam 2, time 1
	fxcxnpma 23,31,15,23	# stat 2, beam 3, time 1
	lfpsux	 22,30,0	# beam 4, time 2
	fxcpmadd 24,31,16,24	# stat 2, beam 4, time 1
	fxcxnpma 25,30,11,25	# stat 1, beam 5, time 1
	lfpsux	 30,24,0	# stat 1, time 3

	fxcxnpma 18,27,6,18	# stat 1, beam 0, time 2
	fxcpmadd 19,27,7,19	# stat 1, beam 1, time 2
	fxcxnpma 20,26,2,20	# stat 0, beam 2, time 2
	fxcpmadd 21,26,3,21	# stat 0, beam 3, time 2
	stfpsx	 23,29,8	# beam 3, time 1
	fxcxnpma 24,31,16,24	# stat 2, beam 4, time 1
	lfpsux	 23,31,0	# beam 5, time 2
	fxcpmadd 25,31,17,25	# stat 2, beam 5, time 1

	fxcpmadd 18,28,12,18	# stat 2, beam 0, time 2
	fxcxnpma 19,27,7,19	# stat 1, beam 1, time 2
	fxcpmadd 20,27,8,20	# stat 1, beam 2, time 2
	fxcxnpma 21,26,3,21	# stat 0, beam 3, time 2
	fxcpmadd 22,26,4,22	# stat 0, beam 4, time 2
	stfpsx	 24,30,8	# beam 4, time 1
	fxcxnpma 25,31,17,25	# stat 2, beam 5, time 1
	lfpsux	 24,26,0	# beam 0, time 3

	bdnz	 0b

	# epilogue
	# phase 0

	fxcxnpma 18,28,12,18	# stat 2, beam 0, time 2
	lfpsux	 31,25,0	# stat 2, time 3
	fxcpmadd 19,28,13,19	# stat 2, beam 1, time 2
	fxcxnpma 20,27,8,20	# stat 1, beam 2, time 2
	fxcpmadd 21,27,9,21	# stat 1, beam 3, time 2
	fxcxnpma 22,26,4,22	# stat 0, beam 4, time 2
	stfpsx	 25,31,8	# beam 5, time 1
	fxcpmadd 23,26,5,23	# stat 0, beam 5, time 2
	lfpsux	 25,27,0	# beam 1, time 3

	fxcpmadd 24,29,0,24	# stat 0, beam 0, time 3
	stfpsx	 18,26,8	# beam 0, time 2
	fxcxnpma 19,28,13,19	# stat 2, beam 1, time 2
	lfpsux	 18,28,0	# beam 2, time 3
	fxcpmadd 20,28,14,20	# stat 2, beam 2, time 2
	fxcxnpma 21,27,9,21	# stat 1, beam 3, time 2
	fxcpmadd 22,27,10,22	# stat 1, beam 4, time 2
	fxcxnpma 23,26,5,23	# stat 0, beam 5, time 2

	fxcxnpma 24,29,0,24	# stat 0, beam 0, time 3
	fxcpmadd 25,29,1,25	# stat 0, beam 1, time 3
	stfpsx	 19,27,8	# beam 1, time 2
	fxcxnpma 20,28,14,20	# stat 2, beam 2, time 2
	lfpsux	 19,29,0	# beam 3, time 3
	fxcpmadd 21,28,15,21	# stat 2, beam 3, time 2
	fxcxnpma 22,27,10,22	# stat 1, beam 4, time 2
	fxcpmadd 23,27,11,23	# stat 1, beam 5, time 2

	fxcpmadd 24,30,6,24	# stat 1, beam 0, time 3
	fxcxnpma 25,29,1,25	# stat 0, beam 1, time 3
	fxcpmadd 18,29,2,18	# stat 0, beam 2, time 3
	stfpsx	 20,28,8	# beam 2, time 2
	fxcxnpma 21,28,15,21	# stat 2, beam 3, time 2
	lfpsux	 20,30,0	# beam 4, time 3
	fxcpmadd 22,28,16,22	# stat 2, beam 4, time 2
	fxcxnpma 23,27,11,23	# stat 1, beam 5, time 2

	fxcxnpma 24,30,6,24	# stat 1, beam 0, time 3
	fxcpmadd 25,30,7,25	# stat 1, beam 1, time 3
	fxcxnpma 18,29,2,18	# stat 0, beam 2, time 3
	fxcpmadd 19,29,3,19	# stat 0, beam 3, time 3
	stfpsx	 21,29,8	# beam 3, time 2
	fxcxnpma 22,28,16,22	# stat 2, beam 4, time 2
	lfpsux	 21,31,0	# beam 5, time 3
	fxcpmadd 23,28,17,23	# stat 2, beam 5, time 2

	fxcpmadd 24,31,12,24	# stat 2, beam 0, time 3
	fxcxnpma 25,30,7,25	# stat 1, beam 1, time 3
	fxcpmadd 18,30,8,18	# stat 1, beam 2, time 3
	fxcxnpma 19,29,3,19	# stat 0, beam 3, time 3
	fxcpmadd 20,29,4,20	# stat 0, beam 4, time 3
	stfpsx	 22,30,8	# beam 4, time 2
	fxcxnpma 23,28,17,23	# stat 2, beam 5, time 2

	# phase 1

	fxcxnpma 24,31,12,24	# stat 2, beam 0, time 3
	fxcpmadd 25,31,13,25	# stat 2, beam 1, time 3
	fxcxnpma 18,30,8,18	# stat 1, beam 2, time 3
	fxcpmadd 19,30,9,19	# stat 1, beam 3, time 3
	fxcxnpma 20,29,4,20	# stat 0, beam 4, time 3
	stfpsx	 23,31,8	# beam 5, time 2
	fxcpmadd 21,29,5,21	# stat 0, beam 5, time 3

	stfpsx	 24,0,26	# beam 0, time 3
	fxcxnpma 25,31,13,25	# stat 2, beam 1, time 3
	fxcpmadd 18,31,14,18	# stat 2, beam 2, time 3
	fxcxnpma 19,30,9,19	# stat 1, beam 3, time 3
	fxcpmadd 20,30,10,20	# stat 1, beam 4, time 3
	fxcxnpma 21,29,5,21	# stat 0, beam 5, time 3

	stfpsx	 25,0,27	# beam 1, time 3
	fxcxnpma 18,31,14,18	# stat 2, beam 2, time 3
	fxcpmadd 19,31,15,19	# stat 2, beam 3, time 3
	fxcxnpma 20,30,10,20	# stat 1, beam 4, time 3
	fxcpmadd 21,30,11,21	# stat 1, beam 5, time 3

	stfpsx	 18,0,28	# beam 2, time 3
	fxcxnpma 19,31,15,19	# stat 2, beam 3, time 3
	fxcpmadd 20,31,16,20	# stat 2, beam 4, time 3
	fxcxnpma 21,30,11,21	# stat 1, beam 5, time 3

	stfpsx	 19,0,29	# beam 3, time 3
	fxcxnpma 20,31,16,20	# stat 2, beam 4, time 3
	fxcpmadd 21,31,17,21	# stat 2, beam 5, time 3

	stfpsx	 20,0,30	# beam 4, time 3
	fxcxnpma 21,31,17,21	# stat 2, beam 5, time 3

	# phase 2

	stfpsx	 21,0,31	# beam 5, time 3
