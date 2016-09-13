#if defined HAVE_BGP

# bool LOFAR::RTCP::containsOnlyZeros(std::complex<short> *samples, unsigned nrSamples);

.global	_ZN5LOFAR4RTCP17containsOnlyZerosISt7complexIsEEEbPKT_j
_ZN5LOFAR4RTCP17containsOnlyZerosISt7complexIsEEEbPKT_j:
	mtctr	4
	li	4,4
	subi	3,3,4

0:	lwzux	0,3,4
	lwzux	5,3,4
	or.	0,0,5
	bdnzt	2,0b

	mfcr	3
	rlwinm	3,3,3,31,31
	blr


# bool LOFAR::RTCP::containsOnlyZeros(std::complex<signed char> *samples, unsigned nrSamples);

.global	_ZN5LOFAR4RTCP17containsOnlyZerosISt7complexIaEEEbPKT_j
_ZN5LOFAR4RTCP17containsOnlyZerosISt7complexIaEEEbPKT_j:
	mtctr	4
	li	4,4
	subi	3,3,4

0:	lwzux	0,3,4
	cmpwi	0,0
	bdnzt	2,0b

	mfcr	3
	rlwinm	3,3,3,31,31
	blr


# bool LOFAR::RTCP::containsOnlyZeros(LOFAR::TYPES::i4complex *samples, unsigned nrSamples);

.global	_ZN5LOFAR4RTCP17containsOnlyZerosINS_5TYPES9i4complexEEEbPKT_j
_ZN5LOFAR4RTCP17containsOnlyZerosINS_5TYPES9i4complexEEEbPKT_j:
	mtctr	4
	li	4,4
	subi	3,3,4

0:	lhzux	0,3,4
	cmpwi	0,0
	bdnzt	2,0b

	mfcr	3
	rlwinm	3,3,3,31,31
	blr

#endif        
