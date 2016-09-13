#ifndef LOFAR_CNPROC_INVERSE_PPF_H
#define LOFAR_CNPROC_INVERSE_PPF_H

#if 0 || !defined HAVE_BGP
#define INVERSE_PPF_C_IMPLEMENTATION
#endif

/*
 The subbandList is specified in the Parset file (and class), exactly in the format we need:
 #subbands x the index van de FFT bin. So, 248 x [0..511]

 The station PPF first does the FIR filtering, next an FFT.
 The station FFT goes from real to complex: 1024 reals to 1024 complex.
 Of those 1024 results, the lower half is discarded, since they are the complex conjungates of the upper half.
 From the FFTW manual: In many practical applications, the input data in[i] are purely real numbers,
 in which case the DFT output satisfies the Hermitian redundancy: out[i] is the conjugate of out[n-i].
 Next, from the 512 upper values, 248 subbands are selected. I.e. more than half of the frequencies are thrown away.

 For the inverse PPF, we first do an inverse FFT, and next the FIR filter with the inverted constants.
 In memory, we have to keep 1 beam, 248 subbands. The CEP PPF was bypassed, so there are no channels.
 Also, we can assume that each core processes only 1 polarization.
 In total, there can be as many as 50 beams and 2 polarizations, so we need 100 cores for the processing.

 There are 3 options:

 - complex to complex FFT

 - complex to real FFT *** This is what this code currently uses. ***
 * Destroys the input data.
 * Input must be in "half complex" format.

 This consists of the non-redundant half of the complex output for a 1d real-input DFT of size n,
 stored as a sequence of n real numbers (double) in the format:

 r0, r1, r2, ..., rn/2, i(n+1)/2-1, ..., i2, i1

 Here, rk is the real part of the kth output, and ik is the imaginary
 part. (Division by 2 is rounded down.) For a halfcomplex array hc[n],
 the kth component thus has its real part in hc[k] and its imaginary
 part in hc[n-k], with the exception of k == 0 or n/2 (the latter only
 if n is even)—in these two cases, the imaginary part is zero due to
 symmetries of the real-input DFT, and is not stored. Thus, the r2hc
 transform of n real values is a halfcomplex array of length n, and
 vice versa for hc2r. Aside from the differing format, the output of
 FFTW_R2HC/FFTW_HC2R is otherwise exactly the same as for the
 corresponding 1d r2c/c2r transform (i.e. FFTW_FORWARD/FFTW_BACKWARD
 transforms, respectively). Recall that these transforms are
 unnormalized, so r2hc followed by hc2r will result in the original
 data multiplied by n. Furthermore, like the c2r transform, an
 out-of-place hc2r transform will destroy its input array.

 - complex to real FFT, multidimensional version with N=1
 * Also destroys input data
 * normal input format

 TODO: Which option gives the best performance?

 A BG/P compute node has 2 GB of memory, which is shared between 4 cores.
 So, we have only 512 MB per core.
 */

#include <lofar_config.h>

#include <Common/lofar_complex.h>
#include <Common/Timer.h>
#include <Interface/TransposedBeamFormedData.h>
#include <Interface/InverseFilteredData.h>
#include <Interface/Align.h>
#include <Interface/AlignedStdAllocator.h>
#include <vector>
#include <FilterBank.h>
#include <FIR.h>

// #undef HAVE_FFTW3

// On the BG/P, FFT2 uses the double floating point units, FFT3 works, but only uses one.
#if defined HAVE_FFTW3
#include <fftw3.h>
#elif defined HAVE_FFTW2
#include <fftw.h>
#include <rfftw.h>
#else
#error Should have FFTW3 or FFTW2 installed
#endif

#if defined HAVE_FFTW3
#define fftw_real(x)     ((x)[0])
#define fftw_imag(x)     ((x)[1])
#elif defined HAVE_FFTW2
#define fftw_real(x)     (c_re(x))
#define fftw_imag(x)     (c_im(x))
#endif

#define ON_STATION_FILTER_SIZE 1024
#define ON_STATION_FILTER_TAPS 16


#define USE_FFT_HALF_COMPLEX 1


namespace LOFAR {
namespace RTCP {

class InversePPF: boost::noncopyable {
public:
  InversePPF(std::vector<unsigned>& subbandList, unsigned nrSamplesPerIntegration, bool verbose);
  ~InversePPF();

  void performInversePPF(const TransposedBeamFormedData& transposedBeamFormedData, InverseFilteredData& inverseFilteredData);

private:

  void initFFT();
  void destroyFFT();
  void createFFTInput(const TransposedBeamFormedData& transposedBeamFormedData, unsigned time);
  void performInverseFFT();
  void performFiltering(InverseFilteredData& invertedFilteredData, unsigned time);
  void performInversePPFTimeStep(const TransposedBeamFormedData& transposedBeamFormedData, InverseFilteredData& invertedFilteredData, unsigned time);

  FilterBank itsFilterBank;
  std::vector<FIR<float> > itsFIRs;

#if defined HAVE_FFTW3
  fftwf_plan itsPlan;
#elif defined HAVE_FFTW2
  fftw_plan itsPlan;
#endif

  float* itsFftInData;
  float* itsFftOutData;

  std::vector<unsigned>& itsSubbandList;
  unsigned itsNrSubbands;
  unsigned itsNrSamplesPerIntegration;

  bool itsVerbose;
  bool itsSubbandsAreContiguous;
};

} // namespace RTCP
} // namespace LOFAR

#endif // LOFAR_CNPROC_INVERSE_PPF_H
