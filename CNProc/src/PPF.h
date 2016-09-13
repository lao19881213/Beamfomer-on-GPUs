#ifndef LOFAR_CNPROC_PPF_H
#define LOFAR_CNPROC_PPF_H

#if 0 || !defined HAVE_BGP
#define PPF_C_IMPLEMENTATION
#endif


#include <BandPass.h>
#include <FilterBank.h>
#include <FIR.h>
#include <Interface/TransposedData.h>
#include <Interface/FilteredData.h>
#include <Interface/SubbandMetaData.h>
#include <Interface/AlignedStdAllocator.h>

#include <boost/multi_array.hpp>
#include <boost/noncopyable.hpp>

#if defined HAVE_FFTW3
#include <fftw3.h>
#elif defined HAVE_FFTW2
#include <fftw.h>
#else
#error Should have FFTW3 or FFTW2 installed
#endif


namespace LOFAR {
namespace RTCP {

template <typename SAMPLE_TYPE> class PPF: boost::noncopyable
{
  public:
    PPF(unsigned nrStations, unsigned nrChannels, unsigned nrSamplesPerIntegration, double channelBandwidth, bool delayCompensation, bool correctBandPass, bool verbose);
    ~PPF();

    void doWork(unsigned stat, double centerFrequency, const SubbandMetaData *, const TransposedData<SAMPLE_TYPE> *, FilteredData *);

#if !defined PPF_C_IMPLEMENTATION
    static void initConstantTable();
#endif

  private:
    void init_fft(), destroy_fft();

#if defined PPF_C_IMPLEMENTATION
    fcomplex phaseShift(unsigned time, unsigned chan, double baseFrequency, double delayAtBegin, double delayAfterEnd) const;
#else
    void     computePhaseShifts(struct phase_shift phaseShifts[/*itsNrSamplesPerIntegration*/], double delayAtBegin, double delayAfterEnd, double baseFrequency) const;
#endif

    void computeFlags(unsigned stat, const SubbandMetaData *metaData, FilteredData *);
    void filter(unsigned stat, double centerFrequency, const SubbandMetaData *, const TransposedData<SAMPLE_TYPE> *, FilteredData *);
    void bypass(unsigned stat, double centerFrequency, const SubbandMetaData *, const TransposedData<SAMPLE_TYPE> *, FilteredData *);

    const unsigned itsNrStations, itsNrSamplesPerIntegration;
    const unsigned itsNrChannels;
    unsigned       itsLogNrChannels;
    const double   itsChannelBandwidth;
    const bool     itsDelayCompensation, itsCorrectBandPass;
    const BandPass itsBandPass;
    FilterBank     itsFilterBank;
    boost::multi_array<FIR<fcomplex>, 3> itsFIRs; //[itsNrStations][NR_POLARIZATIONS][itsNrChannels]

#if defined PPF_C_IMPLEMENTATION
    boost::multi_array<fcomplex, 3> itsFFTinData; //[NR_TAPS - 1 + itsNrSamplesPerIntegration][NR_POLARIZATIONS][itsNrChannels]
#else
    boost::multi_array<fcomplex, 2, AlignedStdAllocator<fcomplex, 32> > itsDelayLines; //[4][itsNrSamplesPerIntegration]
    boost::multi_array<fcomplex, 3, AlignedStdAllocator<fcomplex, 32> > itsFFTinData; //[itsNrSamplesPerIntegration][NR_POLARIZATIONS][itsNrChannels + 4]
    boost::multi_array<fcomplex, 3, AlignedStdAllocator<fcomplex, 32> > itsFFToutData; //[2][NR_POLARIZATIONS][itsNrChannels]
#endif

#if defined HAVE_FFTW3
    fftwf_plan itsFFTWPlan;
#elif defined HAVE_FFTW2
    fftw_plan  itsFFTWPlan;
#endif
};

} // namespace RTCP
} // namespace LOFAR

#endif
