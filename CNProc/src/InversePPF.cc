/*
 For comments on how this class works, see InversePPF.h.
*/

//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <InversePPF.h>

using namespace LOFAR;
using namespace LOFAR::RTCP;
using namespace LOFAR::TYPES;

#include <FIR_InvertedStationPPFWeights.h> // defines invertedStationPPFWeights array

static NSTimer firTimer("FIR", true);
static NSTimer fftTimer("FFT", true);
static NSTimer fftInTimer("create FFT input", true);

InversePPF::InversePPF(std::vector<unsigned>& subbandList, unsigned nrSamplesPerIntegration, bool verbose) :
  itsFilterBank(false, ON_STATION_FILTER_TAPS, ON_STATION_FILTER_SIZE, (float*) invertedStationPPFWeights), itsSubbandList(subbandList), itsNrSubbands(itsSubbandList.size()),
      itsNrSamplesPerIntegration(nrSamplesPerIntegration), itsVerbose(verbose) {

  double origInputSize = (itsNrSubbands * itsNrSamplesPerIntegration * sizeof(fcomplex)) / (1024.0 * 1024.0);
  double fftBufSize = (ON_STATION_FILTER_SIZE * sizeof(float)) / (1024.0);
  double outputSize = (ON_STATION_FILTER_SIZE * itsNrSamplesPerIntegration * sizeof(float)) / (1024.0 * 1024.0);

  if (itsVerbose) {
    cerr << "size of original input data: " << origInputSize << " MB" << endl;
    cerr << "size of FFT buffers: " << fftBufSize << " KB" << endl;
    cerr << "size of output: " << outputSize << " MB" << endl;
    cerr << "total memory usage: " << (origInputSize + outputSize) << " MB" << endl;
  }

  // Init the FIR filters themselves with the weights of the filterbank.
  itsFIRs.resize(ON_STATION_FILTER_SIZE);
  for (unsigned chan = 0; chan < ON_STATION_FILTER_SIZE; chan++) {
    itsFIRs[chan].initFilter(&itsFilterBank, chan);
  }

  // See if the selected subbands are a contiguous list. If so, we can index the data more efficiently.
  itsSubbandsAreContiguous = true;
  unsigned prev = itsSubbandList[0];
  for(unsigned i=1; i<itsNrSubbands; i++) {
    unsigned sb = itsSubbandList[i];
    if(sb != prev+1) {
      cout << "EEE" << endl;
      itsSubbandsAreContiguous = false;
      break;
    }
    prev = sb;
  }

  initFFT();
}


InversePPF::~InversePPF() {
  destroyFFT();
}


void InversePPF::initFFT() {
#if defined HAVE_FFTW3
  itsFftInData = (float*) fftwf_malloc(ON_STATION_FILTER_SIZE * sizeof(float));
  itsFftOutData = (float*) fftwf_malloc(ON_STATION_FILTER_SIZE * sizeof(float));

  itsPlan = fftwf_plan_r2r_1d(ON_STATION_FILTER_SIZE, itsFftInData, itsFftOutData, FFTW_HC2R, FFTW_ESTIMATE);

//  itsPlan = fftwf_plan_dft_c2r_1d(ON_STATION_FILTER_SIZE, itsFftInData, itsFftOutData, FFTW_ESTIMATE);

#elif defined HAVE_FFTW2
  itsFftInData = (float*) malloc(ON_STATION_FILTER_SIZE * sizeof(float));
  itsFftOutData = (float*) malloc(ON_STATION_FILTER_SIZE * sizeof(float));

  itsPlan = rfftw_create_plan(ON_STATION_FILTER_SIZE, FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE);
#endif

  if (itsFftInData == NULL || itsFftOutData == NULL) {
    cerr << "Out of memory" << endl;
    exit(1);
  }
}

void InversePPF::destroyFFT() {
#if defined HAVE_FFTW3
  fftwf_destroy_plan(itsPlan);
  fftwf_free(itsFftInData);
  fftwf_free(itsFftOutData);
#elif defined HAVE_FFTW2
  rfftw_destroy_plan(itsPlan);
  free(itsFftInData);
  free(itsFftOutData);
#endif
}


// in hc format, we store n/2+1 reals and n/2-1 imags
// Goes from tansposedBeamFormedData to itsFftInData.
// Fill input buffer, using "half complex" format.
void InversePPF::createFFTInput(const TransposedBeamFormedData& transposedBeamFormedData, unsigned time) {
  fftInTimer.start();

  if(itsSubbandsAreContiguous) {
    const unsigned start = itsSubbandList[0]; // inclusive
    const unsigned end = start + itsNrSubbands-1; // inclusive

    if(start > 0) {
      memset(itsFftInData, 0, start * sizeof(float)); // subbands before start, real
      memset(itsFftInData + ON_STATION_FILTER_SIZE-start, 0, (start-1) * sizeof(float));  // subbands before start, imag
    }

    if(end < ON_STATION_FILTER_SIZE/2) {
      memset(itsFftInData + end, 0, (ON_STATION_FILTER_SIZE/2 - end) * sizeof(float)); // subbands after end, real
      memset(itsFftInData + ((ON_STATION_FILTER_SIZE/2) + 1), 0, ((ON_STATION_FILTER_SIZE/2 - end)-1) * sizeof(float));
    }

    if(start == 0) { // special case, the half complex format doesn store the imag part of the 0th element (it is always 0)
      fcomplex sample = transposedBeamFormedData.samples[0][0 /* channel, but there only is 1 now */][time];
      itsFftInData[0] = real(sample);
      for (unsigned sb = 1; sb < itsNrSubbands; sb++) {
	fcomplex sample = transposedBeamFormedData.samples[sb][0 /* channel, but there only is 1 now */][time];
	itsFftInData[sb] = real(sample);
	itsFftInData[ON_STATION_FILTER_SIZE - sb] = imag(sample);
      }
    } else {
      for (unsigned i = 0; i < itsNrSubbands; i++) {
	unsigned sb = start + i;
	fcomplex sample = transposedBeamFormedData.samples[sb][0 /* channel, but there only is 1 now */][time];
	itsFftInData[i] = real(sample);
	itsFftInData[ON_STATION_FILTER_SIZE - sb] = imag(sample);
      }
    }
  } else {
    // First set the unselected subbands to zero.
    // We have to do this every time, since the input is destroyed by the FFT.
    // However, the time this takes is very small compared to the time to fill in the real data below.
    memset(itsFftInData, 0, ON_STATION_FILTER_SIZE * sizeof(float));

    // There can be gaps in the subband list.
    // Copy the samples from the different subbands to their correct places.
    for (unsigned i = 0; i < itsNrSubbands; i++) {
      unsigned sb = itsSubbandList[i];
      fcomplex sample = transposedBeamFormedData.samples[sb][0 /* channel, but there only is 1 now */][time];
      
      itsFftInData[sb] = real(sample);
      if(sb != 0) {
	itsFftInData[ON_STATION_FILTER_SIZE - sb] = imag(sample);
      }
    }
  }

  fftInTimer.stop();
}


// This method writes the result to itsFFtOutData
void InversePPF::performInverseFFT() {
  fftTimer.start();

#if defined HAVE_FFTW3
  // in and out are not the same buffer, and the input is destroyed by the fftw call.
  fftwf_execute(itsPlan);
#elif defined HAVE_FFTW2
  // Do the inverse FFT. NB: this call destoys the input data.
  rfftw_one(itsPlan, (fftw_real*) itsFftInData, (fftw_real*) itsFftOutData);
#endif

  fftTimer.stop();
}


// Reads itsFftOutData, writes to invertedFilteredData.
void InversePPF::performFiltering(InverseFilteredData& invertedFilteredData, unsigned time) {
  firTimer.start();

  unsigned index = time * ON_STATION_FILTER_SIZE;
  for (unsigned minorTime = 0; minorTime < ON_STATION_FILTER_SIZE; minorTime++) {
    const float sample = itsFftOutData[minorTime];
    const float result = itsFIRs[minorTime].processNextSample(sample);
    invertedFilteredData.samples[index++] = result;
  }

  firTimer.stop();
}


void InversePPF::performInversePPFTimeStep(const TransposedBeamFormedData& transposedBeamFormedData, InverseFilteredData& invertedFilteredData, unsigned time) {
  createFFTInput(transposedBeamFormedData, time);
  performInverseFFT();
  performFiltering(invertedFilteredData, time);
}


void InversePPF::performInversePPF(const TransposedBeamFormedData& transposedBeamFormedData, InverseFilteredData& invertedFilteredData) {
  for (unsigned time = 0; time < itsNrSamplesPerIntegration; time++) {
    performInversePPFTimeStep(transposedBeamFormedData, invertedFilteredData, time);
  }
}
