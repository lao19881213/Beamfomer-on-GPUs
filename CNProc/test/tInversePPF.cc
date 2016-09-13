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
#include <InversePPF.h>

#include <FIR_OriginalStationPPFWeights.h> // defines originalStationPPFWeights array
#include <FIR_InvertedStationPPFWeights.h> // defines invertedStationPPFWeights array

// #undef HAVE_FFTW3

// On the BG/P, FFT2 uses the double floating point units, FFT3 works, but only uses one.
#if defined HAVE_FFTW3
#include <fftw3.h>
//#error using fftw3
#elif defined HAVE_FFTW2
#include <fftw.h>
#include <rfftw.h>
//#error using fftw2
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

using namespace LOFAR;
using namespace LOFAR::RTCP;
using namespace LOFAR::TYPES;

#if defined HAVE_FFTW3
fftwf_plan plan;
#elif defined HAVE_FFTW2
rfftw_plan plan;
#endif

const static unsigned onStationFilterSize = 1024;
const static unsigned nrTaps = 16;

static unsigned nrSubbands = 248;
//static unsigned nrSubbands = 4;
static unsigned nrChannels = 1; // for the NuMoon pipeline, there are no separate channels.
//static unsigned nrSamplesPerIntegration = 768 * 256 / 4; // one quarter of a second
static unsigned nrSamplesPerIntegration = 19648; // roughly 0.1 seconds
//static unsigned nrSamplesPerIntegration = 64;
static double subbandBandwidth = 195312.5;
static double centerFrequency = (nrSamplesPerIntegration / 2) * subbandBandwidth;
static double signalFrequency = centerFrequency - (0.5 * subbandBandwidth);

float originalStationPPFWeightsFloat[1024][16];
float* fftInData;
float* fftOutData;

static void initFFT() {
#if defined HAVE_FFTW3
  fftInData = (float*) fftwf_malloc(onStationFilterSize * sizeof(float));
  fftOutData = (float*) fftwf_malloc(onStationFilterSize * sizeof(float));

  plan = fftwf_plan_r2r_1d(onStationFilterSize, fftInData, fftOutData, FFTW_R2HC, FFTW_ESTIMATE);
#elif defined HAVE_FFTW2
  fftInData = (float*) malloc(onStationFilterSize * sizeof(float));
  fftOutData = (float*) malloc(onStationFilterSize * sizeof(float));

  plan = rfftw_create_plan(onStationFilterSize, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE);
#endif

  if (fftInData == NULL || fftOutData == NULL) {
    cerr << "Out of memory" << endl;
    exit(1);
  }
}

static void destroyFFT() {
#if defined HAVE_FFTW3
  fftwf_free(fftInData);
  fftwf_free(fftOutData);
  fftwf_destroy_plan(plan);
#elif defined HAVE_FFTW2
  free(fftInData);
  free(fftOutData);
  rfftw_destroy_plan(plan);
#endif
}

static void generateInputSignal(InverseFilteredData& originalData) {
  for (unsigned time = 0; time < nrSamplesPerIntegration * onStationFilterSize; time++) {
    double val = sin(time/(768.0*2.0));
    originalData.samples[time] = val;
  }
}

static void performStationFFT(TransposedBeamFormedData& transposedBeamFormedData, std::vector<unsigned>& subbandList, unsigned time) {
#if defined HAVE_FFTW3
  fftwf_execute(plan);
#elif defined HAVE_FFTW2
  rfftw_one(plan, (fftw_real*) fftInData, (fftw_real*) fftOutData);
#endif

  // Put data in the right order, go from half complex to normal format
  for (unsigned subbandIndex = 0; subbandIndex < subbandList.size(); subbandIndex++) {
    unsigned subband = subbandList[subbandIndex];
    fcomplex sample = makefcomplex(fftOutData[subband], fftOutData[onStationFilterSize - subband - 1]);
    transposedBeamFormedData.samples[subband][0 /* channel, but there is only one now */][time] = sample;
  }
}

static void performStationFilter(InverseFilteredData& originalData, std::vector<FIR<float> >& FIRs, unsigned time) {
  for (unsigned minorTime = 0; minorTime < onStationFilterSize; minorTime++) {
    float sample = originalData.samples[time * onStationFilterSize + minorTime];
    float result = FIRs[minorTime].processNextSample(sample);
    fftInData[minorTime] = result;
  }
}

#if 0
static void printData(InverseFilteredData& data) {
  for (unsigned time = 0; time < nrSamplesPerIntegration * onStationFilterSize; time++) {
    float sample = data.samples[time];
    fprintf(stdout, "%20.10lf\n", sample);
  }
}
#endif

#if 0
static void cepFilterTest() {
  // CEP filter test
  FilterBank fb(true, 16, 256, KAISER);
  boost::multi_array<FIR<fcomplex> , 1> firs(boost::extents[16]);

  // Init the FIR filters themselves with the weights of the filterbank.
  for (unsigned chan = 0; chan < nrChannels; chan++) {
    firs[chan].initFilter(&fb, chan);
  }

  cout << "START CEP WEIGHTS" << endl;
  fb.printWeights();
  cout << "END CEP WEIGHTS" << endl;
}
#endif

#if 0
static void fftTest() {
  float* inputData = (float*) malloc(onStationFilterSize * sizeof(float));

  fftwf_plan inversePlan = fftwf_plan_r2r_1d(onStationFilterSize, fftOutData, fftInData, FFTW_HC2R, FFTW_ESTIMATE);

  // generate signal
  for (unsigned time = 0; time < onStationFilterSize; time++) {
    double val = sin(signalFrequency * time / subbandBandwidth);
    fftInData[time] = val;
    inputData[time] = val;
  }

#if 0
//  cout << "START FFT TEST INPUT" << endl;
  for (unsigned time = 0; time < onStationFilterSize; time++) {
    float sample = fftInData[time];
    fprintf(stdout, "%20.10lf\n", sample);
  }
//  cout << "END FFT TEST INPUT" << endl;
#endif

  // simulate some zeros
  for(unsigned int i=248; i<onStationFilterSize; i++) {
    fftInData[i] = 0.0;
    inputData[i] = 0.0;
  }

  fftwf_execute(plan);

#if 0
  // Put data in the right order, go from half complex to normal format
  for (unsigned subband = 0; subband < nrSubbands; subband++) {
    fcomplex sample = makefcomplex(fftOutData[subband], fftOutData[onStationFilterSize - subband - 1]);
    transposedBeamFormedData.samples[subband][0 /* channel */][time] = sample;
  }
#endif

  fftwf_execute(inversePlan);

  float maxError = 0.0f;

  for (unsigned time = 0; time < onStationFilterSize; time++) {
    float error = fabsf(inputData[time] - (fftInData[time]/((float)onStationFilterSize))); // the error
    if(error > maxError) {
      maxError = error;
    }
//    fprintf(stdout, "%20.10lf\n", error);
//    fprintf(stdout, "%20.10lf\n", fftInData[time]);
  }

  cerr << "max error = " << maxError << endl;
  free(inputData);
}
#endif

#if 0
// Do a station filter + inverse filter, but not the FFTs.
static void filterTest(InverseFilteredData& originalData) {
  FilterBank originalStationFilterBank(true, nrTaps, onStationFilterSize, (float*) originalStationPPFWeightsFloat);
  std::vector<FIR<float> > FIRs;
  FIRs.resize(onStationFilterSize); // Init the FIR filters themselves with the weights of the filterbank.
  for (unsigned chan = 0; chan < onStationFilterSize; chan++) {
    FIRs[chan].initFilter(&originalStationFilterBank, chan);
  }
//  cout << "START ORIG STATION WEIGHTS" << endl;
//  originalStationFilterBank.printWeights();
//  cout << "END ORIG STATION WEIGHTS" << endl;

  FilterBank invertedStationFilterBank(true, nrTaps, onStationFilterSize, (float*) invertedStationPPFWeights);

  std::vector<FIR<float> > inverseFIRs;
  inverseFIRs.resize(onStationFilterSize); // Init the FIR filters themselves with the weights of the filterbank.
  for (unsigned chan = 0; chan < onStationFilterSize; chan++) {
    inverseFIRs[chan].initFilter(&invertedStationFilterBank, chan);
  }

//  cout << "START INV STATION WEIGHTS" << endl;
//  invertedStationFilterBank.printWeights();
//  cout << "END INV STATION WEIGHTS" << endl;

  for(unsigned major=0; major<nrSamplesPerIntegration; major++) {
    for(unsigned minor = 0; minor < onStationFilterSize; minor++) {
      float sample = originalData.samples[major * onStationFilterSize + minor];
      float result = FIRs[minor].processNextSample(sample);
      float resultInv = inverseFIRs[minor].processNextSample(result);

      fprintf(stdout, "%20.10lf    %20.10lf    %20.10lf\n", sample, result, resultInv);
    }
  }
}
#endif

int main() {

  NSTimer iPPFTimer("Full inverse PPF", true);

  // copy the integer filter constants into a float array.
  for (unsigned filter = 0; filter < onStationFilterSize; filter++) {
    for (unsigned tap = 0; tap < nrTaps; tap++) {
      originalStationPPFWeightsFloat[filter][tap] = originalStationPPFWeights[filter][tap];
    }
  }

  FilterBank originalStationFilterBank(true, nrTaps, onStationFilterSize, (float*) originalStationPPFWeightsFloat);
  std::vector<FIR<float> > FIRs;
  FIRs.resize(onStationFilterSize); // Init the FIR filters themselves with the weights of the filterbank.
  for (unsigned chan = 0; chan < onStationFilterSize; chan++) {
    FIRs[chan].initFilter(&originalStationFilterBank, chan);
  }

#if 0
  cout << "START ORIG STATION WEIGHTS" << endl;
  originalStationFilterBank.printWeights();
  cout << "END ORIG STATION WEIGHTS" << endl;
#endif

  // The original data has the same data format as the original data, so reuse it here for this test
  InverseFilteredData	   originalData(nrSamplesPerIntegration, onStationFilterSize);
  TransposedBeamFormedData transposedBeamFormedData(nrSubbands, nrChannels, nrSamplesPerIntegration);
  InverseFilteredData	   invertedFilteredData(nrSamplesPerIntegration, onStationFilterSize);
  std::vector<unsigned>	   subbandList(nrSubbands);

  // for now, we just select the first n subbands.
  for (unsigned sb = 0; sb < nrSubbands; sb ++)
    subbandList[sb] = sb;

  InversePPF inversePPF(subbandList, nrSamplesPerIntegration, true);
  initFFT();

//  fftTest();

  cerr << "generating input signal" << endl;

  generateInputSignal(originalData);

//  printData(originalData);

//  filterTest(originalData);
//  exit(0);

  cerr << "simulating station filter" << endl;

  for (unsigned time = 0; time < nrSamplesPerIntegration; time++) {
    performStationFilter(originalData, FIRs, time);
    performStationFFT(transposedBeamFormedData, subbandList, time);
  }

#if 0
  for (unsigned sb = 0; sb < nrSubbands; sb++)
  for (unsigned time = 0; time < nrSamplesPerIntegration; time++) {
    fcomplex sample = transposedBeamFormedData.samples[sb][0][time]; // [sb][chan][time]
    fprintf(stdout, "%20.10lf\n", real(sample));
  }
#endif

  const unsigned nIter = 1;

  cerr << "performing inversePPF " << nIter << " time(s)" << endl;

  for(unsigned i=0; i<nIter; i++) {
    iPPFTimer.start();
    inversePPF.performInversePPF(transposedBeamFormedData, invertedFilteredData);
    iPPFTimer.stop();
  }

  cerr << "inversePPF done" << endl;

  //  cout << "result:" << endl;

//  printData(invertedFilteredData);

  destroyFFT();
  return 0;
}
