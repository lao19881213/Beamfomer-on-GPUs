//# Always #include <lofar_config.h> first!
#include <lofar_config.h>

#include <FilterBank.h>
#include <cstring>

#include <math.h>
#include <iostream>
#include <cstring>

#include <Common/LofarLogger.h>
#include <Interface/Align.h>
#include <Interface/Exceptions.h>

#if defined HAVE_FFTW3
#include <fftw3.h>
#define fftw_real(x)     ((x)[0])
#define fftw_imag(x)     ((x)[1])
#elif defined HAVE_FFTW2
#include <fftw.h>
#define fftw_real(x)     (c_re(x))
#define fftw_imag(x)     (c_im(x))
#else
#error Should have FFTW3 or FFTW2 installed
#endif

namespace LOFAR {
namespace RTCP {

#if USE_ORIGINAL_FILTER
#include <FIR_OriginalCepPPFWeights.h>
#endif

// For documentation on this class, see the header file.

FilterBank::FilterBank(bool verbose, unsigned taps, unsigned channels, WindowType windowType)
:
  itsWindowType(windowType), itsNrTaps(taps), itsNrChannels(channels), itsVerbose(verbose), itsNegated(false)
{
  generate_filter();
}


FilterBank::FilterBank(bool verbose, unsigned taps, unsigned channels, float newWeights[])
:
  itsWindowType(PREDEFINED_FILTER), itsNrTaps(taps), itsNrChannels(channels), itsVerbose(verbose), itsNegated(false)
{
  weights.resize(boost::extents[itsNrChannels][itsNrTaps]);
  memcpy(weights.origin(), newWeights, (itsNrChannels * itsNrTaps) * sizeof(float));
}


// hamming window function
void FilterBank::hamming(unsigned n, double d[])
{
  if (n == 1) {
    d[0] = 1.0;
    return;
  }

  unsigned m = n - 1;

  for (unsigned i = 0; i < n; i++) {
    d[i] = 0.54 - 0.46 * cos((2.0 * M_PI * i) / m);
  }
}


// blackman window function
void FilterBank::blackman(unsigned n, double d[])
{
  if (n == 1) {
    d[0] = 1.0;
    return;
  }

  unsigned m = n - 1;

  for (unsigned i = 0; i < n; i++) {
    double k = i / m;
    d[i] = 0.42 - 0.5 * cos(2.0 * M_PI * k) + 0.08 * cos(4.0 * M_PI * k);
  }
}


// Guassian window function
void FilterBank::gaussian(int n, double a, double d[])
{
  int index = 0;

  for (int i = -(n - 1); i <= n - 1; i += 2) {
    d[index++] = exp(-0.5 * pow((a / n * i), 2));
  }
}


// Compute the modified Bessel function I_0(x) for any real x.
// This method was taken from the ROOT package, See http://root.cern.ch/root.
// It was released undet the GNU LESSER GENERAL PUBLIC LICENSE Version 2.1
double FilterBank::besselI0(double x)
{
  // Parameters of the polynomial approximation
  const double p1 = 1.0, p2 = 3.5156229, p3 = 3.0899424, p4 = 1.2067492, p5 = 0.2659732, p6 = 3.60768e-2, p7 = 4.5813e-3;

  const double q1 = 0.39894228, q2 = 1.328592e-2, q3 = 2.25319e-3, q4 = -1.57565e-3, q5 = 9.16281e-3, q6 = -2.057706e-2, q7 = 2.635537e-2, q8 = -1.647633e-2,
      q9 = 3.92377e-3;

  const double k1 = 3.75;
  double ax = fabs(x);

  double y = 0, result = 0;

  if (ax < k1) {
    double xx = x / k1;
    y = xx * xx;
    result = p1 + y * (p2 + y * (p3 + y * (p4 + y * (p5 + y * (p6 + y * p7)))));
  } else {
    y = k1 / ax;
    result = (exp(ax) / sqrt(ax)) * (q1 + y * (q2 + y * (q3 + y * (q4 + y * (q5 + y * (q6 + y * (q7 + y * (q8 + y * q9))))))));
  }

  return result;
}


// Kaiser window function
void FilterBank::kaiser(int n, double beta, double d[])
{
  if (n == 1) {
    d[0] = 1.0;
    return;
  }

  int m = n - 1;

  for (int i = 0; i < n; i++) {
    double k = 2.0 * beta / m * sqrt(i * (m - i));
    d[i] = besselI0(k) / besselI0(beta);
  }
}


// One-dimensional interpolation. Interpolate Y, defined at the points X, 
// at N evenly spaced points between 0 and 1. The sample points X must be strictly monotonic
void FilterBank::interpolate(const double x[], const double y[], unsigned xlen, unsigned n, double result[])
{
  unsigned nextX = 0;
  unsigned index = 0;

  for (double interpolatedX = 0; interpolatedX <= 1.0; interpolatedX += 1.0 / (n - 1), index++) {
    while (x[nextX] <= interpolatedX && nextX < xlen - 1)
      nextX++;

    if (nextX == 0) {
      LOG_ERROR("ERROR in FilterBank::interpolate");
    }

    double prevXVal = x[nextX - 1];
    double nextXVal = x[nextX];
    double prevYVal = y[nextX - 1];
    double nextYVal = y[nextX];

    double rc = (nextYVal - prevYVal) / (nextXVal - prevXVal);

    double newVal = prevYVal + (interpolatedX - prevXVal) * rc;
    result[index] = newVal;
  }
}


// Compute the filter, similar to Octave's fir2(n, f, m, grid_n, ramp_n, window);
// Window and result must be of size n+1.
// grid_n: length of ideal frequency response function
// ramp_n: transition width for jumps in filter response
// defaults to grid_n/20; a wider ramp gives wider transitions
// but has better stopband characteristics.
void FilterBank::generate_fir_filter(unsigned n, double w, const double window[], double result[])
{
  // make sure grid is big enough for the window
  // the grid must be at least (n+1)/2
  // for all filters where the order is a power of two minus 1, grid_n = n+1;
  unsigned grid_n = nextPowerOfTwo(n + 1);

  unsigned ramp_n = 2; // grid_n/20;

  // Apply ramps to discontinuities
  // this is a low pass filter
  // maybe we can omit the "w, 0" point?
  // I did observe a small difference
  double f[] = { 0.0, w - ramp_n / grid_n / 2.0, w, w + ramp_n / grid_n / 2.0, 1.0 };
  double m[] = { 1.0, 1.0, 0.0, 0.0, 0.0 };

  // grid is a 1-D array with grid_n+1 points. Values are 1 in filter passband, 0 otherwise
  std::vector<double> grid(grid_n + 1);

  // interpolate between grid points
  interpolate(f, m, 5 /* length of f and m arrays */, grid_n + 1, &grid[0]);

#if 0
  std::stringstream logStr;
  logStr << "interpolated = [";
  for(unsigned i=0; i<grid_n+1; i++) {
    logStr << grid[i];
    if(i != grid_n+1-1) logStr << ", ";
  }
  logStr << "];";
  LOG_DEBUG(logStr.str());
#endif

  // the grid we do an ifft on is:
  // grid appended with grid_n*2 zeros
  // appended with original grid values from indices grid_n..2, i.e., the values in reverse order
  // (note, arrays start at 1 in octave!)
  // the input for the ifft is of size 4*grid_n
  // input = [grid ; zeros(grid_n*2,1) ;grid(grid_n:-1:2)];

#if defined HAVE_FFTW3
  fftwf_complex* cinput = (fftwf_complex*) fftwf_malloc(grid_n * 4 * sizeof(fftwf_complex));
  fftwf_complex* coutput = (fftwf_complex*) fftwf_malloc(grid_n * 4 * sizeof(fftwf_complex));
#elif defined HAVE_FFTW2
  fftw_complex* cinput = (fftw_complex*) fftw_malloc(grid_n*4*sizeof(fftw_complex));
  fftw_complex* coutput = (fftw_complex*) fftw_malloc(grid_n*4*sizeof(fftw_complex));
#endif

  if (cinput == NULL || coutput == NULL) {
    THROW(CNProcException, "cannot allocate buffers");
  }

  // wipe imaginary part
  for (unsigned i = 0; i < grid_n * 4; i++) {
    fftw_imag(cinput[i]) = 0.0;
  }

  // copy first part of grid
  for (unsigned i = 0; i < grid_n + 1; i++) {
    fftw_real(cinput[i]) = grid[i];
  }

  // append zeros
  for (unsigned i = grid_n + 1; i <= grid_n * 3; i++) {
    fftw_real(cinput[i]) = 0.0;
  }

  // now append the grid in reverse order
  for (unsigned i = grid_n - 1, index = 0; i >= 1; i --, index ++) {
    fftw_real(cinput[grid_n * 3 + 1 + index]) = grid[i];
  }

#if 0
  std::stringstream logStr;
  logStr << "ifft_in = [";
  for(unsigned i=0; i<grid_n*4; i++) {
    logStr << fftw_real(cinput[i]) << " " << fftw_imag(cinput[i]);
    if(i != grid_n*4-1) logStr << ", ";
  }
  logStr << "];";
  LOG_DEBUG(logStr.str());
#endif

#if defined HAVE_FFTW3
  fftwf_plan plan = fftwf_plan_dft_1d(grid_n * 4, cinput, coutput, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftwf_execute(plan);
#elif defined HAVE_FFTW2
  fftw_plan plan = fftw_create_plan(grid_n * 4, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_one(plan, cinput, coutput);
#endif

#if 0
  for(unsigned i=0; i<grid_n*4; i++) {
    LOG_DEBUG_STR("ifft result [" << i << "] = " << fftw_real(coutput[i]) << " " << fftw_imag(coutput[i]));
  }
#endif

  //                        half                   end
  // 1 2       n+1          2(n+1)      3(n+1)     4(n+1)
  //                                    x x x x x x x x x     # last quarter
  //   x x x x x x                                            # first quarter

  // last_quarter  = b([end-n+1:2:end]); # the size is only 1/8, since we skip half of the elements
  // first_quarter = b(2:2:(n+1));       # the size is only 1/8, since we skip half of the elements

  unsigned index = 0;

  for (unsigned i = 4 * grid_n - n; i < 4 * grid_n; i += 2) {
    result[index] = fftw_real(coutput[i]);
    index++;
  }

  for (unsigned i = 1; i <= n; i += 2) {
    result[index] = fftw_real(coutput[i]);
    index++;
  }

#if defined HAVE_FFTW3
  fftwf_destroy_plan(plan);
  fftwf_free(cinput);
  fftwf_free(coutput);
#elif defined HAVE_FFTW2
  fftw_destroy_plan(plan);
  fftw_free(cinput);
  fftw_free(coutput);
#endif

  // multiply with window
  for (unsigned i = 0; i <= n; i++) {
    result[i] *= window[i];
  }

  // normalize
  double factor = result[n / 2];
  for (unsigned i = 0; i <= n; i++) {
    result[i] /= factor;
  }

#if 0
  std::stringstream logStr;
  logStr << "result = [";
  for(unsigned i=0; i<=n; i++) {
    logStr << result[i];
    if(i != n) logStr << ", ";
  }
  logStr << "];";
  LOG_DEBUG(logStr.str());
#endif
}


#if ! USE_ORIGINAL_FILTER
// This method initializes the weights array.
void FilterBank::generate_filter()
{
  unsigned n = itsNrChannels * itsNrTaps;

  std::stringstream logStr;

  if (itsVerbose) {
    logStr << "generating FIR filter bank with " << itsNrChannels << " channels and " << itsNrTaps << " taps (" << n << " total), using a ";
  }

  std::vector<double> window(n);

  switch (itsWindowType) {
  case HAMMING: {
    // Use a n-point Hamming window.
    if (itsVerbose) {
      logStr << "Hamming window";
      LOG_DEBUG(logStr.str());
    }
    hamming(n, &window[0]);
    break;
  }
  case BLACKMAN: {
    // Use a n-point Blackman window.
    if (itsVerbose) {
      logStr << "Blackman window";
      LOG_DEBUG(logStr.str());
    }
    blackman(n, &window[0]);
    break;
  }
  case GAUSSIAN: {
    // Use a n-point Gaussian window.
    double alpha = 3.5;
    if (itsVerbose) {
      logStr << "Gaussian window with alpha = " << alpha;
      LOG_DEBUG(logStr.str());
    }
    gaussian(n, alpha, &window[0]);
    break;
  }
  case KAISER: {
    // Use a n-point Kaiser window.
    // The beta parameter is found in matlab / octave with
    // [n,Wn,bta,filtype]=kaiserord([fsin/channels 1.4*fsin/channels],[1 0],[10^(0.5/20) 10^(-91/20)],fsin);
    // where fsin is the sample freq
    double beta = 9.0695;
    if (itsVerbose) {
      logStr << "Kaiser window with beta = " << beta;
      LOG_DEBUG(logStr.str());
    }
    kaiser(n, beta, &window[0]);
    break;
  }
  default:
    THROW(CNProcException, "unknown window type");
  }

#if 0
  std::stringstream logStr;
  logStr << "window = [";
  for(unsigned i=0; i<n; i++) {
    logStr << window[i];
    if(i != n-1) logStr << ", ";
  }
  logStr << "];";
  LOG_DEBUG(logStr.str());
#endif

  std::vector<double> result(n);

  generate_fir_filter(n - 1, 1.0 / itsNrChannels, &window[0], &result[0]);

  weights.resize(boost::extents[itsNrChannels][itsNrTaps]);

  unsigned index = 0;
  for (int tap = itsNrTaps - 1; tap >= 0; tap--) { // store the taps in reverse!
    for (unsigned channel = 0; channel < itsNrChannels; channel++) {
      // Correct total power.
      // we use the 256 channel case as a reference, so we
      // multiply by 256, and divide by the number of channels
      weights[channel][tap] = result[index] * 256.0 / itsNrChannels;
      index++;
    }
  }

#if 0
  LOG_DEBUG("final taps: ");
  std::stringstream logStr;
  for(unsigned channel=0; channel<itsNrChannels; channel++) {
    logStr << "channel: " << channel << "| ";
    for(unsigned tap=0; tap<itsNrTaps; tap++) {
      logStr << " " << weights[channel][tap];
    }
    LOG_DEBUG(logStr.str());
  }
#endif
}

#else // USE_ORIGINAL_FILTER
// This method initializes the weights array.
void FilterBank::generate_filter()
{
  if(itsVerbose) {
    LOG_DEBUG("using original static 256 channel FIR filter bank");
  }

  if(itsNrTaps != 16 || itsNrChannels != 256) {
    THROW(CNProcException, "not supported!");
  }
  weights.resize(boost::extents[itsNrChannels][itsNrTaps]);
  memcpy(weights.origin(), origWeights, (itsNrChannels * itsNrTaps) * sizeof(float));
  itsNegated = true;
}
#endif // USE_ORIGINAL_FILTER
// In CEP, the first subband is from -98 KHz to 98 KHz, rather than from 0 to 195 KHz.
// To avoid that the FFT outputs the channels in the wrong order (from 128 to
// 255 followed by channels 0 to 127), we multiply each second FFT input by -1.
// This is efficiently achieved by negating the FIR filter constants of all
// uneven FIR filters.
void FilterBank::negateWeights() {
  for (int tap = itsNrTaps - 1; tap >= 0; tap--) { // store the taps in reverse!
    for (unsigned channel = 0; channel < itsNrChannels; channel++) {
      // Negate all odd channels
      if (channel % 2 != 0) {
        weights[channel][tap] = -weights[channel][tap];
      }
    }
  }
  itsNegated = !itsNegated;
}


// Used for debugging.
void FilterBank::reverseTaps() {
  for (unsigned channel = 0; channel < itsNrChannels; channel++) {
    for (unsigned tap = 0; tap < itsNrTaps/2; tap++) {
      float tmp = weights[channel][itsNrTaps - tap - 1];
      weights[channel][itsNrTaps - tap - 1] = weights[channel][tap];
      weights[channel][tap] = tmp;
    }
  }
}


// Print the weights array in the natural order, in a format that can be read by gnuplot.
void FilterBank::printWeights() {
  cout << (itsNegated ? "NEGATED" : "NORMAL(NOT NEGATED)") << endl;
  for (int tap = itsNrTaps - 1; tap >= 0; tap--) { // taps are stored in reverse!
    for (unsigned channel = 0; channel < itsNrChannels; channel++) {
      if (itsNegated && channel % 2 != 0) {
        cout << -weights[channel][tap] << endl; // odd channels are negated
      } else {
        cout << weights[channel][tap] << endl;
      }
    }
  }
}

} // namespace RTCP
} // namespace LOFAR
