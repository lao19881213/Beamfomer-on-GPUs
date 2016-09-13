//# tIntToFloat.cc: test Int2Float CUDA kernel
//# Copyright (C) 2013  ASTRON (Netherlands Institute for Radio Astronomy)
//# P.O. Box 2, 7990 AA Dwingeloo, The Netherlands
//#
//# This file is part of the LOFAR software suite.
//# The LOFAR software suite is free software: you can redistribute it and/or
//# modify it under the terms of the GNU General Public License as published
//# by the Free Software Foundation, either version 3 of the License, or
//# (at your option) any later version.
//#
//# The LOFAR software suite is distributed in the hope that it will be useful,
//# but WITHOUT ANY WARRANTY; without even the implied warranty of
//# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//# GNU General Public License for more details.
//#
//# You should have received a copy of the GNU General Public License along
//# with the LOFAR software suite. If not, see <http://www.gnu.org/licenses/>.
//#
//# $Id: tIntToFloat.cc 26571 2013-09-16 18:18:19Z amesfoort $

#include <lofar_config.h>

#include <cstdlib>
#include <cstring>
#include <cmath> 
#include <cassert>
#include <string>
#include <sstream>
#include <typeinfo>
#include <vector>
#include <limits>

#include <boost/lexical_cast.hpp>
#include <UnitTest++.h>

#include <Common/LofarLogger.h>
#include <Common/LofarTypes.h>
#include <GPUProc/gpu_wrapper.h>
#include <GPUProc/gpu_utils.h>
#include <GPUProc/MultiDimArrayHostBuffer.h>

using namespace std;
using namespace LOFAR::Cobalt;

using LOFAR::i16complex;
using LOFAR::i8complex;
using LOFAR::i4complex; // TODO: add support for 4 bit mode (only 1 func was done)

static gpu::Stream *stream;

// default compile definitions
const unsigned NR_STATIONS = 5;
const unsigned NR_CHANNELS = 64;
const unsigned NR_SAMPLES_PER_CHANNEL = 16;
const unsigned NR_SAMPLES_PER_SUBBAND = NR_SAMPLES_PER_CHANNEL * NR_CHANNELS;
const unsigned NR_BITS_PER_SAMPLE = 8;
const unsigned NR_POLARIZATIONS = 2;


// Initialize input AND output before calling runKernel().
// We copy both to the GPU, to make sure the final output is really from the kernel.
// T is an LCS i*complex type.
template <typename T>
void runKernel(gpu::Function kfunc,
               MultiDimArrayHostBuffer<T, 3>& input,
               MultiDimArrayHostBuffer<complex<float>, 3>& output)
{
  gpu::Context ctx(stream->getContext());

  gpu::DeviceMemory devInput (ctx, input.size());
  gpu::DeviceMemory devOutput(ctx, output.size());

  kfunc.setArg(0, devOutput);
  kfunc.setArg(1, devInput);

  gpu::Grid globalWorkSize(1, NR_STATIONS, 1);  
  gpu::Block localWorkSize(256, 1, 1); 

  // Overwrite devOutput, so result verification is more reliable.
  stream->writeBuffer(devOutput, output);

  stream->writeBuffer(devInput, input);
  stream->launchKernel(kfunc, globalWorkSize, localWorkSize);
  stream->readBuffer(output, devOutput);
  stream->synchronize(); // wait until transfer completes
}

gpu::Function initKernel(gpu::Context ctx, const CompileDefinitions& defs)
{
  // Compile to ptx. Copies the kernel to the current dir
  // (also the complex header, needed for compilation).
  string kernelPath("IntToFloat.cu");
  CompileFlags flags(defaultCompileFlags());
  vector<gpu::Device> devices(1, gpu::Device(0));
  string ptx(createPTX(kernelPath, defs, flags, devices));
  gpu::Module module(createModule(ctx, kernelPath, ptx));
  gpu::Function kfunc(module, "intToFloat");

  return kfunc;
}

CompileDefinitions getDefaultCompileDefinitions()
{
  CompileDefinitions defs;

  defs["NR_STATIONS"]            = boost::lexical_cast<string>(NR_STATIONS);
  defs["NR_CHANNELS"]            = boost::lexical_cast<string>(NR_CHANNELS);
  defs["NR_SAMPLES_PER_CHANNEL"] = boost::lexical_cast<string>(NR_SAMPLES_PER_CHANNEL);
  defs["NR_SAMPLES_PER_SUBBAND"] = boost::lexical_cast<string>(NR_SAMPLES_PER_SUBBAND);
  defs["NR_BITS_PER_SAMPLE"]     = boost::lexical_cast<string>(NR_BITS_PER_SAMPLE);
  defs["NR_POLARIZATIONS"]       = boost::lexical_cast<string>(NR_POLARIZATIONS);

  return defs;
}

// T is an LCS i*complex type.
template <typename T>
vector<complex<float> > runTest(int defaultVal)
{
  gpu::Context ctx(stream->getContext());

  // Don't use the Kernel class helpers to retrieve buffer sizes,
  // because we test the kernel, not the Kernel class.
  MultiDimArrayHostBuffer<T, 3>              input (boost::extents[NR_STATIONS][NR_SAMPLES_PER_SUBBAND][NR_POLARIZATIONS], ctx);
  MultiDimArrayHostBuffer<complex<float>, 3> output(boost::extents[NR_STATIONS][NR_POLARIZATIONS][NR_SAMPLES_PER_SUBBAND], ctx);

  // set input
  for (size_t i = 0; i < input.num_elements(); i++) {
    input.origin()[i].real() = defaultVal;
    input.origin()[i].imag() = defaultVal;
  }

  // set output for proper verification later
  memset(output.origin(), 0, output.size());

  CompileDefinitions compileDefs(getDefaultCompileDefinitions());

  // Apply test-specific parameters for mem alloc sizes and kernel compilation.
  unsigned nrBitsPerSample = sizeof(T) * 8 / 2;
  compileDefs["NR_BITS_PER_SAMPLE"] = boost::lexical_cast<string>(nrBitsPerSample);
  gpu::Function kfunc(initKernel(ctx, compileDefs));

  runKernel(kfunc, input, output);

  // Tests that use this function only check the first and last 2 output floats.
  const unsigned nrResultVals = 2;
  assert(output.num_elements() >= nrResultVals * sizeof(complex<float>) / sizeof(float));
  vector<complex<float> > outputrv(nrResultVals);
  outputrv[0] = output.origin()[0];
  outputrv[1] = output.origin()[output.num_elements() - 1];
  return outputrv;
}

// T is an LCS i*complex type.
// Separate function, because it'll need to be specialized for 4 bit mode,
// as i4complex has no T::value_type.
template <typename T>
void setSample(T& sample, int& val)
{
  sample.real() = val;
  if (val == numeric_limits<typename T::value_type>::max())
    val = numeric_limits<typename T::value_type>::min(); // wrap back to minimum
  else
    val += 1;

  sample.imag() = val;
  if (val == numeric_limits<typename T::value_type>::max())
    val = numeric_limits<typename T::value_type>::min(); // wrap back to minimum
  else
    val += 1;
}

// 4 bit mode specialization
template <>
void setSample<i4complex>(i4complex& sample, int& val)
{
  int real = val;
  if (val == 7)
    val = -8; // wrap back to minimum
  else
    val += 1;

  int imag = val;
  if (val == 7)
    val = -8; // wrap back to minimum
  else
    val += 1;

  // Avoid makei4complex(): we need to manage clamping manually to test the kernel.
  *reinterpret_cast<int8_t *>(&sample) = (real << 4) | imag;
}

// T is an LCS i*complex type.
template <typename T>
void initArray(MultiDimArrayHostBuffer<T, 3>& buf)
{
  // Repeatedly write (-128, -127, ..., 0, ... 127) + station_idx; or whatever is the max value range for T.
  for (unsigned s = 0; s < buf.shape()[0]; s++) {
    int val = numeric_limits<typename T::value_type>::min() + (s % 13); // s % 13 to avoid overflow with 4 bit samples (16 is enough), and make it odd ball
    for (unsigned t = 0; t < buf.shape()[1]; t++) {
      for (unsigned p = 0; p < buf.shape()[2]; p++) {
        setSample(buf[s][t][p], val); // set complex val to (x, x+1) circularly
      }
    }
  }
}

void checkSubSample16(float s, int& expectedVal)
{
  CHECK_CLOSE((float)expectedVal, s, 0.00000001);
  if (expectedVal == numeric_limits<int16_t>::max())
    expectedVal = numeric_limits<int16_t>::min(); // wrap back to minimum
  else
    expectedVal += 1;
}

// In 8 bit mode, -128, if expressable, is converted and clamped to -127.0f.
// Also, everything is scaled to the amplitude of 16 bit mode (taking the effect of correlation into account).
void checkSubSample8(float s, int& expectedVal)
{
  const int scale = 16;
  if (expectedVal == numeric_limits<int8_t>::min()) {
    CHECK_CLOSE((float)(scale * (expectedVal+1)), s, 0.00000001); // check clamping
  } else {
    CHECK_CLOSE((float)(scale * expectedVal), s, 0.00000001);
  }
  if (expectedVal == numeric_limits<int8_t>::max())
    expectedVal = numeric_limits<int8_t>::min(); // wrap back to minimum
  else
    expectedVal += 1;
}

// T is an LCS i*complex type. Needed for numeric_limits.
// Note that the IntToFloat kernel also transposes: it outputs first all polX, then all polY.
template <typename T>
void checkTransposedArray(const MultiDimArrayHostBuffer<complex<float>, 3>& buf)
{
  for (unsigned s = 0; s < buf.shape()[0]; s++) {
    int expectedVal = numeric_limits<typename T::value_type>::min() + (s % 13);
    // NOTE: the kernel transposed the time and pol dims. We generate the expected numbers
    // in the same order as init, so step through the data with these dims reversed. 
    for (unsigned t = 0; t < buf.shape()[2]; t++) {
      for (unsigned p = 0; p < buf.shape()[1]; p++) {
        // Templating checkSample*() isn't much use as unlike setSample(),
        // all three variants here need specialization due to clamping and scaling.
        if (sizeof(T) == sizeof(i16complex)) {
          checkSubSample16(buf[s][p][t].real(), expectedVal);
          checkSubSample16(buf[s][p][t].imag(), expectedVal);
        } else if (sizeof(T) == sizeof(i8complex)) {
          checkSubSample8(buf[s][p][t].real(), expectedVal);
          checkSubSample8(buf[s][p][t].imag(), expectedVal);
        } else {
          assert(sizeof(T) == 0); // abort: 4 bit mode or unexpected T is unsupported
        }
      }
    }
  }
}

// T is an LCS i*complex type.
template <typename T>
void runTest2()
{
  gpu::Context ctx(stream->getContext());

  // Don't use the Kernel class helpers to retrieve buffer sizes,
  // because we test the kernel, not the Kernel class.
  MultiDimArrayHostBuffer<T, 3>              input (boost::extents[NR_STATIONS][NR_SAMPLES_PER_SUBBAND][NR_POLARIZATIONS], ctx);
  MultiDimArrayHostBuffer<complex<float>, 3> output(boost::extents[NR_STATIONS][NR_POLARIZATIONS][NR_SAMPLES_PER_SUBBAND], ctx);

  initArray(input);

  // set output for proper verification later
  memset(output.origin(), 0, output.size());

  CompileDefinitions compileDefs(getDefaultCompileDefinitions());

  // Apply test-specific parameters for mem alloc sizes and kernel compilation.
  unsigned nrBitsPerSample = sizeof(T) * 8 / 2; // 4, 8, or 16 bit mode
  compileDefs["NR_BITS_PER_SAMPLE"] = boost::lexical_cast<string>(nrBitsPerSample);
  gpu::Function kfunc(initKernel(ctx, compileDefs));


  runKernel(kfunc, input, output);

  checkTransposedArray<T>(output);
}


// Unit tests of value conversion
TEST(CornerCaseMinus128)
{
  // Test the corner case for 8 bit input, -128 should be clamped to scaled -127
  vector<complex<float> > results(runTest<i8complex>(-128));

  const float scale = 16.0f;
  CHECK_CLOSE(scale * -127.0, results[0].real(), 0.00000001);
  CHECK_CLOSE(scale * -127.0, results[0].imag(), 0.00000001);
  CHECK_CLOSE(scale * -127.0, results[1].real(), 0.00000001);
  CHECK_CLOSE(scale * -127.0, results[1].imag(), 0.00000001);
}

TEST(CornerCaseMinus128short)
{
  // The -128 to -127 clamp should not be applied to 16 bit samples
  vector<complex<float> > results(runTest<i16complex>(-128));

  CHECK_CLOSE(-128.0, results[0].real(), 0.00000001);
  CHECK_CLOSE(-128.0, results[0].imag(), 0.00000001);
  CHECK_CLOSE(-128.0, results[1].real(), 0.00000001);
  CHECK_CLOSE(-128.0, results[1].imag(), 0.00000001);
}

TEST(CornerCaseMinus127)
{
  // Minus 127 should stay -127
  vector<complex<float> > results(runTest<i8complex>(-127));

  const float scale = 16.0f;
  CHECK_CLOSE(scale * -127.0, results[0].real(), 0.00000001);
  CHECK_CLOSE(scale * -127.0, results[0].imag(), 0.00000001);
  CHECK_CLOSE(scale * -127.0, results[1].real(), 0.00000001);
  CHECK_CLOSE(scale * -127.0, results[1].imag(), 0.00000001);
}

TEST(IntToFloatSimple)
{
  // Test if 10 is converted
  vector<complex<float> > results(runTest<i8complex>(10));

  const float scale = 16.0f;
  CHECK_CLOSE(scale * 10.0, results[0].real(), 0.00000001);
  CHECK_CLOSE(scale * 10.0, results[0].imag(), 0.00000001);
  CHECK_CLOSE(scale * 10.0, results[1].real(), 0.00000001);
  CHECK_CLOSE(scale * 10.0, results[1].imag(), 0.00000001);
}

TEST(IntToFloatSimpleShort)
{
  // Test if 2034 is converted
  vector<complex<float> > results(runTest<i16complex>(2034));

  CHECK_CLOSE(2034.0, results[0].real(), 0.00000001);
  CHECK_CLOSE(2034.0, results[0].imag(), 0.00000001);
  CHECK_CLOSE(2034.0, results[1].real(), 0.00000001);
  CHECK_CLOSE(2034.0, results[1].imag(), 0.00000001);
}

// These tests use different values and as such also the transpose.
TEST(AllVals8)
{
  runTest2<i8complex>();
}
TEST(AllVals16)
{
  runTest2<i16complex>();
}

gpu::Stream initDevice()
{
  // Set up device (GPU) environment
  try {
    gpu::Platform pf;
    cout << "Detected " << pf.size() << " GPU devices" << endl;
  } catch (gpu::CUDAException& e) {
    cerr << e.what() << endl;
    exit(3); // test skipped
  }
  gpu::Device device(0);
  vector<gpu::Device> devices(1, device);
  gpu::Context ctx(device);
  gpu::Stream cuStream(ctx);

  return cuStream;
}

int main()
{
  INIT_LOGGER("tIntToFloat");

  // init global(s): device, context/stream.
  gpu::Stream strm(initDevice());
  stream = &strm;

  int exitStatus = UnitTest::RunAllTests();
  return exitStatus > 0 ? 1 : 0;
}

