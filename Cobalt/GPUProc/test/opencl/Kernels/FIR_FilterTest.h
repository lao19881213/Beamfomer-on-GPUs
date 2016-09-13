//# FIR_FilterTest.h
//# Copyright (C) 2012-2013  ASTRON (Netherlands Institute for Radio Astronomy)
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
//# $Id: FIR_FilterTest.h 25313 2013-06-13 02:12:26Z amesfoort $

#ifndef GPUPROC_FIR_FILTERTEST_H
#define GPUPROC_FIR_FILTERTEST_H

#include "CL/cl.hpp"
#include "UnitTest.h"
#include <complex>
#include <iostream>
#include <iomanip>
#include <GPUProc/FilterBank.h>
#include <GPUProc/Kernels/FIR_FilterKernel.h>

namespace LOFAR
{
    namespace Cobalt
    {
        struct FIR_FilterTest : public UnitTest
        {
            FIR_FilterTest(const Parset &ps)
                : UnitTest(ps, "FIR.cl")
            {
            	bool testOk = true;

                MultiArraySharedBuffer<float, 5> filteredData(
                    boost::extents[ps.nrStations()][NR_POLARIZATIONS][ps.nrSamplesPerChannel()][ps.nrChannelsPerSubband()][ps.nrBytesPerComplexSample()],
                    queue, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY);
                MultiArraySharedBuffer<signed char, 5> inputSamples(
                    boost::extents[ps.nrStations()][ps.nrPPFTaps() - 1 + ps.nrSamplesPerChannel()][ps.nrChannelsPerSubband()][NR_POLARIZATIONS][ps.nrBytesPerComplexSample()],
                    queue, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY);
                MultiArraySharedBuffer<float, 2> firWeights(
                    boost::extents[ps.nrChannelsPerSubband()][ps.nrPPFTaps()],
                    queue, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY);
                FIR_FilterKernel firFilterKernel(ps, queue, program, filteredData, inputSamples, firWeights);

                std::cout << "FIR_FilterTest: total num_el: firWeight=" << firWeights.num_elements() << " input="
                          << inputSamples.num_elements() << " output=" << filteredData.num_elements() << std::endl;

                unsigned station, sample, ch, pol;

                // Test 1: Single impulse test on single non-zero weight
                station = ch = pol = 0;
                sample = ps.nrPPFTaps() - 1; // skip FIR init samples
                firWeights.origin()[0] = 2.0f;
                inputSamples[station][sample][ch][pol][0] = 3;

                firWeights.hostToDevice(CL_FALSE);
                inputSamples.hostToDevice(CL_FALSE);
                firFilterKernel.enqueue(queue, counter);
                filteredData.deviceToHost(CL_TRUE);

                // Expected output: St0, pol0, ch0, sampl0: 6. The rest all 0.
                if (filteredData.origin()[0] != 6.0f) {
                    std::cerr << "FIR_FilterTest 1: Expected at idx 0: 6; got: " << std::setprecision(9+1) << filteredData.origin()[0] << std::endl;
                    testOk = false;
                }
                const unsigned nrExpectedZeros = filteredData.num_elements() - 1;
                unsigned nrZeros = 0;
                for (unsigned i = 1; i < filteredData.num_elements(); i++) {
                    if (filteredData.origin()[i] == 0.0f) {
                        nrZeros += 1;
                    }
                }
                if (nrZeros == nrExpectedZeros) {
                    std::cout << "FIR_FilterTest 1: test OK" << std::endl;
                } else {
                    std::cerr << "FIR_FilterTest 1: Unexpected non-zero(s). Only " << nrZeros << " zeros out of " << nrExpectedZeros << std::endl;
                    testOk = false;
                }


                // Test 2: Impulse train 2*NR_TAPS apart. All st, all ch, all pol.
                for (ch = 0; ch < ps.nrChannelsPerSubband(); ch++) {
                    for (unsigned tap = 0; tap < ps.nrPPFTaps(); tap++) {
                        firWeights[ch][tap] = ch + tap;
                    }
                }

                for (station = 0; station < ps.nrStations(); station++) {
                    for (sample = ps.nrPPFTaps() - 1; sample < ps.nrPPFTaps() - 1 + ps.nrSamplesPerChannel(); sample += 2 * ps.nrPPFTaps()) {
                        for (ch = 0; ch < ps.nrChannelsPerSubband(); ch++) {
                            for (pol = 0; pol < NR_POLARIZATIONS; pol++) {
                                inputSamples[station][sample][ch][pol][0] = station;
                            }
                        }
                    }
                }

                firWeights.hostToDevice(CL_FALSE);
                inputSamples.hostToDevice(CL_FALSE);
                firFilterKernel.enqueue(queue, counter);
                filteredData.deviceToHost(CL_TRUE);

                // Expected output: sequences of (filterbank scaled by station nr, NR_TAPS zeros)
                unsigned nrErrors = 0;
                for (station = 0; station < ps.nrStations(); station++) {
                    for (pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        unsigned s;
                        for (sample = 0; sample < ps.nrSamplesPerChannel() / (2 * ps.nrPPFTaps()); sample += s) {
                            for (s = 0; s < ps.nrPPFTaps(); s++) {
                                for (ch = 0; ch < ps.nrChannelsPerSubband(); ch++) {
                                    if (filteredData[station][pol][sample + s][ch][0] != station * firWeights[ch][s]) {
                                        if (++nrErrors < 100) { // limit spam
                                            std::cerr << "2a.filtered["<<station<<"]["<<pol<<"]["<<sample+s<<"]["<<ch<<
                                                "][0] (sample="<<sample<<" s="<<s<<") = " << std::setprecision(9+1) << filteredData[station][pol][sample + s][ch][0] << std::endl;
                                        }
                                    }
                                    if (filteredData[station][pol][sample + s][ch][1] != 0.0f) {
                                        if (++nrErrors < 100) {
                                            std::cerr << "2a imag non-zero: " << std::setprecision(9+1) << filteredData[station][pol][sample + s][ch][1] << std::endl;
                                        }
                                    }
                                }
                            }

                            for ( ; s < 2 * ps.nrPPFTaps(); s++) {
                                for (ch = 0; ch < ps.nrChannelsPerSubband(); ch++) {
                                    if (filteredData[station][pol][sample + s][ch][0] != 0.0f || filteredData[station][pol][sample + s][ch][1] != 0.0f) {
                                        if (++nrErrors < 100) {
                                            std::cerr << "2b.filtered["<<station<<"]["<<pol<<"]["<<sample+s<<"]["<<ch<<
                                                "][0] (sample="<<sample<<" s="<<s<<") = " << std::setprecision(9+1) << filteredData[station][pol][sample + s][ch][0] <<
                                                ", "<<filteredData[station][pol][sample + s][ch][1] << std::endl;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if (nrErrors == 0) {
                    std::cout << "FIR_FilterTest 2: test OK" << std::endl;
                } else {
                    std::cerr << "FIR_FilterTest 2: " << nrErrors << " unexpected output values" << std::endl;
                    testOk = false;
                }


                // Test 3: Scaled step test (scaled DC gain) on KAISER filterbank. Non-zero imag input.
                FilterBank filterBank(true, ps.nrPPFTaps(), ps.nrChannelsPerSubband(), KAISER);
                filterBank.negateWeights(); // not needed for testing, but as we use it
                //filterBank.printWeights();

                assert(firWeights.num_elements() == filterBank.getWeights().num_elements());
                double* expectedSums = new double[ps.nrChannelsPerSubband()];
                memset(expectedSums, 0, ps.nrChannelsPerSubband() * sizeof(double));
                for (ch = 0; ch < ps.nrChannelsPerSubband(); ch++) {
                    for (unsigned tap = 0; tap < ps.nrPPFTaps(); tap++) {
                        firWeights[ch][tap] = filterBank.getWeights()[ch][tap];
                        expectedSums[ch] += firWeights[ch][tap];
                    }
                }

                for (station = 0; station < ps.nrStations(); station++) {
                    for (sample = 0; sample < ps.nrPPFTaps() - 1 + ps.nrSamplesPerChannel(); sample++) {
                        for (ch = 0; ch < ps.nrChannelsPerSubband(); ch++) {
                            for (pol = 0; pol < NR_POLARIZATIONS; pol++) {
                                inputSamples[station][sample][ch][pol][0] = 2; // real
                                inputSamples[station][sample][ch][pol][1] = 3; // imag
                            }
                        }
                    }
                }

                firWeights.hostToDevice(CL_FALSE);
                inputSamples.hostToDevice(CL_FALSE);
                firFilterKernel.enqueue(queue, counter);
                filteredData.deviceToHost(CL_TRUE);

                nrErrors = 0;
                const float eps = 2.0f * std::numeric_limits<float>::epsilon();
                for (station = 0; station < ps.nrStations(); station++) {
                    for (pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        for (sample = 0; sample < ps.nrSamplesPerChannel(); sample++) {
                            for (ch = 0; ch < ps.nrChannelsPerSubband(); ch++) {
                                // Expected sum must also be scaled by 2 and 3, because weights are real only.
                                if (!fpEquals(filteredData[station][pol][sample][ch][0], (float)(2 * expectedSums[ch]), eps)) {
                                    if (++nrErrors < 100) { // limit spam
                                        std::cerr << "3a.filtered["<<station<<"]["<<pol<<"]["<<sample<<"]["<<ch<<
                                            "][0] = " << std::setprecision(9+1) << filteredData[station][pol][sample][ch][0] << " 2*weight = " << 2*expectedSums[ch] << std::endl;
                                    }
                                }
                                if (!fpEquals(filteredData[station][pol][sample][ch][1], (float)(3 * expectedSums[ch]), eps)) {
                                    if (++nrErrors < 100) {
                                        std::cerr << "3b.filtered["<<station<<"]["<<pol<<"]["<<sample<<"]["<<ch<<
                                            "][1] = " << std::setprecision(9+1) << filteredData[station][pol][sample][ch][1] << " 3*weight = " << 3*expectedSums[ch] << std::endl;
                                    }
                                }
                            }
                        }
                    }
                }
                delete[] expectedSums;
                if (nrErrors == 0) {
                    std::cout << "FIR_FilterTest 3: test OK" << std::endl;
                } else {
                    std::cerr << "FIR_FilterTest 3: " << nrErrors << " unexpected output values" << std::endl;
                    testOk = false;
                }


                check(testOk, true);
            }
        };
    }
}

#endif

