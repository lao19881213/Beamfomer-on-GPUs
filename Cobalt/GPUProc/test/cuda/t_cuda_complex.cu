//# t_cuda_complex.cu: tests our complex type for CUDA device code
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
//# $Id: t_cuda_complex.cu 25878 2013-08-01 02:08:38Z amesfoort $

#include <lofar_config.h>

#include <cstdio>
#include <iostream>

// Note: watch what namespace you're referring to in this file! Outside device code, keep it explicit.
#include <complex>

#include "complex.cuh"

cudaError_t addWithCuda( std::complex<float> * output_complex, const std::complex<float> * input_complex, size_t size);

__global__ void addKernel( void *in_ptr, const void *out_ptr)
{
    int i = threadIdx.x;
    // Cast to complex

    complex<float>*in = (complex<float>*) in_ptr;
    complex<float>*out = (complex<float>*) out_ptr;

    //do some computations, We are not testing the correctness of the implementation here.
    out[i] = in[i] + in[i];
    out[i] -= in[i];
    out[i] = out[i];
    out[i] *= 10.0f;
}

int main()
{
    const int arraySize = 5;
    // insert some values
    const std::complex<float> complex_in[5] = { std::complex<float>(1.0,1.0),
                                                std::complex<float>(1,-1),
                                                std::complex<float>(-1,1),
                                                std::complex<float>(-1,-1),
                                                std::complex<float>(4,-4) };
    std::complex<float> complex_out[5] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(complex_out, complex_in,arraySize);
    if (cudaStatus == cudaErrorNoDevice) {
        return 3;
    }
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    const std::complex<float> complex_target[5] = { std::complex<float>(10,10),
                                                    std::complex<float>(10, -10),
                                                    std::complex<float>(-10,10),
                                                    std::complex<float>(-10,-10),
                                                    std::complex<float>(40,-40) };
      

    // validate that the output of the kernel is correct!
    if (complex_out[0] == complex_target[0] &&
        complex_out[1] == complex_target[1] &&
        complex_out[2] == complex_target[2] &&
        complex_out[3] == complex_target[3] &&
        complex_out[4] == complex_target[4]
      )
      {
        return 0;
      }
    else //print the output data and return -1
    {
      std::cout << "The complex values returned from the device were incorrect:" << std::endl;
      std::cout << "complex numbers, expected - received: {";
      for (int idx =0; idx < 5; ++idx)
      {
        std::cout << complex_target[idx] << " - " << complex_out[idx] ;
        if (complex_target[idx]  != complex_out[idx])
          std::cout << "<<<";
        std::cout << std::endl;
      }
      std::cout << " }" << std::endl;
      return -1;
    }
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(std::complex<float>* output_complex,
                        const std::complex<float>* input_complex,
                        size_t size)
{
    std::complex<float> *dev_in = 0;
    std::complex<float> *dev_out = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
        goto out;
    }

    // allocate the std::complex buffers
    cudaStatus = cudaMalloc((void**)&dev_in, size * sizeof(std::complex<float>));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto out;
    }

    cudaStatus = cudaMalloc((void**)&dev_out, size * sizeof(std::complex<float>));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto errin;
    }

    cudaStatus = cudaMemcpy(dev_in, input_complex, size * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto errout;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>((void *) dev_in, (const void *) dev_out);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto errout;
    }

    cudaStatus = cudaMemcpy(output_complex, dev_out, size * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto errout;
    }


errout:
    cudaFree(dev_out);
errin:
    cudaFree(dev_in);
out:
    return cudaStatus;
}

