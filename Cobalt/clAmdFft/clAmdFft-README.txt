clAmdFft Readme

Version:       1.8
Release Date:  September 2012

ChangeLog:

____________
Current version:
Fixed:
  * Failures in real transforms seen on 7xxx series GPUs with certain
      problem sizes involving powers of 3 and 5 

Known Issues:
  * Library may return invalid results on CPU devices.
	  
____________
Version 1.8.276 (beta):
Fixed:
  * Memory leaks affecting use cases where 'clAmdFftEnqueueTransform' is used in a loop
	  
____________
Version 1.8.269 (beta):
New:
  * clAmdFft now supports real-to-complex and complex-to-real transforms;
      refer to documentation for details
  * This release tested using the 12.4 Catalyst software suite
	  
Known Issues:
  * Some degradation in performance of real transforms due to known
      runtime/driver issues
  * Failures in real transforms have been seen on 7xxx series GPUs with certain
      problem sizes involving powers of 3 and 5  
  
____________
Version 1.6.244:
Fixed:
  * Failures observed in v1.6.236 in backward transforms of certain power of 2
      (involving radix 4 and radix 8) problem sizes.
	  
____________
Version 1.6.236:
New:
  * Performance of the FFT library has been improved for Radix-2 1D and 2D transforms
  * Support for R4XXX GPUs is deprecated and no longer tested
  * Preview: Support for AMD Radeon� HD7000 series GPUs
  * This release tested using the 8.92 runtime driver and the 2.6 APP SDK
____________
Version 1.4:
New:
  * clAmdFft now supports transform lengths whose factors consist exclusively 
      of powers of 2, 3, and 5
  * clAmdFft supports double precision data types
  * clAmdFft executes on OpenCL 1.0 compliant devices
  * This release tested using the 8.872 runtime driver and the 2.5 APP SDK
  * A helper bash script appmlEnv.sh has been added to the root installation
      directory to assist in properly setting up a terminal environment to 
      execute clAmdFft samples

Fixed:
  * If the library is required to allocate a temporary buffer, and the user does
      not specify a temporary buffer on the Enqueue call, the library will 
      allocate a temporary buffer internally and the lifetime of that temporary 
      buffer is managed by the lifetime of the FFT plan; deleting the plan will 
      release the buffer.
  * Test failures on CPU device for 32-bit systems  (Windows/Linux) 

Known Issues:
  * Failures have been seen on graphics cards using R4550 (RV710) GPUs.
  
____________
Version 1.2:
New:
  * Reduced the number of internal LDS bank conflicts for our 1D FFT transforms,
      increasing performance.
  * Padded reads/writes to global memory, decreasing bank conflicts and 
      increasing performance on 2D transforms.
  * This release tested using the 8.841 runtime driver and the 2.4 APP SDK

Fixed:
  * Failures have been seen attempting to queue work on the second GPU device on
      a multi GPU 5970 card on Linux.

Known Issues:
  * It is recommended that users query for and explicitely create an 
      intermediate buffer if clAmdFft requires one.  If the library creates the 
      intermediate buffer internally, a race condition may occur on freeing the 
      buffer on lower end hardware.
  * Failures have been seen on graphics cards using R4550 (RV710) GPUs.
  * Test failures on CPU device for 32-bit systems  (Windows/Linux) 
  * It is recommended that windows users uninstall previous version of clAmdFft 
      before installing newer versions.  Otherwise, Add/Remove programs only 
      removes the latest version.  Linux users can delete the install directory.

____________
Version 1.0:
  * Initial release, available on all platforms

Known Issues:
  * Failures have been seen attempting to queue work on the second GPU device on
      a multi GPU 5970 card on Linux.
_____________________
Building the Samples:

To install the Linux versions of clAmdFft, uncompress the initial download and 
  then execute the install script.

For example:
  tar -xf clAmdFft-${version}.tar.gz
      - This installs three files into the local directory, one being an 
        executable bash script.

  sudo mkdir /opt/clAmdFft-${version}
      - This pre-creates the install directory with proper permissions in /opt 
        if it is to be installed there (This is the default).

  ./install-clAmdFft-${version}.sh
      - This prints an EULA and uncompresses files into the chosen install 
        directory.

  cd ${installDir}/bin64
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${OpenCLLibDir}:${clAmdFftLibDir}
      - Export library dependencies to resolve all external linkages to the 
        client program. The user can create a bash script to help automate this 
        procedure.

  ./clAmdFft.Client -h
      - Understand the command line options that are available to the user 
        through the sample client.

  ./clAmdFft.Client -iv
      - Watch for the version strings to print out; watch for 
        'Client Test *****PASS*****' to print out.

The sample program does not ship with native build files. Instead, a CMake
file is shipped, and users generate a native build file for their system.

For example:
  cd ${installDir}
  mkdir samplesBin/
      - This creates a sister directory to the samples directory that will house
        the native makefiles and the generated files from the build.

  cd samplesBin/
  ccmake ../samples/
      - ccmake is a curses-based cmake program. It takes a parameter that 
        specifies the location of the source code to compile.
      - Hit 'c' to configure for the platform; ensure that the dependencies to 
        external libraries are satisfied, including paths to 'ATI Stream SDK' 
        and 'Boost'.
      - After dependencies are satisfied, hit 'c' again to finalize configure 
        step, then hit 'g' to generate makefile and exit ccmake.

  make help
      - Look at the available options for make.

  make
      - Build the sample client program.

  ./clAmdFft.Sample -iv
      - Watch for the version strings to print out; watch for 
        'Client Test *****PASS*****' to print out.
_______________________________________________________________________________
(C) 2010,2011 Advanced Micro Devices, Inc. All rights reserved. AMD, the AMD 
Arrow logo, ATI, the ATI logo, Radeon, FireStream, FireGL, Catalyst, and 
combinations thereof are trademarks of Advanced Micro Devices, Inc. Microsoft 
(R), Windows, and Windows Vista (R) are registered trademarks of Microsoft 
Corporation in the U.S. and/or other jurisdictions. OpenCL and the OpenCL logo 
are trademarks of Apple Inc. used by permission by Khronos. Other names are for 
informational purposes only and may be trademarks of their respective owners.

The contents of this document are provided in connection with Advanced Micro 
Devices, Inc. ("AMD") products. AMD makes no representations or warranties with 
respect to the accuracy or completeness of the contents of this publication and 
reserves the right to make changes to specifications and product descriptions 
at any time without notice. The information contained herein may be of a 
preliminary or advance nature and is subject to change without notice. No 
license, whether express, implied, arising by estoppel or otherwise, to any 
intellectual property rights is granted by this publication. Except as set forth
in AMD's Standard Terms and Conditions of Sale, AMD assumes no liability 
whatsoever, and disclaims any express or implied warranty, relating to its 
products including, but not limited to, the implied warranty of 
merchantability, fitness for a particular purpose, or infringement of any 
intellectual property right.

AMD's products are not designed, intended, authorized or warranted for use as 
components in systems intended for surgical implant into the body, or in other 
applications intended to support or sustain life, or in any other application 
in which the failure of AMD's product could create a situation where personal 
injury, death, or severe property or environmental damage may occur. AMD 
reserves the right to discontinue or make changes to its products at any time 
without notice.
_______________________________________________________________________________
