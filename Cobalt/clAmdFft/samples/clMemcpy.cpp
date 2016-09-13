////////////////////////////////////////////
//	Copyright (C) 2010,2011 Advanced Micro Devices, Inc. All Rights Reserved.
////////////////////////////////////////////

// clAmdFft.clMemcpy.cpp : OpenCL memory copy kernel generator
//
//
//
////////////////////////////////////////////////////////////////////////////////

//	TODO: Add 2d/tiled memory copies.

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/opencl.h>
#include <iostream>
#include <vector>
#include <time.h>

#include <sstream>
#include <string>
using std::stringstream;

#include <boost/program_options.hpp>
namespace po = boost::program_options;

//#include "../statTimer/clAmdFft.statisticalTimer.extern.h"
//#include "../include/clAmdFft.sharedLibrary.h"

#include "../../../common/statisticalTimer.h"

#include "../../../common/amd-unicode.h"

class clDataType{
public:
	virtual bool setSize(size_t size) = 0; // set size
	virtual size_t getSize() = 0;		// size in bytes
	virtual size_t getTypeSize() = 0;	// size of base type in bytes
	virtual std::string getName() = 0;	// get cl type name
};

class clFloat:public clDataType{
public:

	clFloat()
	{
		clSize = 2;
	}

	size_t getSize()
	{
		return clSize * sizeof(float);
	}

	size_t getTypeSize()
	{
		return sizeof(float);
	}

	std::string getName()
	{
		std::stringstream name;
		name << "float";
		if(clSize > 1)
		{
			name << clSize;
		}

		std::string ret = name.str().c_str();
		return ret;
	}

	bool setSize(size_t size)
	{
		if(size < 0 || size > 16 || (size & (size - 1)) )
		{
			return false;
		}
		else
		{
			clSize = size;
			return true;
		}
	}
	// The null kernel generator has its own special set of paramters

private:
	size_t clSize;
};

class clDouble:public clDataType{
public:

	clDouble()
	{
		clSize = 1;
	}

	size_t getSize()
	{
		return clSize * sizeof(double);
	}

	size_t getTypeSize()
	{
		return sizeof(double);
	}

	std::string getName()
	{
		std::stringstream name;
		name << "double";
		if(clSize > 1)
		{
			name << clSize;
		}
		
		std::string ret = name.str().c_str();
		return ret;
	}

	bool setSize(size_t size)
	{
		if(size < 0 || size > 4 || (size & (size - 1)))
		{
			return false;
		}
		else
		{
			clSize = size;
			return true;
		}
	}
	// The null kernel generator has its own special set of paramters

private:
	size_t clSize;
};

// This is a helper function to query a device for it's caps and check whether a certain user supplied cap is present
// stolen from the clAmdRuntime library
bool checkDevExt( std::string cap, std::vector< cl_device_id >& devices )
{
	for( size_t d = 0; d < devices.size( ); ++d)
	{
		size_t deviceExtSize	= 0;
		 ::clGetDeviceInfo( devices[ d ], CL_DEVICE_EXTENSIONS, 0, NULL, &deviceExtSize ),
			"Getting CL_DEVICE_EXTENSIONS Platform Info string size ( ::clGetDeviceInfo() )";

		std::vector< char > szDeviceExt( deviceExtSize );
		::clGetDeviceInfo( devices[ d ], CL_DEVICE_EXTENSIONS, deviceExtSize, &szDeviceExt[ 0 ], NULL ),
			"Getting CL_DEVICE_EXTENSIONS Platform Info string ( ::clGetDeviceInfo() )";

		std::string strDeviceExt = &szDeviceExt[ 0 ];

		if( strDeviceExt.find( cap.c_str( ), 0 ) == std::string::npos )
			return 0;
	}

	return true;
}

#define INDENT "    "

// memcpy kernel generator, very simple
//
 void GenerateMemcpyKernel (stringstream &ssn, const int registerCount, const int dumbyRegisterCount, const int workGroupSize , clDataType * clType,  const bool useBarrier, int ldsPasses, const int dataItemCount, const int writeOnly, const int readOnly, const int memcpyOnly, const bool supportDoublePrecision)
{
// kernel generator - dumb

	//std::stringstream ssn         (std::stringstream::out);
	static const bool first_choice = true;

	ssn << "//------------------------------\n"
	     "// !!!!!NULL Memcopy KERNEL!!!!\n\n";

	// include double precision support
	
	if(supportDoublePrecision)
	{
		ssn<< "\n#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n\n";
	}

	// set the workgroup size to our specification, this will effect the number of wavefronts used	
	ssn << "__attribute__((reqd_work_group_size(" << workGroupSize << ",1,1)))\n"
		<< "__kernel void\n"
		<< "memcpy" << "(\n";


// basically do inplace memcopy unless memcpyOnly is true, then do out of place

	if(!memcpyOnly)
	{
		ssn << INDENT "__global " << clType->getName() << " *gcomplx\n";
		ssn << ")\n{\n";
	}
	else
	{
		ssn << INDENT "__global const " << clType->getName() << " *in,\n";
		ssn << INDENT "__global " << clType->getName() << " *out\n";
		ssn << ")\n{\n";

		// a strict memcopy kernel does not require much, just code it here and return

		ssn << INDENT "int gid = get_global_id(0);\n";
		ssn << INDENT "out[gid] = in[gid];\n";
		ssn << INDENT "return;";
		ssn << "\n}\n";

		return;
	}

	
	// create registers for kernel to use for memcopies
	ssn << "\n" << clType->getName() << " R0";
	for(int i = 1; i < registerCount + dumbyRegisterCount; i++)
	{
		ssn << ",R" << i;
	}

	ssn << ";\n";

	// identifiers for local work item and global group id
	ssn << "\nuint me = get_local_id(0);";
	ssn << "\nuint batch = get_group_id(0);";

	ssn << "\nglobal ";
	
	// if read only kernel use const to disable read caching
	if(writeOnly)
	{
	ssn << "const ";
	}
		
	ssn << clType->getName() << "* gpc = gcomplx + me * " <<  registerCount << " + batch * " << registerCount * workGroupSize << ";";
	
	if(ldsPasses > 0)
	{
		// allocate LDS
		ssn << "\n__local " << clType->getName() << " ldsBuff[" << registerCount * workGroupSize << "];";
		ssn << "\n__local " << clType->getName() << "* lds = ldsBuff + me * " << registerCount << ";";
	}
	
	ssn << "\n";

	// If write only kernel, don't read back regs to global memory
	if(writeOnly || !readOnly)
	{
		// copy data from Global Memory to regs
		for(int i = 0; i < registerCount; i++)
		{
			ssn << "\nR" << i << "= gpc[" << i << "];";
		}
	}

	ssn << "\n";

	// make number of LDS passes specified, copy from regs to lds back to regs
	for(int j = 0; j < ldsPasses; j++) 
	{

		// copy data from regs to LDS
		for(int i = 0; i < registerCount; i++)
		{
			
			ssn << "\nlds[" << i << "] = R" << i << ";";
		}

		ssn << "\n";
	
		// insert memory barrier
		if(useBarrier == true)
		{
			ssn << "\nbarrier(CLK_LOCAL_MEM_FENCE);\n";
		}

		// copy data from LDS back to regs
		for(int i = 0; i < registerCount; i++)
		{
			ssn << "\nR" << i << " = lds[" << (registerCount -1 ) - i << "];";
		}

		ssn << "\n";

	}
	
	// if dumby registers are specified, just assign a value to them
	// do some math
	int rIndex = 0;
	for(int i = registerCount; i < registerCount + dumbyRegisterCount; i++)
	{
		if( i == registerCount)
		{
			ssn << "\nR" << i << " = R0 * 3.1459;";
	//		ssn << "\nR0 = R" << i << ";";  // write results to R0 to kee it from being optimized out
		}
		else
		{
			ssn << "\nR" << i << " = R" << i <<" + R" << i - 1 << " * 3.1459;";
		}
	
		ssn << "\nR" << rIndex <<" = R" << i << ";";
		rIndex ++;
		if(rIndex >= registerCount)
		{
			rIndex = 0;
		}
	}

	ssn << "\n";

	// if readonly or not a writeonly kernel copy registers back to global memory
	if(readOnly || !writeOnly)
	{
		for(int i = 0; i < registerCount; i++)
		{
			ssn << "\ngpc[" << i << "] = R" << i << ";";
		}
	}
	ssn << "\n}\n";
}

// http://cottonvibes.blogspot.com/2011/01/dynamically-allocate-aligned-memory.html
// Alignment must be power of 2 (1,2,4,8,16...2^15)
void* aligned_malloc(size_t size, size_t alignment) {
    assert(alignment <= 0x8000);
    uintptr_t r = (uintptr_t)malloc(size + --alignment + 2);
    uintptr_t o = (r + 2 + alignment) & ~(uintptr_t)alignment;
    if (!r) return NULL;
    ((uint16_t*)o)[-1] = (uint16_t)(o-r);
    return (void*)o;
}
 
void aligned_free(void* p) {
    if (!p) return;
    free((void*)((uintptr_t)p-((uint16_t*)p)[-1]));
}


int main(int argc, char** argv)
{
    int err;                            // error code returned from api calls

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation
	cl_platform_id platform;
    cl_device_id device_id;             // compute device id 
	cl_uint platforms;
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel
    
    cl_mem input;                       // device memory used for the input array
	cl_mem output;                      // device memory used for the output array for strict memcopy kernel

	cl_device_type deviceType = CL_DEVICE_TYPE_GPU; // make the GPU the default device type
    
    int workgroupSize = 0;		 // workgroup size / number of work items per wavefront
	int registerCount = 16;		 // registers allocated in kernels for memcopy operations
	int dumbyRegisterCount = 16; // registers allocated, but not used for memcopies
	int dataItemCount = 0;		 // total number of items (type float,float2,4) to copy to/from OpenCL device
	int dataItemCountEnd = 0;		 // total number of items (type float,float2,4) to copy to/from OpenCL device
	int ldsPasses = 1;			 // number of 'passes' copying data to/from LDS
	clDataType * clType; 			 // default float type to use
	bool useBarrier = true;			// include memory barrier in kernels
	bool memcpyOnly = false;	// if true, creates strict memcopy kernels, not registers allocated (in CL code)
	bool writeOnly = false;		// only perform write operations
	bool readOnly = false;		// only perform read operations.
	bool bDisableOptimization = false; // disable OpenCL compiler optimizations if true
	bool bDoublePrecision = false;
	bool bZeroMemcopy = false;	// if true, host memory is used by GPU

	cl_ulong start = 0; // profiling start and end times
	cl_ulong end = 0;

	clFloat lFloat;
	clDouble lDouble;
	clType =  &lFloat; // float is default

	try
	{
		// Declare the supported options.
		po::options_description desc( "clMemcpy client command line options" );
		desc.add_options()
			( "help,h",			"produces this help message" )
			( "version,v",		"Print out build date/version" )
			( "gpu,g",			"Force instantiation of an OpenCL GPU device" )
			( "cpu,c",			"Force instantiation of an OpenCL CPU device" )
			( "float,f",		po::value< int >(), "Float type to use in kernels, 1,2,4,8,16 (default: float2)" )
			( "double,d",		po::value< int >(), "Use double type to use in kernels, 1,2,4 (default: double 1)" )
			( "regs,r",			po::value< int >( &registerCount )->default_value( 16 ),	"Specify number of registers to use in kernels (default: 16)" )
			( "dumbyRegs,q",	po::value< int >( &dumbyRegisterCount )->default_value( 0 ),	"Specify number 'dumby registers' to allocate in kernels" )
			( "memcpyOnly,m",   "Generate strict memcopy kernel (default: false)" )
			( "itemCount,i",	po::value< int >( &dataItemCount )->default_value( 0 ), "Number of items to transfer (default: max allocatable)" )
			( "itemCountEnd,j",	po::value< int >( &dataItemCountEnd )->default_value( 0 ), "End of item count, start at i go to j in powers of 2." )
			( "ldsPasses,l",	po::value< int >( &ldsPasses )->default_value( 1 ), "Number of 'passes' using LDS (default: 1, 0 = no LDS used)" )
			( "barrier,b",      po::value< bool >( &useBarrier )->default_value( true ), "Include memory barrier in kernel" )
			( "writeOnly,x",      "Write only kernels (default:  false)" )
			( "readOnly,y",       "Read only kernels (default: false" )
			( "disableOptimization,n",       "Disable OpenCL compiler optimizations (default: false" )
			( "zeroMemcopy,z",       "Use zero memcopy kernels, only valid on APUs (default 0)" )
			( "workgroupSize,w",	po::value< int >( &workgroupSize )->default_value( 64 ), "Workgroup size (default 64)" )
			;

		po::variables_map vm;
		po::store( po::parse_command_line( argc, argv, desc ), vm );
		po::notify( vm );

		stringstream str;

		if( vm.count( "version" ) )
		{
			str << "clMemcopy version: " << __DATE__ << " " << __TIME__ <<std::endl;
			std::cout << str.str();
			str.str() = "";
			return 0;
		}

		if( vm.count( "help" ) )
		{
			//	This needs to be 'cout' as program-options does not support wcout yet
			std::cout << desc << std::endl;
			return 0;
		}

		size_t mutex = ((vm.count( "gpu" ) > 0) ? 1 : 0)
			| ((vm.count( "cpu" ) > 0) ? 2 : 0);
		if ((mutex & (mutex-1)) != 0) {
			str << "You have selected mutually-exclusive OpenCL device options:" << std::endl;
			if (vm.count ( "gpu" )  > 0) str << "    gpu,g   Force instantiation of an OpenCL GPU device" << std::endl;
			if (vm.count ( "cpu" )  > 0) str << "    cpu,c   Force instantiation of an OpenCL CPU device" << std::endl;
			{	
				std::cout << str.str();
				return 1;
			}
		}

		mutex = ((vm.count( "writeOnly" ) > 0) ? 1 : 0)
			| ((vm.count( "readOnly" ) > 0) ? 2 : 0);
		if ((mutex & (mutex-1)) != 0) {
			str << "You have selected mutually-exclusive OpenCL device options:" << std::endl;
			if (vm.count ( "writeOnly" )  > 0) str << "    writeOnly,x   Generate write only kernels" << std::endl;
			if (vm.count ( "readOnly" )  > 0) str << "    readOnly,y   Generate read only kernels" << std::endl;
			{	
				std::cout << str.str();
				return 1;
			}
		}

		if( vm.count( "gpu" ) )
		{
			deviceType	= CL_DEVICE_TYPE_GPU;
		}
		
		if( vm.count( "cpu" ) )
		{
			deviceType	= CL_DEVICE_TYPE_CPU;
		}

		if( vm.count( "writeOnly" ) )
		{
			writeOnly = true;
		}
		
		if( vm.count( "readOnly" ) )
		{
			readOnly = true;
		}

		if( vm.count( "zeroMemcopy" ) )
		{
			bZeroMemcopy = true;
		}

		int typeCount = 0;

		if( vm.count( "float" ) )
		{
			if(!clType->setSize(vm["float"].as<int>()))
			{
				std::cout << "Float (float,-f) type must be 1,2,4,8, or 16.";
				return 1;
			}
			typeCount ++;
		}

		if( vm.count( "double" ) )
		{
			clType =  &lDouble;
			if(!clType->setSize(vm["double"].as<int>()))
			{
				std::cout << "Double (double,-d) type must be 1, or 2.";
				return 1;
			}
			bDoublePrecision = true;
			typeCount ++;
		}

		if(typeCount > 1)
		{
			std::cout << "Only one register type may be specified (Float,Double).";
			return 1;
		}

		if( vm.count( "memcpyOnly" ) )
		{
			memcpyOnly = true;
			registerCount = 1;
		}

		if( vm.count( "disableOptimization" ) )
		{
			bDisableOptimization = true;
		}

		if(workgroupSize < 1)
		{
			printf("Error: workgroup size can not be 0");
			return 1;
		}

		// if the register count is < 1, it's a pure memcpy kernel
		if(registerCount < 1)
		{
			registerCount = 1;
			memcpyOnly = true;
		}

	}
	catch( std::exception& e )
	{
		std::cout << "clMemcopy error condition reported:" << std::endl << e.what() << std::endl;
		return 1;
	}

	// enumerate platforms to see if anything is available.
	//
	err=clGetPlatformIDs(1, &platform, &platforms);
	if (err != CL_SUCCESS)
    {
        printf("Error: Failed to get a platform.!\n");
        return EXIT_FAILURE;
    }
  
    // Connect to a compute device
    //
    err = clGetDeviceIDs(platform, deviceType, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
  
    // Create a compute context 
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    if (!commands)
    {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

	// find how much global memory can safely be allocated
	//
	cl_ulong maxMemAlloc = 0;
	err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE,sizeof(cl_ulong), &maxMemAlloc, NULL);
	
	if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read MAX_MEM_ALLOC_SIZE from device!\n");
        return EXIT_FAILURE;
    }

	// find how much local memory can safely be allocated
	//
	cl_ulong maxLocalMemAlloc = 0;
	err = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE,sizeof(cl_ulong), &maxLocalMemAlloc, NULL);
	
	if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read CL_DEVICE_LOCAL_MEM_SIZE from device!\n");
        return EXIT_FAILURE;
    }
	
	// check if double precision is supported
	//	If the user specifies double precision, check that the device supports double precision first
	if( bDoublePrecision )
	{
		std::vector< cl_device_id > dev;
		dev.push_back(device_id);

		bool retAmdFp64 = checkDevExt( "cl_amd_fp64", dev );
		if( retAmdFp64 != true )
		{
			//	If AMD's extention is not supported, check for Khronos extention
			bool retKhrFp64 = checkDevExt( "cl_khr_fp64", dev );
			if( retKhrFp64 != true )
			{
				 printf("Error: Device %d does not support double precission\n", device_id);
				return EXIT_FAILURE;
			}
		}
	}

	
	do
	{

		// generate a kernel
		//
		stringstream kernelSource;

		GenerateMemcpyKernel(kernelSource, registerCount, dumbyRegisterCount, workgroupSize, clType, useBarrier, ldsPasses, dataItemCount, writeOnly, readOnly, memcpyOnly, bDoublePrecision);
	
		if( !dataItemCountEnd ) // 
		{	
			printf("\n%s\n", kernelSource.str().c_str());
		}
		// calculate how many data items we want to move, float1,2,4
		//
		if(dataItemCount == 0)
		{
			if( memcpyOnly )
			{
				maxMemAlloc /= 2;  // need two buffers
			}
			dataItemCount = (int)(maxMemAlloc / (clType->getSize()));
			dataItemCount /= registerCount * workgroupSize;
			dataItemCount *= registerCount * workgroupSize;
		}
	

		// Fill our data set with random float values
		//
    
		void* data = aligned_malloc(clType->getSize() * dataItemCount, 256);              // original data set given to device
		if(data == NULL)
		{
			printf("Error: Failed allcating host data buffer!\n");
			return EXIT_FAILURE;
		}

		srand ( (unsigned int) time(NULL) );
		for(int i = 0; i < dataItemCount * clType->getSize(); i++)
		{  
			*((char *)data + i) = rand() / (char)RAND_MAX;
		}

		// Create the compute program from the source buffer
		//
		std::string stringKern = kernelSource.str();
		const char *charKern = stringKern.c_str();
		program = clCreateProgramWithSource(context, 1, (const char **) &charKern, NULL, &err);
		if (!program)
		{
			printf("Error: Failed to create compute program!\n");
			return EXIT_FAILURE;
		}

		// Build the program executable
		//
		if(bDisableOptimization)
		{
			err = clBuildProgram(program, 0, NULL, "-g -cl-opt-disable", NULL, NULL);
		}
		else
		{
			err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
		}

		if (err != CL_SUCCESS)
		{
			size_t len;
			char buffer[2048];

			printf("Error: Failed to build program executable!\n");
			clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
			printf("%s\n", buffer);
			exit(1);
		}

		// Create the compute kernel in the program we wish to run
		//
		kernel = clCreateKernel(program, "memcpy", &err);
		if (!kernel || err != CL_SUCCESS)
		{
			printf("Error: Failed to create compute kernel!\n");
			exit(1);
		}
		/*
		//	Discover and load the timer module if present
		void* timerLibHandle = LoadSharedLibrary( "lib", "clAmdFft.StatTimer", false );
		if( timerLibHandle == NULL )
		{
			terr << _T( "Could not find the external timing library; timings disabled" ) << std::endl;
		}

	
		//	Timer module discovered and loaded successfully
		//	Initialize function pointers to call into the shared module
		PFGETSTATTIMER get_timer = reinterpret_cast< PFGETSTATTIMER > ( LoadFunctionAddr( timerLibHandle, "getStatTimer" ) );

		//	Create and initialize our timer class, if the external timer shared library loaded
		baseStatTimer* timer = NULL;
		*/
		size_t	writeTimer,readTimer,executeTimer = 0;
		StatisticalTimer &timer = StatisticalTimer::getInstance();
	

	
		//	timer->setNormalize( true );
			timer.Reserve( 3, 1 );

			writeTimer	= timer.getUniqueID( "write", 0 );
			readTimer	= timer.getUniqueID( "read", 1 );
			executeTimer	= timer.getUniqueID( "execute", 2);
		
	
		// Create the input and output arrays in device memory for our calculation
		//

		cl_mem_flags memFlags = CL_MEM_READ_ONLY;
		void *hostPtr = NULL;
		void *hostPtrOut = NULL; // use to map point to output buffer for memcopy only kernels

		// this option will only work on APUs same physical memory is used by host and device
		if(bZeroMemcopy)
		{
			memFlags |= CL_MEM_ALLOC_HOST_PTR;
		//	memFlags |= CL_MEM_USE_PERSISTENT_MEM_AMD;
		}
	
		input = clCreateBuffer(context,  memFlags,  clType->getSize() * dataItemCount, NULL, NULL);
	
		if (!input)
		{
			printf("Error: Failed to allocate device memory!!\n");
			exit(1);
		}

		if(memcpyOnly)
		{
			output = clCreateBuffer(context,  memFlags,  clType->getSize() * dataItemCount, NULL, NULL);
			 if (!output)
			{
				printf("Error: Failed to allocate device memory!\n");
				exit(1);
			}
		}



		if( bZeroMemcopy )
		{
	//		err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, clType->getSize() * dataItemCount, data, 0, NULL,  NULL); //test
	//		if( err != CL_SUCCESS )
	//		{
	//		    printf("Error: Failed to copy host buffer to cl buffer (zero memcopy)!\n");
	//			return EXIT_FAILURE;
	//		}

			hostPtr = clEnqueueMapBuffer(commands, input, CL_TRUE, CL_MAP_WRITE, 0,  clType->getSize() * dataItemCount, 0, NULL, NULL, &err);
			if( err != CL_SUCCESS )
			{
				printf("Error: Failed to map host pointer to zero memcopy buffer!\n");
				return EXIT_FAILURE;
			}
	
			if(memcpyOnly)
			{
				hostPtrOut = clEnqueueMapBuffer(commands, output, CL_TRUE, CL_MAP_WRITE, 0,  clType->getSize() * dataItemCount, 0, NULL, NULL, &err);
				if( err != CL_SUCCESS )
				{
					printf("Error: Failed to map host pointer to zero memcopy buffer!\n");
					return EXIT_FAILURE;
				}
			}

			// start timing writing to buffer (device or zero mem copy)
			timer.Start(writeTimer);
		
			memcpy( hostPtr, data, clType->getSize() * dataItemCount);
		}
		else
		{
			// start timing writing to buffer (device or zero mem copy)
			timer.Start(writeTimer);
		}
		
		// Write our data set into the input array in device memory 
		//
		if( !bZeroMemcopy )
		{
			cl_event eventKernelTiming; // for timing

			err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, clType->getSize() * dataItemCount, data, 0, NULL,  &eventKernelTiming);
			if (err != CL_SUCCESS)
			{
				printf("Error: Failed to write to source array!\n");
				exit(1);
			}
			clFinish(commands);

		
			clGetEventProfilingInfo(eventKernelTiming, CL_PROFILING_COMMAND_START,
											   sizeof(start), &start, NULL);

			clGetEventProfilingInfo(eventKernelTiming, CL_PROFILING_COMMAND_END,
											   sizeof(end), &end, NULL);

			cl_ulong diff = end-start;
		
			if( !dataItemCountEnd)
			{
			printf("\nbuffer write GPU timer %lld",diff);
			}
		}
		
		timer.Stop(writeTimer);

 		// Set the arguments to our compute kernel
		//
 
		err = 0;
		err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);

		if(memcpyOnly)
		{
			err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
		}

	//    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
	 //   err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to set kernel arguments! %d\n", err);
			exit(1);
		}
	

		// Get the maximum work group size for executing the kernel on the device
		//
		err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
		if (err != CL_SUCCESS)
		{
			printf("Error: Failed to retrieve kernel work group info! %d\n", err);
			exit(1);
		}

		// Execute the kernel over the entire range of our 1d input data set
		// using the maximum number of work group items for this device
		//
		if(!memcpyOnly)
		{
			global = dataItemCount / registerCount;
		}
		else
		{
			global = dataItemCount;
		}

		if(workgroupSize < local && workgroupSize != 0)
		{
			local = workgroupSize;
		}
		if(workgroupSize > local)
		{
			printf("Error: Max supported workgroup size is %d, requested was %d", (unsigned int)local, workgroupSize);
			exit(1);
		}

		cl_event eventKernelTiming; // for timing

		timer.Start(executeTimer); // measure kernel execution time

		err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, &eventKernelTiming);

		// Wait for the command commands to get serviced before reading back results
		//
   
		//clWaitForEvents(1, &eventGlobal);
		clFinish(commands);

		timer.Stop(executeTimer); // end of kernel execution

		clGetEventProfilingInfo(eventKernelTiming, CL_PROFILING_COMMAND_START,
										   sizeof(start), &start, NULL);

		clGetEventProfilingInfo(eventKernelTiming, CL_PROFILING_COMMAND_END,
										   sizeof(end), &end, NULL);
    
		clReleaseEvent(eventKernelTiming);

		if (err)
		{
			printf("Error: Failed to execute kernel!\n");
			return EXIT_FAILURE;
		}

		timer.Start(readTimer); // measure time to read back from memory
	
		// Read back the results from the device to verify the output
		//
    
		if( !bZeroMemcopy )
		{
			err = clEnqueueReadBuffer( commands, input, CL_TRUE, 0, clType->getSize() * dataItemCount, data, 0, NULL, NULL);  
			if (err != CL_SUCCESS)
			{
				printf("Error: Failed to read output array! %d\n", err);
				exit(1);
			}
		}

		timer.Stop(readTimer);
	      
		cl_ulong time = end - start; /* Convert nanoseconds to msecs */
    	
		// Calculate gflops
	
		cl_ulong dataTransferred = dataItemCount * clType->getSize();

		int multiplier = 2;
		if(readOnly || writeOnly)
		{
			multiplier = 1;
		}

		int fftlen = (int)(local * registerCount * clType->getSize()/clType->getTypeSize()) / 2;
		double gflops	= (global/local) * 5 * fftlen * ( log( static_cast< double >( fftlen ) ) / log( 2.0 ) ) / time;
		double MBps = (double)(multiplier * (double)(dataTransferred) / time);

		if( !dataItemCountEnd )
		{
			printf("\nTicks= %ld\nTransfer= %ld bytes\nbandwidth= %lf GB/S", time , dataTransferred, MBps);
		
			if(!memcpyOnly)
			{
				printf("\nType = %s\nfftlen=%d\nGflops %lf\n",clType->getName().c_str(), fftlen, gflops);
			}
		}
		// Shutdown and cleanup
		//
    
		if(bZeroMemcopy)
		{
			err = clEnqueueUnmapMemObject(commands, input, hostPtr, 0, 0, 0);
			if(memcpyOnly)
			{
				err = clEnqueueUnmapMemObject(commands, output, hostPtrOut, 0, 0, 0);
			}

			if(err != CL_SUCCESS)
			{
				printf("Error: Failed to unmap memory objects!\n");
				return EXIT_FAILURE;
			}
		}
	
		clReleaseMemObject(input);
		if( memcpyOnly )
		{
			clReleaseMemObject(output);
		}
	
		if( data )
		{
			aligned_free( data );
		}
	
		printf("\n%10ld,\t%f,\t%f,\t%f,\t%f,",dataTransferred, timer.getMinimumTime(writeTimer) ,timer.getMinimumTime(executeTimer),timer.getMinimumTime(readTimer),timer.getMinimumTime(writeTimer)  + timer.getMinimumTime(executeTimer) + timer.getMinimumTime(readTimer) );

		clReleaseProgram(program);
		clReleaseKernel(kernel);

		dataItemCount*= 2;
} while(dataItemCount <= dataItemCountEnd);


    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}

