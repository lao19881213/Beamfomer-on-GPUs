////////////////////////////////////////////
//	Copyright (C) 2010,2011 Advanced Micro Devices, Inc. All Rights Reserved.
////////////////////////////////////////////

#pragma once
#if !defined( AMD_OPENCL_H )
#define AMD_OPENCL_H
#include <memory>
#include <stdexcept>
#include "amd-unicode.h"

//	Creating a portable defintion of countof
#if defined( _WIN32 )
	#define countOf _countof
#else
	#define countOf( arr ) ( sizeof( arr ) / sizeof( arr[ 0 ] ) )
#endif

/*
 * \brief OpenCL related initialization (ripped from AMD stream sample code )
 *        Create Context, Device list
 *        Load CL file, compile, link CL source 
 *		  Build program and kernel objects
 */
std::vector< cl_device_id > initializeCL( cl_device_type deviceType,
										  cl_uint deviceGpuList,
										  cl_context& context,
										  bool printclInfo );

/*
 * \brief OpenCL memory buffer creation
 */
int createOpenCLMemoryBuffer(
		cl_context& context,
		const size_t bufferSizeBytes,
		const cl_uint numBuffers,
		cl_mem buffer[],
		cl_mem_flags accessibility
		);

/*
 * \brief OpenCL command queue creation (ripped from AMD stream sample code )
 *        Create Command Queue
 *        Create OpenCL memory buffer objects
 */
void createOpenCLCommandQueue( cl_context& context,
							   cl_uint commandQueueFlags,
							   cl_command_queue& commandQueue,
							   std::vector< cl_device_id > devices,
							   const size_t bufferSizeBytesIn,
							   const cl_uint numBuffersIn,
							   cl_mem clMemBufferIn[],
							   const size_t bufferSizeBytesOut,
							   const cl_uint numBuffersOut,
							   cl_mem clMemBufferOut[] );

/*
 * \brief release OpenCL memory buffer
 */
int releaseOpenCLMemBuffer( const cl_uint numBuffers, cl_mem buffer[] );

std::string prettyPrintclFFTStatus( const cl_int& status );

//	This is used to either wrap an OpenCL function call, or to explicitly check a variable for an OpenCL error condition.
//	If an error occurs, we throw.  
//	Note: std::runtime_error does not take unicode strings as input, so only strings supported
inline cl_int OpenCL_V_Throw ( cl_int res, const std::string& msg, size_t lineno )
{ 
	switch( res )
	{
		case	CL_SUCCESS:		/**< No error */
			break;
		default:
		{
			std::stringstream tmp;
			tmp << "OPENCL_V_THROWERROR< ";
			tmp << prettyPrintclFFTStatus( res );
			tmp << " > (";
			tmp << lineno;
			tmp << "): ";
			tmp << msg;
			std::string errorm (tmp.str());
			std::cout << errorm<< std::endl;
			throw	std::runtime_error( errorm );
		}
	}

	return	res;
}
#define OPENCL_V_THROW(_status,_message) OpenCL_V_Throw (_status, _message, __LINE__)

/*
 * \brief Release OpenCL resources (Context, Memory etc.) (ripped from AMD stream sample code )
 */
int cleanupCL( cl_context* context, cl_command_queue* commandQueue, const cl_uint numBuffersIn, cl_mem inputBuffer[], const cl_uint numBuffersOut, cl_mem outputBuffer[], cl_event* outEvent );

#endif
