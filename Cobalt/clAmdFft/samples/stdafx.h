////////////////////////////////////////////
//	Copyright (C) 2010 Advanced Micro Devices, Inc. All Rights Reserved.
////////////////////////////////////////////

// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <complex>
#include <valarray>
#include <stdarg.h>
#if defined( _WIN32 )
	#define NOMINMAX
	#define WIN32_LEAN_AND_MEAN			// Exclude rarely-used stuff from Windows headers

	#include <tchar.h>
	#include <windows.h>
#endif

