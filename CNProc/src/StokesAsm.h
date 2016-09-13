//#  StokesAsm.h: header files for CN assembly
//#
//#  P.O.Box 2, 7990 AA Dwingeloo, The Netherlands, seg@astron.nl
//#
//#  This program is free software; you can redistribute it and/or modify
//#  it under the terms of the GNU General Public License as published by
//#  the Free Software Foundation; either version 2 of the License, or
//#  (at your option) any later version.
//#
//#  This program is distributed in the hope that it will be useful,
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//#  GNU General Public License for more details.
//#
//#  You should have received a copy of the GNU General Public License
//#  along with this program; if not, write to the Free Software
//#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//#
//#  $Id: STOKES_Asm.h 13808 2009-08-19 11:42:05Z romein $

#ifndef LOFAR_CNPROC_STOKES_ASM_H
#define LOFAR_CNPROC_STOKES_ASM_H

#if defined HAVE_BGP

#include <Common/lofar_complex.h>


namespace LOFAR {
namespace RTCP {

extern "C" {
  void _StokesI(float *I,
		const fcomplex (*in)[2], // XY
		unsigned length); // must be at least 8 and multiple of 4

  void _StokesIQUV(float *I, float *Q, float *U, float *V,
		   const fcomplex (*in)[2], // XY
		   unsigned length); // must be at least 8 and multiple of 4
}

} // namespace RTCP
} // namespace LOFAR

#endif
#endif
