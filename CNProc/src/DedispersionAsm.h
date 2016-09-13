#ifndef LOFAR_CNPROC_DEDISPERSIONASM_H
#define LOFAR_CNPROC_DEDISPERSIONASM_H

#if defined HAVE_BGP

#include <Common/lofar_complex.h>

namespace LOFAR {
namespace RTCP {

extern "C"
{
  void _apply_chirp(fcomplex *xBuffer, fcomplex *yBuffer, const fcomplex *chirp, unsigned count);
}

} // namespace LOFAR::RTCP
} // namespace LOFAR

#endif

#endif
