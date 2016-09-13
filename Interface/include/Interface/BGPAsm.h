#ifndef LOFAR_INTERFACE_BGPASM_H
#define LOFAR_INTERFACE_BGPASM_H

#if defined HAVE_BGP

#include <cstring>

namespace LOFAR {
namespace RTCP {

extern "C" {

// all float * must be aligned to 8 bytes

void _add_2_single_precision_vectors(
  /* r3 */ float *dst,
  /* r4 */ const float *src1,
  /* r5 */ const float *src2,
  /* r6 */ unsigned count /* non-zero; multiple of 16 */
);  

void _add_3_single_precision_vectors(
  /* r3 */ float *dst,
  /* r4 */ const float *src1,
  /* r5 */ const float *src2,
  /* r6 */ const float *src3,
  /* r7 */ unsigned count /* non-zero; multiple of 16 */
);  

void _add_4_single_precision_vectors(
  /* r3 */ float *dst,
  /* r4 */ const float *src1,
  /* r5 */ const float *src2,
  /* r6 */ const float *src3,
  /* r7 */ const float *src4,
  /* r8 */ unsigned count /* non-zero; multiple of 16 */
);  

void _add_5_single_precision_vectors(
  /* r3 */ float *dst,
  /* r4 */ const float *src1,
  /* r5 */ const float *src2,
  /* r6 */ const float *src3,
  /* r7 */ const float *src4,
  /* r8 */ const float *src5,
  /* r9 */ unsigned count /* non-zero; multiple of 16 */
);  

void _add_6_single_precision_vectors(
  /* r3 */ float *dst,
  /* r4 */ const float *src1,
  /* r5 */ const float *src2,
  /* r6 */ const float *src3,
  /* r7 */ const float *src4,
  /* r8 */ const float *src5,
  /* r9 */ const float *src6,
  /* r10 */ unsigned count /* non-zero; multiple of 16 */
);  

} // extern "C"

// Similar functions that do not need or have an ASM version

// defined just to aid the use of macros
static inline void _add_1_single_precision_vectors(
  float *dst,
  const float *src1,
  unsigned count /* non-zero; multiple of 16 */
) {
  // nothing to add, so just copy the values
  memcpy( dst, src1, count * sizeof(float) );
}

static inline void _add_7_single_precision_vectors(
  float *dst,
  const float *src1,
  const float *src2,
  const float *src3,
  const float *src4,
  const float *src5,
  const float *src6,
  const float *src7,
  unsigned count /* non-zero; multiple of 16 */
) {
  _add_4_single_precision_vectors( dst, src1, src2, src3, src4, count );
  _add_4_single_precision_vectors( dst, dst,  src5, src6, src7, count );
}

} // namespace LOFAR::RTCP
} // namespace LOFAR

#endif

#endif
