#ifndef LOFAR_CNPROC_BEAMFORMERASM_H
#define LOFAR_CNPROC_BEAMFORMERASM_H

#if defined HAVE_BGP

#include <Interface/BGPAsm.h>
#include <cstring>

namespace LOFAR {
namespace RTCP {

extern "C" {

#if 0
void _beamform_3beams(
  /* r3 */ fcomplex *dst,
  /* r4 */ size_t dst_beam_stride,
  /* r5 */ unsigned nr_stations,
  /* r6 */ unsigned nr_times,
  /* r7 */ fcomplex *samples,
  /* r8 */ unsigned station_samples_stride,
  /* r9 */ fcomplex *weights,
  /* r10 */ unsigned station_weights_stride
);

void _beamform_1station_1beam(
  /* r3 */ fcomplex *complex_voltages,
  /* r4 */ const fcomplex *samples,
  /* r5 */ const fcomplex *weight,
  /* r6 */ unsigned nr_times
);

void _beamform_2stations_1beam(
  /* r3 */ fcomplex *complex_voltages,
  /* r4 */ const fcomplex *samples,
  /* r5 */ unsigned samples_stride,
  /* r6 */ const fcomplex *weight,
  /* r7 */ unsigned weights_stride,
  /* r8 */ unsigned nr_times
);
#endif

void *_beamform_3stations_6beams(
  /* r3 */ fcomplex *complex_voltages,
  /* r4 */ unsigned complex_voltages_stride,
  /* r5 */ const fcomplex *samples,
  /* r6 */ unsigned samples_stride,
  /* r7 */ const fcomplex *weights,
  /* r8 */ unsigned weights_stride,
  /* r9 */ unsigned nr_times,
  /* r10 */ bool first_time // if !first_time, then add to complex_voltages
);

void *_beamform_up_to_6_stations_and_3_beams(
  /* r3 */ fcomplex *complex_voltages,
  /* r4 */ unsigned complex_voltages_stride,
  /* r5 */ const fcomplex *samples,
  /* r6 */ unsigned samples_stride,
  /* r7 */ const fcomplex *weights,
  /* r8 */ unsigned weights_stride,
  /* r9 */ unsigned nr_times,
  /* r10 */ bool first_time, // if !first_time, then add to complex_voltages
  /* 8(r1) */ unsigned nr_stations, // 1-6
  /* 12(r1) */ unsigned nr_beams // 1-3
);

#if 0
void _beamform_4stations_3beams(
  /* r3 */ fcomplex *complex_voltages,
  /* r4 */ unsigned complex_voltages_stride,
  /* r5 */ const fcomplex *samples,
  /* r6 */ unsigned samples_stride,
  /* r7 */ const fcomplex *weights,
  /* r8 */ unsigned weights_stride,
  /* r9 */ unsigned nr_times
);

void _beamform_6beams_2times(
  /* r3 */ fcomplex *dst,
  /* r4 */ size_t dst_beam_stride,
  /* r5 */ unsigned nr_stations,
  /* r6 */ fcomplex *samples,
  /* r7 */ unsigned station_samples_stride,
  /* r8 */ fcomplex *weights,
  /* r9 */ unsigned station_weights_stride
);
#endif

} // extern "C"

} // namespace LOFAR::RTCP
} // namespace LOFAR

#endif

#endif
