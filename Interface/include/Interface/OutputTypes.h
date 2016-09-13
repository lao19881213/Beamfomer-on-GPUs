#ifndef LOFAR_RTCP_INTERFACE_OUTPUT_TYPES_H
#define LOFAR_RTCP_INTERFACE_OUTPUT_TYPES_H

namespace LOFAR {
namespace RTCP {

enum OutputType
{
  CORRELATED_DATA = 1,
  BEAM_FORMED_DATA,
  TRIGGER_DATA,

  // define LAST and FIRST in the enum to make them valid values within the
  // allocated range for the enum (=minimal number of bits to store all values)
  LAST_OUTPUT_TYPE,
  FIRST_OUTPUT_TYPE = 1
};

inline OutputType operator ++ (OutputType &outputType) // prefix ++
{
  return (outputType = static_cast<OutputType>(outputType + 1));
}


inline OutputType operator ++ (OutputType &outputType, int) // postfix ++
{
  return static_cast<OutputType>((outputType = static_cast<OutputType>(outputType + 1)) - 1);
}

} // namespace RTCP
} // namespace LOFAR

#endif
