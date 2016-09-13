#include "SampleBuffer.h"

#include "BufferSettings.h"
#include "SharedMemory.h"

namespace LOFAR
{
  namespace Cobalt
  {
    void removeSampleBuffers( const BufferSettings &settings )
    {
      // Remove the provided dataKey, as it could be a custom setting
      SharedMemoryArena::remove(settings.dataKey);
    }
  }
}

