#ifndef LOFAR_CNPROC_TRIGGER_H
#define LOFAR_CNPROC_TRIGGER_H


#include <Interface/TriggerData.h>


namespace LOFAR {
namespace RTCP {


class Trigger
{
  public:
    void compute(TriggerData *triggerData) { triggerData->trigger = false; }

  private:
};


} // namespace RTCP
} // namespace LOFAR

#endif
