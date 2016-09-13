#ifndef LOFAR_INTERFACE_TRIGGER_DATA_H
#define LOFAR_INTERFACE_TRIGGER_DATA_H

#include <Stream/Stream.h>
#include <Interface/StreamableData.h>


namespace LOFAR {
namespace RTCP {


class TriggerData: public StreamableData
{
  public:
    TriggerData() : trigger(false) {}

    bool trigger;

  protected:  
    virtual void readData(Stream *str) { str->read(&trigger, sizeof trigger); }
    virtual void writeData(Stream *str) { str->write(&trigger, sizeof trigger); }
};


} // namespace RTCP
} // namespace LOFAR

#endif
