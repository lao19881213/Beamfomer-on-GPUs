#ifndef LOFAR_INTERFACE_FINAL_METADATA_H
#define LOFAR_INTERFACE_FINAL_METADATA_H

#include <Stream/Stream.h>
#include <string>
#include <vector>
#include <cstddef>
#include <ostream>

namespace LOFAR {
namespace RTCP {

class FinalMetaData
{
  public:
    struct BrokenRCU {
      std::string station; // CS001, etc
      std::string type;    // RCU, LBA, HBA
      size_t seqnr;        // RCU/antenna number
      std::string time;    // date time of break

      BrokenRCU() {}
      BrokenRCU(const std::string &station, const std::string &type, size_t seqnr, const std::string &time):
        station(station), type(type), seqnr(seqnr), time(time) {}

      bool operator==(const BrokenRCU &other) const {
        return station == other.station && type == other.type && seqnr == other.seqnr && time == other.time;
      }
    };

    std::vector<BrokenRCU>  brokenRCUsAtBegin, brokenRCUsDuring;

    void write(Stream &s);
    void read(Stream &s);
};

std::ostream& operator<<(std::ostream& os, const struct FinalMetaData::BrokenRCU &rcu);

std::ostream& operator<<(std::ostream& os, const FinalMetaData &finalMetaData);

} // namespace RTCP
} // namespace LOFAR

#endif
