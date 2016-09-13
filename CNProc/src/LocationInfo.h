#ifndef LOFAR_CNPROC_LOCATIONINFO_H
#define LOFAR_CNPROC_LOCATIONINFO_H

#include <vector>

#if defined HAVE_MPI
// we do not need mpi.h here, but including it after bgp_personality.h leads
// to compilation errors
#define MPICH_IGNORE_CXX_SEEK
#include <mpi.h>
#endif // HAVE_MPI

#if defined HAVE_BGP
#include <common/bgp_personality.h>
#endif // HAVE_BGP


namespace LOFAR {
namespace RTCP {

class LocationInfo
{
  public:
	     LocationInfo();

    unsigned remapOnTree(unsigned pset, unsigned core) const;

    void     print() const;

    unsigned rank() const;
    unsigned nrNodes() const;
    unsigned psetNumber() const;
    unsigned rankInPset() const;
    unsigned nrPsets() const;
    unsigned psetSize() const;

  private:
#if defined HAVE_BGP
    void getPersonality();

    _BGP_Personality_t    itsPersonality;
    std::vector<unsigned> itsPsetNumbers;
#endif

    unsigned              itsPsetNumber, itsRankInPset, itsNrPsets, itsPsetSize;
    unsigned              itsRank, itsNrNodes;
};


inline unsigned LocationInfo::rank() const
{
  return itsRank;
}


inline unsigned LocationInfo::nrNodes() const
{
  return itsNrNodes;
}


inline unsigned LocationInfo::psetNumber() const
{
  return itsPsetNumber;
}


inline unsigned LocationInfo::rankInPset() const
{
  return itsRankInPset;
}


inline unsigned LocationInfo::nrPsets() const
{
  return itsNrPsets;
}


inline unsigned LocationInfo::psetSize() const
{
  return itsPsetSize;
}

} // namespace RTCP
} // namespace LOFAR

#endif
