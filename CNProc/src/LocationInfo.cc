#include <lofar_config.h>

#include <LocationInfo.h>

#include <Interface/CN_Mapping.h>
#include <Interface/PrintVector.h>
#include <Interface/Exceptions.h>

#include <Common/LofarLogger.h>

#if defined HAVE_BGP
#include <common/bgp_personality_inlines.h>
#include <spi/kernel_interface.h>
#endif


#include <iostream>
#include <boost/lexical_cast.hpp>


namespace LOFAR {
namespace RTCP {

LocationInfo::LocationInfo()
{
#if defined HAVE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, reinterpret_cast<int *>(&itsRank));
  MPI_Comm_size(MPI_COMM_WORLD, reinterpret_cast<int *>(&itsNrNodes));
#else
  itsRank    = 0;
  itsNrNodes = 1;
#endif

#if defined HAVE_BGP
  getPersonality();
#else
  const char *nrPsetsStr  = getenv("NR_PSETS");
  const char *psetSizeStr = getenv("PSET_SIZE");

  if (nrPsetsStr == 0 || psetSizeStr == 0)
    THROW(CNProcException, "environment variables NR_PSETS and PSET_SIZE must be defined");

  itsNrPsets    = boost::lexical_cast<unsigned>(nrPsetsStr);
  itsPsetSize   = boost::lexical_cast<unsigned>(psetSizeStr);

#if defined CLUSTER_SCHEDULING
  itsPsetNumber = 0;
  itsRankInPset = itsRank;
#else
  itsPsetNumber = itsRank % itsNrPsets;
  itsRankInPset = itsRank / itsNrPsets;
#endif

#endif

  ASSERT( itsPsetNumber < itsNrPsets );
  ASSERT( itsRankInPset < itsPsetSize );
}


#if defined HAVE_BGP

void LocationInfo::getPersonality()
{
  // allow this only once due to the MPI_Bcast that needs to be synced.
  static bool initialised = false;

  if (initialised) {
    THROW(CNProcException,"LocationInfo::getPersonality called for a second time");
  }

  initialised = true;

  if (Kernel_GetPersonality(&itsPersonality, sizeof itsPersonality) != 0) {
    LOG_FATAL("could not get personality");
    exit(1);
  }

  itsPsetNumbers.resize(itsNrNodes);
  itsPsetNumber = BGP_Personality_psetNum(&itsPersonality);
  itsPsetNumbers[itsRank] = itsPsetNumber;

#if defined HAVE_MPI
  for (unsigned core = 0; core < itsNrNodes; core ++)
    MPI_Bcast(&itsPsetNumbers[core], 1, MPI_INT, core, MPI_COMM_WORLD);
#endif

  itsRankInPset = 0;

  for (unsigned rank = 0; rank < itsRank; rank ++)
    if (itsPsetNumbers[rank] == itsPsetNumber)
      ++ itsRankInPset;

  itsNrPsets  = *std::max_element(itsPsetNumbers.begin(), itsPsetNumbers.end()) + 1;
  itsPsetSize = itsNrNodes / itsNrPsets;
}


unsigned LocationInfo::remapOnTree(unsigned pset, unsigned core) const
{
  core = CN_Mapping::mapCoreOnPset(core, pset);

  for (unsigned rank = 0;; rank ++)
    if (itsPsetNumbers[rank] == pset && core -- == 0)
      return rank;
}


void LocationInfo::print() const
{
  LOG_DEBUG_STR("topology = ("
	<< BGP_Personality_xSize(&itsPersonality) << ','
	<< BGP_Personality_ySize(&itsPersonality) << ','
	<< BGP_Personality_zSize(&itsPersonality) << "), torus wraparound = ("
	<< (BGP_Personality_isTorusX(&itsPersonality) ? 'T' : 'F') << ','
	<< (BGP_Personality_isTorusY(&itsPersonality) ? 'T' : 'F') << ','
	<< (BGP_Personality_isTorusZ(&itsPersonality) ? 'T' : 'F') << ')');

  std::vector<std::vector<unsigned> > cores(BGP_Personality_numIONodes(&itsPersonality));

  for (unsigned rank = 0; rank < itsPsetNumbers.size(); rank ++)
    cores[itsPsetNumbers[rank]].push_back(rank);

  for (unsigned pset = 0; pset < BGP_Personality_numIONodes(&itsPersonality); pset ++)
    LOG_DEBUG_STR("pset " << pset << " contains cores " << cores[pset]);
}

#else

unsigned LocationInfo::remapOnTree(unsigned pset, unsigned core) const
{
  return pset + itsNrPsets * core;
}

void LocationInfo::print() const
{
}

#endif

} // namespace RTCP
} // namespace LOFAR
