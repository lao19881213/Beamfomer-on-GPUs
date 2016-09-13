#ifndef LOFAR_CNPROC_RING_H
#define LOFAR_CNPROC_RING_H

namespace LOFAR {
namespace RTCP {

#include <vector>

/*
 * Ring handles a ring of subbands or beams to be processed by a compute node.
 *
 * pset:        the pset index of this node
 * numperpset:  the number of subbands or beams to be processed per pset
 * core:        the core index of this node
 * numcores:    the number of cores per pset (that will be used)
 */

class Ring
{
  public:
    Ring(unsigned pset, unsigned numperpset, unsigned core, unsigned numcores);

    // emulate a cast to (unsigned) for ease of use, and add a few shorthands
    operator unsigned () const;

    void next();

    // returns the relative core number within this pset to process this 'second' of data
    unsigned relative() const;

    // is the current element the last to be processed for this 'second' of data?
    bool isLast() const;

    // list the elements to process
    std::vector<unsigned> list() const;

    const unsigned pset;
    const unsigned core;

    const unsigned numperpset;
    const unsigned numcores;

    const unsigned first;
    const unsigned last;
    const unsigned increment;

    void skipFirstBlocks(size_t n);

  private:
    unsigned current;
};


inline Ring::Ring(unsigned pset, unsigned numperpset, unsigned core, unsigned numcores)
:
  pset(pset),
  core(core),
  numperpset(numperpset),
  numcores(numcores),
  first(pset * numperpset),
  last((pset + 1) * numperpset),
  increment(numcores % numperpset),
  current(first + core % numperpset)
{
}


inline Ring::operator unsigned () const
{
  return current;
}


inline void Ring::next()
{
  if ((current += increment) >= last)
    current -= last - first;
}


inline unsigned Ring::relative() const
{
  return current - first;
}


inline std::vector<unsigned> Ring::list() const
{
  std::vector<unsigned> list;

  for (Ring copy = *this; list.empty() || copy.current != current; copy.next())
    list.push_back(copy);

  return list;
}


inline bool Ring::isLast() const
{
  return current + increment >= last || numcores >= numperpset;
}

inline void Ring::skipFirstBlocks(size_t n)
{
  // TODO: extend towards skipping from any position

  for( unsigned b = 0, activecore = 0; b < n; b++ ) {
    for (unsigned sb = 0; sb < numperpset; sb++) {
      if (activecore == core)
        next();
      
      if (++activecore == numcores)
        activecore = 0;
    }
  }
}

} // namespace RTCP
} // namespace LOFAR

#endif
