#include <lofar_config.h>

#include "Interface/SparseSet.h"

#include <cassert>
#include <bitset>
#include <iostream>


#define BITSET_SIZE	4096


using namespace LOFAR;
using namespace std;

bool equal(const SparseSet<unsigned> &sset, const bitset<BITSET_SIZE> &bset)
{
  if (sset.count() != bset.count())
    return false;

  for (unsigned i = 0; i < BITSET_SIZE; i ++)
    if (sset.test(i) != bset.test(i))
      return false;

  return true;
}

int main(void)
{
  //SparseSet sset;
  //sset.include(7, 11).include(12, 15).include(17).include(20, 23);
  //std::cout << sset << '\n';

  for (unsigned i = 0; i < 100; i ++) {
    SparseSet<unsigned>	sset, sset_union;
    bitset<BITSET_SIZE> bset, bset_union;

    for (unsigned j = 0; j < 100; j ++) {
      unsigned first = (unsigned) (drand48() * (BITSET_SIZE - 100));
      unsigned last  = (unsigned) (drand48() * 100) + first + 1;

      if (drand48() > .4) {
	sset.include(first, last);

	for (unsigned k = first; k < last; k ++)
	  bset.set(k);
      } else {
	sset.exclude(first, last);

	for (unsigned k = first; k < last; k ++)
	  bset.reset(k);
      }

      assert(equal(sset, bset));

      if (drand48() < .1) {
	sset_union |= sset;
	bset_union |= bset;
	assert(equal(sset_union, bset_union));
	sset.reset();
	bset.reset();
      }
    }
  }

  for (int i = 0; i < 23; i ++) {
    {
      SparseSet<unsigned> sset;
      bitset<BITSET_SIZE> bset(0x00727780 >> i);
      sset.include(7, 11).include(12, 15).include(17).include(20, 23).exclude(0, i) -= i;
      assert(equal(sset, bset));
    }
    {
      SparseSet<unsigned> sset;
      bitset<BITSET_SIZE> bset(0x00727780ULL); bset <<= i;
      sset.include(7, 11).include(12, 15).include(17).include(20, 23) += i;
      assert(equal(sset, bset));
    }
  }

  return 0;
}
