#include <lofar_config.h>

#include <Interface/Align.h>
#include <Interface/Allocator.h>
#include <Interface/Exceptions.h>
#include <Common/NewHandler.h>
#include <Common/LofarLogger.h>

#include <malloc.h>


namespace LOFAR {
namespace RTCP {


MallocedArena::MallocedArena(size_t size, size_t alignment)
{
  itsBegin = heapAllocator.allocate(size, alignment);
  itsSize = size;
}


MallocedArena::~MallocedArena()
{
  heapAllocator.deallocate(itsBegin);
}


FixedArena::FixedArena(void *begin, size_t size)
{
  itsBegin = begin;
  itsSize  = size;
}


Allocator::~Allocator()
{
}


HeapAllocator::~HeapAllocator()
{
}


void *HeapAllocator::allocate(size_t size, size_t alignment)
{
  void *ptr;

  if (alignment == 1) {
    // no alignment requirements, so divert to malloc
    ptr = malloc(size);

    if (!ptr)
      THROW(BadAllocException,"HeapAllocator could not allocate " << size << " bytes");
  } else {
    ASSERT(alignment != 0);

#if _POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600
    // required by posix_memalign
    ASSERT(alignment % sizeof(void*) == 0);

    if (posix_memalign(&ptr, alignment, size) != 0)
      THROW(BadAllocException,"HeapAllocator could not allocate " << size << " bytes");
#else
    // required by memalign
    ASSERT(powerOfTwo(alignment));

    if ((ptr = memalign(alignment, size)) == 0)
      THROW(BadAllocException,"HeapAllocator could not allocate " << size << " bytes");
#endif
  }

  return ptr;
}


void HeapAllocator::deallocate(void *ptr)
{
  free(ptr);
}


HeapAllocator heapAllocator;


SparseSetAllocator::SparseSetAllocator(const Arena &arena)
{
  // mark full arena as free
  freeList.include(arena.begin(), (void *) ((char *) arena.begin() + arena.size()));
}


void *SparseSetAllocator::allocate(size_t size, size_t alignment)
{
  ScopedLock sl(mutex);

  // look for a free range large enough
  for (SparseSet<void *>::const_iterator it = freeList.getRanges().begin(); it != freeList.getRanges().end(); it ++) {
    void *begin = align(it->begin, alignment);

    if ((char *) it->end - (char *) begin >= (ptrdiff_t) size) {
      // enough space -- reserve it
      freeList.exclude(begin, (void *) ((char *) begin + size));

      // register pointer
      sizes[begin] = size;

      return begin;
    }
  }

  THROW(InterfaceException,"SparseSetAllocator could not allocate " << size << " bytes");
}


void SparseSetAllocator::deallocate(void *ptr)
{
  if (ptr != 0) {
    ScopedLock sl(mutex);

    // look up pointer
    std::map<void *, size_t>::iterator index = sizes.find(ptr);

    if (index == sizes.end())
      THROW(InterfaceException,"Pointer was not allocated");

    // free allocated space
    freeList.include(ptr, (void *) ((char *) ptr + index->second));

    // unregister pointer
    sizes.erase(index);
  }
}


} // namespace RTCP
} // namespace LOFAR
