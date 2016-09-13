#ifndef LOFAR_INTERFACE_ALLOCATOR_H
#define LOFAR_INTERFACE_ALLOCATOR_H

#include <Interface/SparseSet.h>
#include <Common/Thread/Mutex.h>

#include <map>

namespace LOFAR {
namespace RTCP {

// There is a strict separation between a memory allocator and the physical
// memory (arena) that it manages.

/*
 * Arena owns a chunk of memory for SparseSetAllocator to play in.
 */
class Arena
{
  public:
    void	*begin() const { return itsBegin; }
    size_t	size() const { return itsSize; }
  
  protected:
    void	*itsBegin;
    size_t	itsSize;
};


/*
 * MallocedArena allocates memory using malloc.
 */
class MallocedArena: public Arena
{
  public:
		MallocedArena(size_t size, size_t alignment);
    ~MallocedArena();
};


/*
 * FixedArena represents an externally allocated
 * piece of memory.
 */
class FixedArena : public Arena
{
  public:
    FixedArena(void *begin, size_t size);
};


/*
 * An Allocator can both allocate and deallocate pointers.
 */
class Allocator
{
  public:
    virtual			~Allocator();

    virtual void		*allocate(size_t size, size_t alignment = 1) = 0;
    virtual void		deallocate(void *) = 0;

    /*
     * Allows TYPE *foo = allocator.allocateTyped() without type-casting.
     */
    class TypedAllocator {
    public:
      TypedAllocator(Allocator &allocator, size_t alignment): allocator(allocator), alignment(alignment) {}

      // cast-operator overloading is the only way to let C++ automatically deduce the type that we want
      // to return.
      template<typename T> operator T* () {
        return static_cast<T*>(allocator.allocate(sizeof(T), alignment));
      }
    private:
      Allocator &allocator;
      const size_t alignment;
    };

    TypedAllocator allocateTyped(size_t alignment = 1) { return TypedAllocator(*this, alignment); }
};


/*
 * Allocates memory on the heap.
 */
class HeapAllocator : public Allocator
{
  public:
    virtual			~HeapAllocator();

    virtual void		*allocate(size_t size, size_t alignment = 1);
    virtual void		deallocate(void *);
};

extern HeapAllocator heapAllocator;


/*
 * Allocates memory within an Arena, using a simple
 * memory manager based on a SparseSet.
 *
 * The allocator is deterministic.
 */
class SparseSetAllocator : public Allocator
{
  public:
				SparseSetAllocator(const Arena &);

    virtual void		*allocate(size_t size, size_t alignment = 1);
    virtual void		deallocate(void *);

    bool                        empty() { ScopedLock sl(mutex); return sizes.empty(); }

  private:
    Mutex                       mutex;

    SparseSet<void *>		freeList;
    std::map<void *, size_t>	sizes;
};

} // namespace RTCP
} // namespace LOFAR

#endif
