#ifndef LOFAR_INTERFACE_MULTI_DIM_ARRAY_H
#define LOFAR_INTERFACE_MULTI_DIM_ARRAY_H

#include <Interface/Align.h>
#include <Interface/Allocator.h>
#include <Interface/Exceptions.h>
#include <Common/LofarLogger.h>
#include <boost/multi_array.hpp>

#include <memory>
#include <ostream>
#include <stdexcept>

#if _POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600
#include <cstdlib>
#else
#include <malloc.h>
#endif


namespace LOFAR {
namespace RTCP {


/*
 * MultiDimArray wraps a boost::multi_array_ref to provide enhanced allocation,
 * alignment, resize, and reshape functionality.
 */
template <typename T, unsigned DIM> class MultiDimArray : public boost::multi_array_ref<T, DIM>
{
  public:
    typedef boost::multi_array_ref<T, DIM> SuperType;
    typedef boost::detail::multi_array::extent_gen<DIM> ExtentList;

    /*
     * Default constructor. Creates an array of size 0 in all dimensions.
     */
    MultiDimArray(Allocator &allocator = heapAllocator)
    :
      SuperType(0, boost::detail::multi_array::extent_gen<DIM>()),
      allocator(&allocator),
      allocated_num_elements(0),
      alignment(0),
      padToAlignment(false),
      construct(true)
    {
    }

    /*
     * In-place constructor. Casts a MultiDimArray onto pre-allocated memory.
     *
     * extents:   dimensions of array
     * ptr:       pointer to memory
     * construct: true:  construct (and later destruct) the elements
     *            false: elements are already constructed (view)
     */
    MultiDimArray(const ExtentList &extents, T *ptr, bool construct = true)
    :
      // Use 'placement new' to force initialisation through constructors if T is a class

      // TODO: Not sure how to handle an exception raised by the constructor of T. The placement
      // delete[] will be called, but that's an empty stub.
      SuperType(construct ? new(ptr)T[nrElements(extents)] : ptr, extents),
      allocator(0),
      allocated_num_elements(nrElements(extents)),
      alignment(alignment),
      padToAlignment(padToAlignment),
      construct(construct)
    {
      // NOTE: Elements are not destructed even if construct == true!
    }


    /*
     * Create an array, including memory allocation.
     *
     * extents:        dimensions of array
     * alignment:      alignment of first element
     * allocator:      allocator to use for allocation
     * padToAlignment: if true, the size of the allocated memory is also padded
     *                 to `alignment'.
     * construct:      true:  construct (and later destruct) the elements
     *                 false: elements are already constructed (view)
     */
    MultiDimArray(const ExtentList &extents, size_t alignment = defaultAlignment(), Allocator &allocator = heapAllocator, bool padToAlignment = false, bool construct = true)
    :
      // Use 'placement new' to force initialisation through constructors if T is a class

      // TODO: Not sure how to handle an exception raised by the constructor of T. The placement
      // delete[] will be called, but that's an empty stub.
      SuperType(allocate(nrElements(extents), alignment, allocator, padToAlignment, construct), extents),
      allocator(&allocator),
      allocated_num_elements(nrElements(extents)),
      alignment(alignment),
      padToAlignment(padToAlignment),
      construct(construct)
    {
    }

    /*
     * Copy constructor. Uses other.allocator to allocate a copy.
     */
    MultiDimArray(const MultiDimArray<T,DIM> &other)
    :
      SuperType(other.num_elements_ && other.allocator ? allocate(other.num_elements_, other.alignment, *other.allocator, other.padToAlignment, other.construct) : 0, other.extent_list_),
//new(other.allocator->allocate(padToAlignment ? align(other.num_elements_ * sizeof(T), other.alignment) : other.num_elements_ * sizeof(T), other.alignment))T[other.num_elements_] : 0, other.extent_list_),
      allocator(other.allocator),
      allocated_num_elements(other.num_elements_),
      alignment(other.alignment),
      padToAlignment(other.padToAlignment),
      construct(true)
    {
      ASSERTSTR(other.allocator, "Cannot copy MultiDimArray that does not have an allocator.");

      *this = other;
    }

    ~MultiDimArray()
    {
      destructElements();

      if (allocator) {
        allocator->deallocate(this->base_);
      }  
    }

    /*
     * Assignment operator. Works on any two arrays of the same dimensionality,
     * type, and total number of elements.
     */
    MultiDimArray<T,DIM> &operator= (const MultiDimArray<T,DIM> &other)
    {
      if (other.num_elements_ != this->num_elements_)
        THROW(InterfaceException, "Tried to assign an array with " << other.num_elements_ << " elements to an array with " << this->num_elements_ << "elements.");

      T *me  = this->origin();
      const T *him = other.origin();

      for (size_t i = 0; i < this->num_elements_; i ++)
        *(me++) = *(him++); 

      return *this;
    }

    /*
     * Resize the array by allocating new memory.
     *
     * extents:        new dimensions of array
     * alignment:      alignment of first element
     * allocator:      allocator to use for allocation
     * padToAlignment: if true, the size of the allocated memory is also padded
     *                 to `alignment'.
     * construct:      true:  construct (and later destruct) the elements
     *                 false: elements are already constructed (view)
     */
    void resize(const ExtentList &extents, size_t alignment, Allocator &allocator, bool padToAlignment = false, bool construct = true)
    {
      MultiDimArray newArray(extents, alignment, allocator, padToAlignment, construct);
      std::swap(this->base_, newArray.base_);
      std::swap(this->storage_, newArray.storage_);
      std::swap(this->extent_list_, newArray.extent_list_);
      std::swap(this->stride_list_, newArray.stride_list_);
      std::swap(this->index_base_list_, newArray.index_base_list_);
      std::swap(this->origin_offset_, newArray.origin_offset_);
      std::swap(this->directional_offset_, newArray.directional_offset_);
      std::swap(this->num_elements_, newArray.num_elements_);
      std::swap(this->allocator, newArray.allocator);
      std::swap(this->allocated_num_elements, newArray.allocated_num_elements);
      std::swap(this->alignment, newArray.alignment);
      std::swap(this->padToAlignment, newArray.padToAlignment);
      std::swap(this->construct, newArray.construct);
    }

    /*
     * Resize the array by allocating new memory. Requires the allocator to be
     * set.
     *
     * extents:        new dimensions of array
     * alignment:      alignment of first element
     */
    void resize(const ExtentList &extents, size_t alignment = defaultAlignment())
    {
      ASSERTSTR(allocator, "Cannot resize MultiDimArray that does not have an allocator.");

      resize(extents, alignment, *allocator);
    }

    /*
     * Resize the array in-place (reshape). Cannot resize the array beyond the
     * memory that was originally allocated.
     *
     * extents:        new dimensions of array
     */
    void resizeInplace(const ExtentList &extents)
    {
      unsigned new_num_elements = nrElements(extents);

      if (new_num_elements > allocated_num_elements)
        THROW(InterfaceException, "MultiDimArray::resizeInplace: requested to resize to " << new_num_elements << " elements, but only " << allocated_num_elements << " are allocated");

      // only destruct and construct all elements if the number of elements actually changes
      if (new_num_elements != this->num_elements_ && construct) {
        destructElements();
        (void)new(this->base_)T[new_num_elements];
      }

      // regenerate the metadata, and use it.
      // Our metadata will be freed due to the swap, but our data won't, because
      // newArray.allocator == 0. Nor will our data be destructed, because
      // newArray.construct == false.
      MultiDimArray newArray(*this, extents);
      //std::swap(this->base_, newArray.base_);
      std::swap(this->storage_, newArray.storage_);
      std::swap(this->extent_list_, newArray.extent_list_);
      std::swap(this->stride_list_, newArray.stride_list_);
      std::swap(this->index_base_list_, newArray.index_base_list_);
      std::swap(this->origin_offset_, newArray.origin_offset_);
      std::swap(this->directional_offset_, newArray.directional_offset_);
      std::swap(this->num_elements_, newArray.num_elements_);
      //std::swap(this->allocator, newArray.allocator);
      //std::swap(this->allocated_num_elements, newArray.allocated_num_elements);
      //std::swap(this->alignment, newArray.alignment);
    }

    /*
     * Resize the array in-place (reshape) by changing only a single dimension.
     * Cannot resize the array beyond the memory that was originally allocated.
     *
     * dimNr:      dimension number to change.
     * newSize:    new size of dimension.
     */
    void resizeOneDimensionInplace(unsigned dimNr, size_t newSize)
    {
      ASSERTSTR(dimNr < DIM, "Cannot resize dimension " << dimNr << " because there are only " << DIM << " dimensions.");

      ExtentList newDims;

      for (size_t i = 0; i < DIM; i++)
        newDims.ranges_[i] = this->extent_list_[i];

      newDims.ranges_[dimNr] = newSize;  

      resizeInplace(newDims);
    }


    static size_t defaultAlignment()
    {
      return sizeof(T) < 16 ? 8 : sizeof(T) < 32 ? 16 : 32;
    }


    static size_t nrElements(const ExtentList &extents)
    {
      size_t size = 1;

      for (unsigned i = 0; i < extents.ranges_.size(); i ++)
	size *= extents.ranges_[i].size();

      return size;
    }

  private:
    // All members need to be mutable to be able to swap them in resize()

    // Allocator with which the array was allocated, or NULL if the memory
    // was externally allocated.
    Allocator *allocator;

    // Number of elements that were originally allocated.
    size_t    allocated_num_elements;

    // Alignment for the first element.
    unsigned  alignment;

    // If padToAlignment is true, the memory allocated is also padded towards
    // the specified alignment.
    bool      padToAlignment;

    // If construct is true, the elements have been constructed by us.
    // If construct is false, the elements have been constructed externally.
    bool      construct;

    T *allocate(size_t nrElements, size_t alignment, Allocator &allocator, bool padToAlignment, bool construct) const {
      size_t dataSize = padToAlignment
                        ? align(nrElements * sizeof(T), alignment)
                        : nrElements * sizeof(T);

      T *ptr = static_cast<T*>(allocator.allocate(dataSize, alignment));

      return construct ? new(ptr)T[nrElements] : ptr;
    }

    // a MultiDimArray made to replace another, using a different shape. Assumes
    // the original MultiDimArray allocated enough memory to hold the new
    // dimensions.
    MultiDimArray(const MultiDimArray<T,DIM> &other, const ExtentList &extents)
    :
      // Use 'placement new' to force initialisation through constructors if T is a class

      // TODO: Not sure how to handle an exception raised by the constructor of T. The placement
      // delete[] will be called, but that's an empty stub.
      SuperType(other.base_, extents),
      allocator(0), // we did not allocate this
      allocated_num_elements(0),
      alignment(other.alignment),
      padToAlignment(other.padToAlignment),
      construct(false) // construction was done by an external source
    {
    }


    void destructElements()
    {
      if (!construct)
        return;

      // explicitly call the destructors in the 'placement new' array since C++
      // cannot do this for us. The delete[] operator cannot know the size of the
      // array, and the placement delete[] operator exists (since new()[] will look
      // for it) but does nothing.
      T *elem = this->origin();

      for (size_t i = 0; i < this->num_elements_; i ++)
        (elem++)->~T();
    }
};


template <typename T> class Vector : public MultiDimArray<T, 1>
{
  public:
    typedef MultiDimArray<T, 1>		   SuperType;
    typedef typename SuperType::ExtentList ExtentList;

    Vector(Allocator &allocator = heapAllocator)
    :
      SuperType(allocator)
    {
    }

    Vector(size_t x, size_t alignment = SuperType::defaultAlignment(), Allocator &allocator = heapAllocator)
    :
      SuperType(boost::extents[x], alignment, allocator)
    {
    }

    Vector(const ExtentList &extents, size_t alignment = SuperType::defaultAlignment(), Allocator &allocator = heapAllocator)
    :
      SuperType(extents, alignment, allocator)
    {
    }

    using SuperType::resize;

    void resize(size_t x, size_t alignment = SuperType::defaultAlignment(), Allocator &allocator = heapAllocator, bool padToAlignment = false)
    {
      SuperType::resize(boost::extents[x], alignment, allocator, padToAlignment);
    }
};


template <typename T> class Matrix : public MultiDimArray<T, 2>
{
  public:
    typedef MultiDimArray<T, 2>		   SuperType;
    typedef typename SuperType::ExtentList ExtentList;

    Matrix(Allocator &allocator = heapAllocator)
    :
      SuperType(allocator)
    {
    }

    Matrix(size_t x, size_t y, size_t alignment = SuperType::defaultAlignment(), Allocator &allocator = heapAllocator, bool padToAlignment = false)
    :
      SuperType(boost::extents[x][y], alignment, allocator, padToAlignment)
    {
    }

    Matrix(const ExtentList &extents, size_t alignment = SuperType::defaultAlignment(), Allocator &allocator = heapAllocator, bool padToAlignment = false)
    :
      SuperType(extents, alignment, allocator, padToAlignment)
    {
    }

    using SuperType::resize;

    void resize(size_t x, size_t y, size_t alignment = SuperType::defaultAlignment(), Allocator &allocator = heapAllocator, bool padToAlignment = false)
    {
      SuperType::resize(boost::extents[x][y], alignment, allocator, padToAlignment);
    }
};


template <typename T> class Cube : public MultiDimArray<T, 3>
{
  public:
    typedef MultiDimArray<T, 3>		   SuperType;
    typedef typename SuperType::ExtentList ExtentList;

    Cube(Allocator &allocator = heapAllocator)
    :
      SuperType(allocator)
    {
    }

    Cube(size_t x, size_t y, size_t z, size_t alignment = SuperType::defaultAlignment(), Allocator &allocator = heapAllocator)
    :
      SuperType(boost::extents[x][y][z], alignment, allocator)
    {
    }

    Cube(const ExtentList &extents, size_t alignment = SuperType::defaultAlignment(), Allocator &allocator = heapAllocator)
    :
      SuperType(extents, alignment, allocator)
    {
    }

    using SuperType::resize;

    void resize(size_t x, size_t y, size_t z, size_t alignment = SuperType::defaultAlignment(), Allocator &allocator = heapAllocator, bool padToAlignment = false)
    {
      SuperType::resize(boost::extents[x][y][z], alignment, allocator, padToAlignment);
    }
};

// output function for full MultiDimArrays
template <typename T, unsigned DIM> inline std::ostream &operator<< (std::ostream& str, const MultiDimArray<T,DIM> &array)
{
  str << "[ ";

  for (size_t i = 0; i < array.size(); i ++) {
    if (i > 0)
      str << ", ";

    str << array[i];
  }

  str << " ]";
  return str;
}

// output function for subdimensions of MultiDimArrays
template <typename T, unsigned DIM, typename TPtr> inline std::ostream &operator<< (std::ostream& str, const typename boost::detail::multi_array::const_sub_array<T,DIM,TPtr> &array)
{
  str << "[ ";

  for (size_t i = 0; i < array.size(); i ++) {
    if (i > 0)
      str << ", ";

    str << array[i];
  }

  str << " ]";
  return str;
}

} // namespace RTCP
} // namespace LOFAR

#endif
