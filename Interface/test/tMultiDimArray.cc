#include <lofar_config.h>
#include <Interface/MultiDimArray.h>
#include <Common/LofarLogger.h>

using namespace LOFAR;
using namespace RTCP;
using namespace boost;

// A class that keeps track of the number of live objects.
struct Object {
  Object()  { val = nrObjects++; }
  ~Object() { nrObjects--; }

  bool operator==(const Object &other) const { return val == other.val; }

  static size_t nrObjects;

  size_t val;
};

size_t Object::nrObjects = 0;

template<unsigned DIM> struct Tester {
  typedef MultiDimArray<Object,DIM> ArrayType;
  typedef typename ArrayType::ExtentList ExtentList;

  void test(const ExtentList &extents);

  void half(ExtentList &extents);

  void assert_live_elements(size_t n);
};

template<unsigned DIM> void Tester<DIM>::test(const Tester<DIM>::ExtentList &extents)
{
  size_t nrElements = ArrayType::nrElements(extents);

  LOG_INFO_STR("Test on " << DIM << " dimensions, " << nrElements << " elements");

  // We assume to start with an empty list
  assert_live_elements(0);

  // Test the constructors
  LOG_INFO_STR("Test default constructor");
  {
    ArrayType array;
  }
  assert_live_elements(0);

  LOG_INFO_STR("Test (extents) constructor");
  {
    ArrayType array(extents);

    ASSERT(array.num_elements() == nrElements);
    assert_live_elements(nrElements);
  }
  assert_live_elements(0);

  LOG_INFO_STR("Test (extents, ptr, construct=false) constructor");
  {
    Object *data = new Object[ArrayType::nrElements(extents)];

    ArrayType array(extents, data, false);

    ASSERT(array.num_elements() == nrElements);
    assert_live_elements(nrElements);

    delete[] data;
  }
  assert_live_elements(0);

  LOG_INFO_STR("Test (extents, ptr, construct=true) constructor");
  {
    char *data = new char[nrElements * sizeof(Object)];

    ArrayType array(extents, reinterpret_cast<Object*>(data), true);

    ASSERT(array.num_elements() == nrElements);
    assert_live_elements(nrElements);

    delete[] data;
  }
  assert_live_elements(0);

  LOG_INFO_STR("Test copy constructor");
  {
    ArrayType array1(extents);
    ArrayType array2(array1);

    assert_live_elements(nrElements * 2);
  }
  assert_live_elements(0);

  // Test assignment (TODO: deep check)
  LOG_INFO_STR("Test assignment operator");
  {
    ArrayType array1(extents);
    ArrayType array2(extents);

    array2 = array1;

    assert_live_elements(nrElements * 2);

    // compare all elements
    for (size_t i = 0; i < nrElements; i++)
      ASSERTSTR( *(array1.origin() + i) == *(array2.origin() + i), "Element " << i << " differs." );
  }
  assert_live_elements(0);

  // Test resizing
  ExtentList alternate_extents(extents);
  half(alternate_extents);

  size_t alternate_nrElements(ArrayType::nrElements(alternate_extents));

  LOG_INFO_STR("Test resize(extents)");
  {
    ArrayType array(extents);

    array.resize(alternate_extents);

    ASSERT(array.num_elements() == alternate_nrElements);
    assert_live_elements(alternate_nrElements);
  }
  assert_live_elements(0);

  // Test resizing in place
  LOG_INFO_STR("Test resizeInplace(extents)");
  {
    ArrayType array(extents);

    array.resizeInplace(alternate_extents);

    ASSERT(array.num_elements() == alternate_nrElements);
    assert_live_elements(alternate_nrElements);

    // try resizing back to original size
    array.resizeInplace(extents);

    ASSERT(array.num_elements() == nrElements);
    assert_live_elements(nrElements);
  }
  assert_live_elements(0);

  LOG_INFO_STR("Test resizeInplace(extents) on array with external data (we provide both the memory and the object construction)");
  {
    Object *data = new Object[nrElements];

    ArrayType array(extents, reinterpret_cast<Object*>(data), false);

    array.resizeInplace(alternate_extents);

    // We do the construction, so MultiDimArray cannot change anything.
    assert_live_elements(nrElements);

    delete[] data;
  }
  assert_live_elements(0);

  LOG_INFO_STR("Test resizeInplace(extents) on array with external memory (we provide the memory, the array constructs the objects)");
  {
    char *data = new char[nrElements * sizeof(Object)];

    ArrayType array(extents, reinterpret_cast<Object*>(data), true);

    array.resizeInplace(alternate_extents);

    // MultiDimArray does the construction, so MultiDimArray can change the
    // number of objects as well.
    assert_live_elements(alternate_nrElements);

    delete[] data;
  }
  assert_live_elements(0);
}

template<unsigned DIM> void Tester<DIM>::half(Tester<DIM>::ExtentList &extents)
{
  for (size_t i = 0; i < DIM; i++)
    extents.ranges_[i] = extents.ranges_[i].size() / 2;
}

template<unsigned DIM> void Tester<DIM>::assert_live_elements(size_t n)
{
  ASSERTSTR( Object::nrObjects == n, "Expected " << n << " live objects, but encountered " << Object::nrObjects );
}

int main()
{
  INIT_LOGGER("tMultiDimArray");

  { Tester<1> tester; tester.test(extents[10]); }
  { Tester<2> tester; tester.test(extents[10][10]); }
  { Tester<3> tester; tester.test(extents[10][10][10]); }
  { Tester<4> tester; tester.test(extents[10][10][10][10]); }
}
