#include "managed_allocator.hpp"

#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <iostream>

// Create a nickname for vectors which use a managed_allocator
template<class T>
using managed_vector = std::vector<T, managed_allocator<T>>;

__global__ void increment_kernel(int *data, size_t n)
{
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if(i < n)
  {
    data[i] += 1;
  }
}

int main()
{
  size_t n = 1 << 20;

  managed_vector<int> vec(n);

  // we can use the vector from the host
  std::iota(vec.begin(), vec.end(), 0);

  std::vector<int> ref(n);
  std::iota(ref.begin(), ref.end(), 0);
  assert(std::equal(ref.begin(), ref.end(), vec.begin()));

  // we can also use it in a CUDA kernel
  size_t block_size = 256;
  size_t num_blocks = (n + (block_size - 1)) / block_size;

  increment_kernel<<<num_blocks, block_size>>>(vec.data(), vec.size());

  cudaDeviceSynchronize();

  std::for_each(ref.begin(), ref.end(), [](int& x)
  {
    x += 1;
  });

  assert(std::equal(ref.begin(), ref.end(), vec.begin()));

  std::cout << "OK" << std::endl;

  return 0;
}

