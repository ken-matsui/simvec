// clang++ -std=c++17 -Wall -O0 -g -fsanitize=address -mfma -mavx512f main.cpp
// sde -skx -- ./a.out
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstddef>

#include <iostream>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <numeric>
#include <new>

#include <x86intrin.h>

// #include "include/simvec.hpp"


// 内積計算
static inline float
innerProduct(const float* a, const float* b, std::size_t n)
{
  static constexpr std::size_t INTERVAL = sizeof(__m512) / sizeof(float);
  __m512 sumx16 = {0};
  for (std::size_t i = 0; i < n; i += INTERVAL) {
    __m512 ax16 = _mm512_load_ps(&a[i]);
    __m512 bx16 = _mm512_load_ps(&b[i]);
#ifdef __FMA__
    sumx16 = _mm512_fmadd_ps(ax16, bx16, sumx16);
#else
    sumx16 = _mm512_add_ps(sumx16, _mm512_mul_ps(ax16, bx16));
#endif
  }

  alignas(__m512) float s[INTERVAL] = {0};
  _mm512_store_ps(s, sumx16);

  std::size_t offset = n - n % INTERVAL;
  return std::inner_product(
      a + offset,
      a + n,
      b + offset,
      std::accumulate(std::begin(s), std::end(s), 0.0f));
}


template<typename T>
static inline T* // memory size (byte), alignment (2^n)
alignedAlloc(std::size_t nBytes, std::size_t alignment=alignof(T))
{
    return reinterpret_cast<T*>(::operator new(nBytes, static_cast<std::align_val_t>(alignment)));
    // posix_memalign
    // void* p;
    // return reinterpret_cast<T*>(::posix_memalign(&p, alignment, nBytes) == 0 ? p : nullptr);
}

int main() {
    static constexpr int ALIGN = alignof(__m512);
    static constexpr int N_ELEMENT = 256; // 配列要素数

    std::unique_ptr<float[]> a(alignedAlloc<float>(N_ELEMENT * sizeof(float), ALIGN));
    std::unique_ptr<float[]> b(alignedAlloc<float>(N_ELEMENT * sizeof(float), ALIGN));
    for (int i = 0; i < N_ELEMENT; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i);
    }
    std::cout << innerProduct(a.get(), b.get(), N_ELEMENT) << std::endl;

    return 0;
}
