// clang++ -std=c++17 -Wall -O0 -g -fsanitize=address -mavx512f main.cpp
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

#include "include/simvec.hpp"


int main() {
    static constexpr std::size_t VECSIZE = 256; // 配列要素数
    simvec::Simvec<float, VECSIZE> a{};
    simvec::Simvec<float, VECSIZE> b{};
    a[0] = b[0] = 1;
    for (std::size_t i = 1; i < VECSIZE; ++i) {
        a[i] = 0;
        b[i] = 0;
    }

    simvec::Simvec<float, VECSIZE> sum{ a + b };
    for (int i = 0; i < VECSIZE; ++i) {
        std::cout << sum[i] << ", ";
    }
    std::cout << std::endl;
    return 0;
}
