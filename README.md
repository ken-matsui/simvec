## simvec

### Usage

Here's a small usage example:

```cpp
#include "include/simvec.hpp"

int main() {
    static constexpr std::size_t VECSIZE = 256;
    simvec::Simvec<float, VECSIZE> a{};
    simvec::Simvec<float, VECSIZE> b{};
    simvec::Simvec<float, VECSIZE> c{};
    a[0] = b[0] = c[0] = 1;
    for (std::size_t i = 1; i < VECSIZE; ++i) {
        a[i] = b[i] = c[i] = 0;
    }

    simvec::Simvec<float, VECSIZE> sum{};
    sum = a + b + c;
    for (const auto& s : sum) {
        std::cout << s << ", ";
    }
    std::cout << std::endl;

    return 0;
}
```

```bash
$ clang++ -std=c++17 -Wall -O0 -g -fsanitize=address -mavx512f main.cpp
$ sde -skx -- ./a.out
3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
```

### Requirements

Simvec requires a fully C++17 compliant compiler. (GCC or Clang, MSVC can not be used.)

### Note

When executed in a CPU environment not compatible with AVX-512 instruction, the following is output.
So please use [emulator](https://software.intel.com/en-us/articles/intel-software-development-emulator) provided by Intel.

```bash
[1]    48770 illegal hardware instruction  ./a.out
```
