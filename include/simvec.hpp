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


namespace simvec {
    // https://qiita.com/kazatsuyu/items/f8c3b304e7f8b35263d8
    template<typename ...Args>
    struct first_enabled {};

    template<typename T, typename ...Args>
    struct first_enabled<std::enable_if<true, T>, Args...> { using type = T; };
    template<typename T, typename ...Args>
    struct first_enabled<std::enable_if<false, T>, Args...>: first_enabled<Args...> {};
    template<typename T, typename ...Args>
    struct first_enabled<T, Args...> { using type = T; };

    template<typename ...Args>
    using first_enabled_t = typename first_enabled<Args...>::type;


    // template <class L, class R>
    // class Plus {
    //     const L& l_; // 左辺値
    //     const R& r_; // 右辺値
    // public:
    //     Plus(const L& l, const R& r)
    //         : l_(l), r_(r) {}

    //     float operator[](std::size_t i) const
    //     {
    //         return l_[i] + r_[i];
    //     }
    // };

    // template <class L, class R>
    // inline Plus<L, R> operator+(const L& l, const R& r)
    // {
    //     return Plus<L, R>(l, r);
    // }

    // Simvec zeros();
    // Simvec ones();

    template<typename T>
    static inline T* // memory size (byte), alignment (2^n)
    alignedAlloc(std::size_t nBytes, std::size_t alignment=alignof(T))
    {
        return reinterpret_cast<T*>(::operator new(nBytes, static_cast<std::align_val_t>(alignment)));
    }


    template <typename T, std::size_t Size>
    struct Simvec {
        using value_type = first_enabled_t<
            std::enable_if<std::is_same_v<T, float>, __m512>,
            std::enable_if<std::is_same_v<T, double>, __m512d>,
            __m512i>;

        std::size_t INTERVAL = sizeof(value_type) / sizeof(T);
        std::unique_ptr<T[]> p;
        Simvec() {
            p.reset(alignedAlloc<T>(Size * sizeof(T), alignof(value_type)));
        }
        Simvec(std::unique_ptr<T[]>&& p_) {
            p = std::move(p_);
        }

        template <class E>
        Simvec& operator=(const E& r)
        {
            for (std::size_t i = 0; i < Size; i += INTERVAL) {
                _mm512_store_ps(&p[i], _mm512_load_ps(&r[i])); // ここも命令を抽象化
            }
            return *this;
        }


        Simvec operator+(const Simvec& r) {
            std::unique_ptr<T[]> sum(alignedAlloc<T>(Size * sizeof(T), alignof(value_type)));
            for (std::size_t i = 0; i < Size; i += INTERVAL) {
                value_type px16 = _mm512_load_ps(&p[i]);
                value_type rx16 = _mm512_load_ps(&r[i]);
                value_type sumx16 = _mm512_add_ps(px16, rx16);
                _mm512_store_ps(&sum[i], sumx16);
            }
            return Simvec{ std::move(sum) };
        }

        T& operator[](std::size_t i) const {
            return p[i];
        }

        constexpr std::size_t size() const noexcept {
            return Size;
        }
    };
}
