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

    // Simvec zeros();
    // Simvec ones();

    template<typename T>
    inline T* // memory size (byte), alignment (2^n)
    aligned_alloc(std::size_t size, std::size_t alignment=alignof(T)) {
        return reinterpret_cast<T*>(::operator new(size, static_cast<std::align_val_t>(alignment)));
    }

    template <typename M, typename T>
    M load_ps(T __a) {
        if constexpr (std::is_same_v<M, __m512>) {
            return _mm512_load_ps(__a);
        }
        else if constexpr (std::is_same_v<M, __m512d>) {
            return _mm512d_load_ps(__a);
        }
        else {
            return _mm512i_load_ps(__a);
        }
    }
    template <typename M, typename T, typename U>
    void store_ps(T __a, U __b) {
        if constexpr (std::is_same_v<M, __m512>) {
            _mm512_store_ps(__a, __b);
        }
        else if constexpr (std::is_same_v<M, __m512d>) {
            _mm512d_store_ps(__a, __b);
        }
        else {
            _mm512i_store_ps(__a, __b);
        }
    }
    template <typename M, typename T, typename U>
    M add_ps(T __a, U __b) {
        if constexpr (std::is_same_v<M, __m512>) {
            return _mm512_add_ps(__a, __b);
        }
        else if constexpr (std::is_same_v<M, __m512d>) {
            return _mm512d_add_ps(__a, __b);
        }
        else {
            return _mm512i_add_ps(__a, __b);
        }
    }


    template <typename T, std::size_t Size>
    class Simvec {
        std::size_t INTERVAL = sizeof(value_type) / sizeof(T);
        std::unique_ptr<T[]> p;

    public:
        using value_type = first_enabled_t<
            std::enable_if<std::is_same_v<T, float>, __m512>,
            std::enable_if<std::is_same_v<T, double>, __m512d>,
            __m512i>;

        Simvec(const Simvec&) = delete;
        Simvec& operator=(const Simvec&) = delete;
        Simvec(Simvec&&) = default;
        Simvec& operator=(Simvec&&) = default;
        Simvec() {
            p.reset(aligned_alloc<T>(Size * sizeof(T), alignof(value_type)));
        }
        explicit Simvec(std::unique_ptr<T[]>&& p_) {
            p = std::move(p_);
        }

        template <class E>
        Simvec& operator=(const E& r) {
            for (std::size_t i = 0; i < Size; i += INTERVAL) {
                store_ps<value_type>(&p[i], r[i]);
            }
            return *this;
        }
        T& operator[](std::size_t i) const {
            return p[i];
        }

        constexpr std::size_t size() const noexcept {
            return Size;
        }

        auto begin() const {
            return p.get();
        }
        auto end() const {
            return p.get() + Size;
        }
    };


    // Expression Temlate (https://faithandbrave.hateblo.jp/entry/20081003/1223026720)
    template <class L, class R>
    class Plus {
        const L& l_;
        const R& r_;
    public:
        using value_type = typename L::value_type;

        Plus(const L& l, const R& r)
            : l_(l), r_(r) {}

        value_type operator[](std::size_t i) const {
            value_type lx16;
            if constexpr (std::is_same_v<value_type, decltype(l_[i])>) {
                lx16 = l_[i];
            }
            else {
                lx16 = load_ps<value_type>(&l_[i]);
            }
            value_type rx16 = load_ps<value_type>(&r_[i]);
            return add_ps<value_type>(lx16, rx16);
        }
    };

    template <class L, class R>
    inline Plus<L, R> operator+(const L& l, const R& r) {
        return Plus<L, R>(l, r);
    }
}
