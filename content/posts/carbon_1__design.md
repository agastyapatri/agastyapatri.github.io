+++
date        = '2026-10-25T16:01:02+05:30'
draft       = false
title       = 'On the design of a tensor library in C++'
tags        = ["linear algebra", "c++"] 
description  = "carbon, part 1: templates, explicit instantiation and type-dependency" 
+++
> **Toolchain used for every capture:** clang 22.1.8, GCC 16.2.1, libstdc++ 16,Fedora 44, x86-64. Library built with `-std=c++20 -DCARBON_USE_BLAS` against OpenBLAS, the same way `CMakeLists.txt` builds it.

---

## Introduction

C++ templates are blueprints used by the compiler to generate classes when they are instantiated with a specific type. Class templates usually take the form of a single header file because the compiler needs to see the full implementation to generate concrete code at the point of use: the template's definition is visible and a concrete type is known at compile time. 

`carbon` is a C++20 tensor library I'm writing which splits the core tensor class template into the usual header+translation-unit pair anyway. **Explicit Instantiation** is used to bring the two constraints (visible definition + concrete type) into a single place so that user translation units link without errors.
The advantages of separating the interface from the implementation, as I see it, are preserved. An unexpected side effect that I ran into is that I now have the ability to **enforce a type set onto the library.**

This post is a discussion on the design of the library and some decisions I had to make while building it.

---

## 1. Library Layout

| file | lines | contents |
|---|---:|---|
| `src/carbon.hpp` | 225 | the `tensor<T>` class with **declarations only** (except trivial accessors), plus free-function template declarations |
| `src/carbon.cpp` | 918 | **every definition**, then 44 explicit-instantiation lines at the very bottom (L863–917) |
| `src/blas.hpp` | 198 | `carbon::blas<T>`: generic kernels plus CBLAS specializations. Included **only** by `carbon.cpp` |

Explicit instantiation for a constrained set of datatypes:

```cpp
//	explicit instantiation, apparently
template class tensor<f32>;
template class tensor<f64>;
template class tensor<i32>;
template class tensor<i64>;
template std::ostream &operator<<(std::ostream &os, const tensor<f32> &t);
// ... ×4
template tensor<f32> gemm(const tensor<f32> &inp1, const tensor<f32> &inp2);
// ... gemm ×4, gemv ×4
template tensor<f32> dot(const tensor<f32> &inp1, const tensor<f32> &inp2);
template tensor<f32> sin(const tensor<f32>& inp);
// ... f32 and f64 get dot + 9 unary ops each; i32 and i64 get dot, cube, square, relu
```

**The list of supported types has to be at the bottom.** This isn't a style choice. An explicit instantiation of a class only covers members that are *already defined* at that line. 

---

## 2. Two kinds of type restriction
Even though `tensor<T>` has been explicitly instantiated for `f32, f64, i32, i64` only, some members in the class only compile, or only make sense, for floating-point tensors. 
For instance, `std::normal_distribution<T>` and `std::uniform_real_distribution<T>` may only be instantiated with floating-point types, enforced with a `static_assert` by libstdc++. 
Some functions (sigmoid, log, exp, tanh) would compile but yield nonsensical data if used with `i32` and `i64`. 

To prevent both of these behaviours, I rely on the `requires` clause, part of C++20's concepts feature. It attaches a compile-time predicate to a declaration: if the predicate is false for a given `T`, that member is removed from consideration for that type, and its body is never instantiated. 

This also lets `template class tensor<i32>` skip the floating-point-only members instead of failing the build. The alternative would be to explicitly instantiate only the valid members for each type, but then a call like `tensor<i32>::log_()` would still compile against the header and only fail with a link-time error, rather than a clear compile-time one at the call site.

### 2.1 Constrained members fail at compile time
`tensor<T>` has 14 floating-point-only members and one integral-only member. Each restriction is a `requires` clause on a non-template member:

```cpp
// src/carbon.hpp:106-113, 127-130 (excerpt)
static tensor rand_normal(std::vector<i32> shape, T mean, T std)
requires std::floating_point<T>;
static tensor randn(std::vector<i32> shape)
requires std::floating_point<T>;
static tensor randu(std::vector<i32> shape, T low = T(0), T high = T(1))
requires std::floating_point<T>;
static tensor randint(std::vector<i32> shape, T low = T(0), T high = T(1))
requires std::integral<T>;
// ...
void log_()
requires std::floating_point<T>;
void exp_()
requires std::floating_point<T>;
```

- **Floating-point only (14):** `rand_normal`, `randn`, `randu`, `randn_he`, `randu_he`, `randn_xavier`, `randu_xavier`, `log_`, `exp_`, `sigmoid_`, `sin_`, `cos_`, `tanh_`, `pow_`.
- **Integral only (1):** `randint`.
- `template class tensor<i32>;` compiles, because the explicit instantiation skips every member whose constraint `int` doesn't satisfy. The rule works **in both directions**: `tensor<f32>` never instantiates `randint`.
- Calling a skipped member is a **compile-time error at the call site**, because the user's compiler can see the `requires` clause in the header:

```cpp
carbon::ti32 t({3}, {1, 2, 3});
t.exp_();   // the compiler raises an error here.
```

(The error text is shown next to the link error in §2.3.)

There is a bit of nuance here: some clauses are needed to compile and others are a choice:

| constraint removed | result | why |
|---|---|---|
| `randint`'s `requires std::integral<T>` | **`template class tensor<f32>;` no longer compiles** | `std::uniform_int_distribution<float>` hits a libstdc++ `static_assert`: *"template argument must be an integral type"*, reported at `carbon.cpp:864`, the `template class` line |
| `rand_normal`'s `requires std::floating_point<T>` | **`template class tensor<i32>;` no longer compiles** | `std::normal_distribution<int>`: *"result_type must be a floating point type"* |
| `exp_`'s `requires std::floating_point<T>` | **compiles fine** for `i32` | `std::exp(int)` returns `double` and the lambda `static_cast`s it back. `exp_()` on `{0,1,2,3}` gives `{1, 2, 7, 20}`, truncated without any warning |

So `randint` and the RNG factories are constrained because otherwise the library doesn't build. `exp_`, `log_` and the other transcendentals are constrained because their integer results would be wrong. Both kinds of constraint look the same in the header, but they exist for different reasons.

### 2.2 Free functions fail at link time
The namespace-level math functions have **no** `requires` clause:
```cpp
// src/carbon.hpp:46-54
template <typename U> tensor<U> sin(const tensor<U>& inp)  ;
template <typename U> tensor<U> cos(const tensor<U>& inp)  ;
template <typename U> tensor<U> log(const tensor<U>& inp)  ;
template <typename U> tensor<U> tanh(const tensor<U>& inp) ;
template <typename U> tensor<U> exp(const tensor<U>& inp)  ;
template <typename U> tensor<U> sigmoid(const tensor<U>& inp);
template <typename U> tensor<U> relu(const tensor<U>& inp);
template <typename U> tensor<U> cube(const tensor<U>& inp);
template <typename U> tensor<U> square(const tensor<U>& inp);
```

- The only thing restricting them is **which lines exist** in the instantiation list. `exp` exists for `f32`/`f64`. `relu`, `square`, `cube`, `dot`, `gemm` and `gemv` exist for all four types.
- `carbon::exp(ti32{...})` **compiles**. The header declaration matches for any `U`. With no definition in sight, the compiler can't instantiate it, so it emits a call to the external symbol `carbon::exp<int>` and trusts that another TU provides it. None does:

```
undefined reference to `carbon::tensor<int> carbon::exp<int>(carbon::tensor<int> const&)'
```

### 2.3 The inconsistency

Same operation, same type, two spellings, two different phases of failure:

|  | `t.exp_()` on a `ti32` | `carbon::exp(t)` on a `ti32` |
|---|---|---|
| restricted by | `requires std::floating_point<T>` in the header | the absence of a line in `carbon.cpp` |
| fails at | **compile time** | **link time** |
| reported by | compiler | linker (`ld.bfd`) |
| points at | your source line *and* the constraint in `carbon.hpp` | an offset in `main`'s `.text` section |
| tells you why | yes: `'int' does not satisfy 'floating_point'` | no: just "undefined reference" |

The two errors side by side (clang 22):

<table>
<tr><th>Compile time: <code>t.exp_()</code></th><th>Link time: <code>carbon::exp(t)</code></th></tr>
<tr>
<td><pre>
err_member.cpp:4:4: error: invalid reference to
      function 'exp_': constraints not satisfied
    4 |         t.exp_();
      |           ^
src/carbon.hpp:130:11: note: because 'int' does
      not satisfy 'floating_point'
  130 |         requires std::floating_point&lt;T&gt;;
      |                  ^
.../c++/16/concepts:118:30: note: because
      'is_floating_point_v&lt;int&gt;' evaluated to false
1 error generated.
</pre></td>
<td><pre>
/usr/bin/ld.bfd: /tmp/err_free-d814c7.o:
      in function `main':
err_free.cpp:(.text+0x11c): undefined reference to
      `carbon::tensor&lt;int&gt;
       carbon::exp&lt;int&gt;(carbon::tensor&lt;int&gt; const&amp;)'
clang++: error: linker command failed with
      exit code 1 (use -v to see invocation)
</pre></td>
</tr>
</table>

---

## 3. Non-member function templates need their own instantiation lines

These declarations inside the class:

```cpp
template <typename U>
friend std::ostream &operator<<(std::ostream &os, const tensor<U> &t);
template <typename U>
friend tensor<U> gemm(const tensor<U> &inp1, const tensor<U> &inp2);
```

make *separate namespace-scope function templates* friends of `tensor<T>`. They aren't members, so `template class tensor<f32>;` doesn't instantiate them. That's why the list carries 4× `operator<<`, 4× `gemm`, 4× `gemv` and 4× `dot`, on top of the four `template class` lines.
- `template <typename U> friend` befriends **every** specialization, so `tensor<f32>` grants friendship to `gemm<i64>`. That's wider than needed, but harmless.
- `operator<<` is declared *only* as a friend (no namespace-scope declaration in the header), so the only way to find it is argument-dependent lookup. That's fine for `std::cout << t`. The free math functions, by contrast, are declared at namespace scope (`carbon.hpp:31–54`) before being befriended.

---

## 4. Three ways the code becomes type-dependent in carbon

carbon uses three different mechanisms to make behavior depend on `T`. Putting them side by side:

| mechanism | where | decided | "wrong" type gets… | granularity |
|---|---|---|---|---|
| `requires` on a member | `exp_`, `randn`, `randint`, … (`carbon.hpp:106–143`) | overload resolution at the **call site** | no member: compile error | one member exists or doesn't |
| `if constexpr` | `operator<<` (`carbon.cpp:361–363`) | inside **one function body**, per instantiation | nothing is wrong; the statement is skipped | one statement |
| full specialization | `blas<f32>`, `blas<f64>` (`blas.hpp:119–191`) | **which class** is used | the primary template (generic loops) | whole implementation |

```cpp
// 1. requires: the member exists only for some T
void exp_() requires std::floating_point<T>;

// 2. if constexpr: one body; the discarded branch is never instantiated for ints
if constexpr (std::floating_point<U>) {
	os << std::fixed << std::setprecision(7);
}

// 3. full specialization: a completely different implementation per T
template<typename T> struct blas { static void gemm(...) { /* triple loop */ } };
#ifdef CARBON_USE_BLAS
template<> struct blas<f32> { static void gemm(...) { cblas_sgemm(...); } };
template<> struct blas<f64> { static void gemm(...) { cblas_dgemm(...); } };
#endif
```

Note that the specialization is also gated by the preprocessor. With `CARBON_USE_BLAS` off, `f32`/`f64` fall back to the primary template too, so the generic kernels are what `i32`/`i64` always run, and what everyone runs in a non-BLAS build.

---

## 5. Payoffs of this design: implementation hiding and better build times
carbon can hide BLAS only because the definitions live in the `.cpp`:

- `carbon.hpp:4` reads `// #include "blas.hpp"`. The header never mentions CBLAS.
- `blas.hpp` is included by exactly one file, `carbon.cpp:2`.
- In `CMakeLists.txt:66–68`, everything about BLAS is `PRIVATE`:
  ```cmake
  target_compile_definitions(carbon PRIVATE CARBON_USE_BLAS)
  target_include_directories(carbon PRIVATE ${CBLAS_INCLUDE_DIR})
  target_link_libraries(carbon PRIVATE BLAS::BLAS)
  ```
- Users of `carbon.hpp` never see `<cblas.h>`, never need `-I/usr/include/openblas`, never see the `CBLAS_*` enums, and never see the `CARBON_USE_BLAS` macro. Switching OpenBLAS ↔ BLIS (`BLA_VENDOR`) means rebuilding carbon and relinking users, with no source changes on their side.

**BLAS is hidden at compile time, not at link time.** carbon is a `STATIC` library, and a `.a` can't carry its own dependencies, so CMake still adds `PRIVATE` link dependencies to consumers' link lines. In `build/CMakeFiles/main.dir/link.txt`:

```
/usr/bin/clang++ ... main.cpp.o -o main  libcarbon.a /usr/lib64/libopenblas.so /usr/lib64/libtbb.so.12.17
```

`libtbb` is there for a different reason: TBB and `_GLIBCXX_USE_TBB_PAR_BACKEND` are `PUBLIC`, so they reach consumers deliberately.

**The build-time payoff, measured.** I compared a small user TU (`randn`, `exp_`, `gemm`, `+`, `sigmoid` on `tf32`) two ways. "Split" includes `carbon.hpp` as-is. "Header-only proxy" includes all of `carbon.cpp`'s definitions as a header, minus the instantiation list, so only what's used gets instantiated. clang 22, `-O3`, `hyperfine`, 10 runs each:

| compile | mean | preprocessed lines |
|---|---:|---:|
| user TU, split (`carbon.hpp` only) | **0.66 s ± 0.03** | 71,470 |
| user TU, header-only proxy | **1.76 s ± 0.02** (2.7× slower) | 165,556 |
| `carbon.cpp` itself (all 4 types, paid once by the library build) | 9.46 s ± 0.28 | n/a |

These results are for one small TU; a TU that used more of the API would widen the gap. 


---

## 6. Tradeoffs

**Gains**
- Faster user builds: 0.66 s vs 1.76 s for the measured TU 
- A hidden implementation: no CBLAS in the public interface 
- One place to read every definition, and a header that reads like an API reference

**Costs**
- Link-time errors for anything missing, and they all look the same.
- **A hand-maintained list:** 44 lines today. One more floating-point type costs **14 lines**: `template class`, `operator<<`, `dot`, `gemm`, `gemv`, and 9 unary functions.
- Placement sensitivity: the list must come after the definitions, or the compiler silently builds an almost empty library.




A closed set of types is the price of splitting a template into a header and a `.cpp`: carbon supports exactly the types I wrote lines for, and every other type fails at compile time or link time, depending on how you ask.

---

## Appendix: type-support matrix

`✓` builds and links · **C** compile-time error (constraint not satisfied) · **L** link-time error (`undefined reference`)

**Members.** All four types get `template class tensor<T>;`. Availability comes from `requires` clauses and from whether a definition exists.

| member | f32 | f64 | i32 | i64 | decided by |
|---|:-:|:-:|:-:|:-:|---|
| constructors, `data` / `shape` / `strides` / `numel` / `ndim`, `at`, `begin` / `end` | ✓ | ✓ | ✓ | ✓ | always |
| `fill`, `zeros`, `ones`, `eye`, `linspace`, `arange` | ✓ | ✓ | ✓ | ✓ | always |
| `relu_`, `square_`, `cube_` | ✓ | ✓ | ✓ | ✓ | always |
| `==`, `!=`, `+`, `-`, `*`, `/` (tensor and scalar) | ✓ | ✓ | ✓ | ✓ | always |
| `log_`, `exp_`, `sigmoid_`, `sin_`, `cos_`, `tanh_`, `pow_` | ✓ | ✓ | C | C | `requires std::floating_point<T>` |
| `rand_normal`, `randn`, `randu` | ✓ | ✓ | C | C | `requires std::floating_point<T>` |
| `randn_he`, `randu_he`, `randn_xavier`, `randu_xavier` | ✓ | ✓ | C | C | `requires std::floating_point<T>` |
| `randint` | C | C | ✓ | ✓ | `requires std::integral<T>` |
| `reshape`, `transpose`, `slice`, `contiguous`, `clone` | L | L | L | L | declared, never defined |
| `sum`, `mean`, `max`, `softmax`, `layernorm` | L | L | L | L | declared, never defined |

**Free function templates.** Availability comes **only** from explicit instantiation lines.

| function | f32 | f64 | i32 | i64 | note |
|---|:-:|:-:|:-:|:-:|---|
| `operator<<` | ✓ | ✓ | ✓ | ✓ | precision set via `if constexpr` |
| `dot` | ✓ | ✓ | ✓ | ✓ | f32/f64 → `cblas_?dot`; ints → generic |
| `gemm`, `gemv` | ✓ | ✓ | ✓⚠ | ✓⚠ | ⚠ generic kernel currently wrong (`is_trans` uses `t = CblasTrans`, so it always returns true); f32/f64 hit it too without `CARBON_USE_BLAS` |
| `relu`, `square`, `cube` | ✓ | ✓ | ✓ | ✓ | |
| `sin`, `cos`, `tanh`, `log`, `exp`, `sigmoid` | ✓ | ✓ | L | L | would compile for ints; left out by choice |
| `matmul` | L | L | L | L | declared + befriended, never defined |

---

