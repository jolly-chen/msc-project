#include <benchmark/benchmark.h>
#include <chrono>
#include <algorithm>
#include <random>
#include <thread>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <new>

using Clock = std::chrono::steady_clock;
using fsecs = std::chrono::duration<double>;

/**
 * Returns aligned pointers when allocations are requested. Default alignment
 * is 64B = 512b, sufficient for AVX-512 and most cache line sizes.
 * Taken from: https://stackoverflow.com/questions/8456236/how-is-a-vectors-data-aligned
 *
 * @tparam ALIGNMENT_IN_BYTES Must be a positive power of 2.
 */
template <typename ElementType, std::size_t ALIGNMENT_IN_BYTES = 64>
class AlignedAllocator {
private:
   static_assert(ALIGNMENT_IN_BYTES >= alignof(ElementType),
                 "Beware that types like int have minimum alignment requirements "
                 "or access will result in crashes.");

public:
   using value_type = ElementType;
   static std::align_val_t constexpr ALIGNMENT{ALIGNMENT_IN_BYTES};

   /**
    * This is only necessary because AlignedAllocator has a second template
    * argument for the alignment that will make the default
    * std::allocator_traits implementation fail during compilation.
    * @see https://stackoverflow.com/a/48062758/2191065
    */
   template <class OtherElementType>
   struct rebind {
      using other = AlignedAllocator<OtherElementType, ALIGNMENT_IN_BYTES>;
   };

public:
   constexpr AlignedAllocator() noexcept = default;

   constexpr AlignedAllocator(const AlignedAllocator &) noexcept = default;

   template <typename U>
   constexpr AlignedAllocator(AlignedAllocator<U, ALIGNMENT_IN_BYTES> const &) noexcept
   {
   }

   [[nodiscard]] ElementType *allocate(std::size_t nElementsToAllocate)
   {
      if (nElementsToAllocate > std::numeric_limits<std::size_t>::max() / sizeof(ElementType)) {
         throw std::bad_array_new_length();
      }

      auto const nBytesToAllocate = nElementsToAllocate * sizeof(ElementType);
      return reinterpret_cast<ElementType *>(::operator new[](nBytesToAllocate, ALIGNMENT));
   }

   void deallocate(ElementType *allocatedPointer, [[maybe_unused]] std::size_t nBytesAllocated)
   {
      /* According to the C++20 draft n4868 § 17.6.3.3, the delete operator
       * must be called with the same alignment argument as the new expression.
       * The size argument can be omitted but if present must also be equal to
       * the one used in new. */
      ::operator delete[](allocatedPointer, ALIGNMENT);
   }
};
template<typename T, std::size_t ALIGNMENT_IN_BYTES = 64>
using AlignedVector = std::vector<T, AlignedAllocator<T, ALIGNMENT_IN_BYTES> >;

static void BM_PauseOverhead(benchmark::State &state) {
    for (auto _ : state) {
        state.PauseTiming();
        AlignedVector<double, 64> v(1e7);
        for (size_t i = 0; i < v.size(); i++) v[i] = rand();
        auto data = v.data();           // Allow v.data() to be clobbered. Pass as non-const
        benchmark::DoNotOptimize(data); // lvalue to avoid undesired compiler optimizations
        benchmark::ClobberMemory();     // Force data to be written to memory.

        state.ResumeTiming();

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}
BENCHMARK(BM_PauseOverhead)->Unit(benchmark::kSecond);

static void BM_ManualOverhead(benchmark::State &state) {
    for (auto _ : state) {
        AlignedVector<double, 64> v(1e7);
        for (size_t i = 0; i < v.size(); i++) v[i] = rand();
        auto data = v.data();           // Allow v.data() to be clobbered. Pass as non-const
        benchmark::DoNotOptimize(data); // lvalue to avoid undesired compiler optimizations
        benchmark::ClobberMemory();     // Force data to be written to memory.

        auto start = Clock::now();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        auto end = Clock::now();
        auto elapsed_seconds = std::chrono::duration_cast<fsecs>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
}
BENCHMARK(BM_ManualOverhead)->UseManualTime()->Unit(benchmark::kSecond);

template <typename T>
inline long long LinearSearch(T *arr, long long n, T val) {
    for (auto i = 0; i < n; i++) {
        if (arr[i] > val) {
            return i-1;
        }
    }
    return -1;
}

// prun -np 1 -v  likwid-pin -c C0:0  ./gbenchmark_evaluation --benchmark_filter=Linear --benchmark_repetitions=3 --benchmark_report_aggregates_only=yes --benchmark_perf_counters=INSTRUCTIONS,L1-dcache-load-misses,L1-dcache-loads,cache-misses,cache-references  --benchmark_counters_tabular=true --benchmark_format=json > das6-cpu/gbenchmark_evaluation.json
// prun -np 1 -v  likwid-pin -c C0:0  ./gbenchmark_evaluation --benchmark_filteLinear --benchmark_repetitions=3 --benchmark_report_aggregates_only=yes --benchmark_perf_counters=INSTRUCTIONS,L1-dcache-load-misses,L1-dcache-loads,cache-misses,cache-references  --benchmark_counters_tabular=true  &>> out
static void BM_LinearSearch(benchmark::State &state) {
    long long bin;
    long long repetitions = 1000;
    int nbins = state.range(0) / sizeof(double);  // Increasing histogram size
    double val = nbins * (state.range(1) / 4.);
    // double val = nbins;

    AlignedVector<double, 64> binedges(nbins);
    auto data = binedges.data();
    for (auto i = 0; i < nbins; i++) binedges[i] = i;

    for (auto _ : state) {
        // auto start = Clock::now();
        for (int n = 0; n < repetitions; n++) {
            // benchmark::DoNotOptimize(n);
            bin = LinearSearch(data, nbins, val);

            // for (auto i = 0; i < nbins; i++) {
            //    if (val > i) {
            //       break;
            //    }

            // data[i] = i*n;
            // static_cast<void>(bin); // prevent unused warnings

            // benchmark::ClobberMemory(); // Force 42 to be written to memory.
            // }
        }
        // auto end = Clock::now();

        // auto elapsed_seconds = std::chrono::duration_cast<fsecs>(end - start);
        // state.SetIterationTime(elapsed_seconds.count());
    }

    state.counters["repetitions"] = repetitions;
    state.counters["nbins"] = nbins;
    state.counters["val"] = val;
    state.counters["bin"] = bin;
}
BENCHMARK(BM_LinearSearch)
    // ->Args({64})        // 1 cache line
    // ->Args({32768})     // 512 lines     x512
    // ->Args({524288})    // 8192          x16
    // ->Args({16777216})  // 262144        x32
    // ->ArgsProduct({{64, 32768-640, 524288-640, 16777216-640, 16777216+640},
    ->ArgsProduct({benchmark::CreateRange(8, 33554432, /*multi=*/8),
                   {0, 1, 2, 3, 4}}) // Args only accepts integer, so this is a hacky way to get [0, 0.5, 1]
                //    benchmark::CreateRange(1, 256, /*multi=*/8),}) // Args only accepts integer, so this is a hacky way to get [0, 0.5, 1]
//    ->RangeMultiplier(4)
//    ->Range(8, 33554432) // bytes
    ->Unit(benchmark::kMillisecond)
    // ->UseManualTime()
;

// template <typename T>
// inline long long StridedAccess(T *arr, long long n, T stride) {
//     for (auto i = 0; i < n; i++) {
//         if (arr[i] > val) {
//             return i-1;
//         }
//     }
//     return -1;
// }

// static void BM_Strided(benchmark::State &state) {
//     long long bin;
//     long long nvals = 10;
//     int nbins = state.range(0) / sizeof(double);  // Increasing histogram size
//     double val = nbins * (state.range(1) / 4.);

//     // Setup assumes binedges is in the cache
//     // TODO: try aligning
//     // TODO: try pinning
//     // // TODO: try linear search? reuse distance

//     // if (reinterpret_cast<std::uintptr_t>(binedges.data() ) % 64 != 0 ) {
//     //    std::cerr << "Vector buffer is not aligned!\n";
//     // }
//     AlignedVector<double, 64> binedges(nbins);
//     auto data = binedges.data();
//     for (auto i = 0; i < nbins; i++) binedges[i] = i;

//     for (auto _ : state) {
//         // benchmark::DoNotOptimize(binedges);
//         // std::vector<double> binedges(nbins);

//         auto start = Clock::now();
//         for (int n = 0; n < 1; n++) {
//             // benchmark::DoNotOptimize(bin);
//             bin = LinearSearch(data, nbins, val);

//             // for (auto i = 0; i < nbins; i++) {
//             //    if (val > i) {
//             //       break;
//             //    }

//             // data[i] = i*n;
//             // static_cast<void>(bin); // prevent unused warnings

//             // benchmark::ClobberMemory(); // Force 42 to be written to memory.
//             // }
//         }
//         auto end = Clock::now();

//         auto elapsed_seconds = std::chrono::duration_cast<fsecs>(end - start);
//         state.SetIterationTime(elapsed_seconds.count());
//     }

//     state.counters["nbins"] = nbins;
//     state.counters["val"] = val;
//     state.counters["bin"] = bin;
// }
// BENCHMARK(BM_Strided)
//     // ->Args({64})        // 1 cache line
//     // ->Args({32768})     // 512 lines     x512
//     // ->Args({524288})    // 8192          x16
//     // ->Args({16777216})  // 262144        x32
//     ->ArgsProduct({{64, 32768-640, 524288-640, 16777216-640, 16777216+640},
//                    {0, 1, 2, 3, 4}}) // Args only accepts integer, so this is a hacky way to get [0, 0.5, 1]
//                 //    benchmark::CreateRange(1, 256, /*multi=*/8),}) // Args only accepts integer, so this is a hacky way to get [0, 0.5, 1]
//     ->Unit(benchmark::kMillisecond)
//     ->UseManualTime();

BENCHMARK_MAIN();

