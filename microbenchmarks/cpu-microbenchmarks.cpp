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

//______________________________________________________________________________
//
// Calibration of Model 1.1
//______________________________________________________________________________

static void BM_UpdateStats(benchmark::State &state)
{
   long long repetitions = 1e4;
   auto dim = state.range(0);

   for (auto _ : state) {
      std::vector<double> stats(2 + dim * 2 + dim * (dim - 1) / 2, 0);
      auto w = rand();
      std::vector<double> coords;
      for (auto i = 0; i < dim; i++)
         coords.emplace_back(rand());

      auto start = Clock::now();
      for (int i = 0; i < repetitions; i++) {
         int offset = 2;
         stats[0] += w;
         stats[1] += w * w;
         for (auto ci = 0U; ci < coords.size(); ci++) {
            stats[offset++] += w * coords[ci];
            stats[offset++] += w * coords[ci] * coords[ci];
            for (auto cpi = 0U; cpi < ci; cpi++) {
               stats[offset++] += w * coords[ci] * coords[cpi];
            }
         }
      }
      auto end = Clock::now();

      auto elapsed_seconds = std::chrono::duration_cast<fsecs>(end - start);
      state.SetIterationTime(elapsed_seconds.count());
   }

   state.counters["repetitions"] = repetitions;
   state.counters["dim"] = dim;
}
BENCHMARK(BM_UpdateStats)
   ->Args({1})
   ->Args({2})
   ->Args({3})
   ->RangeMultiplier(2)
   ->Range(4, 2 << 6)
   ->UseManualTime()
   ->Unit(benchmark::kMillisecond);

/// Binary search taken from ROOT
template <typename T>
inline long long BinarySearch(long long n, const T *array, T value)
{
   const T *pind;
   pind = std::lower_bound(array, array + n, value);
   if ((pind != array + n) && (*pind == value))
      return (pind - array);
   else
      return (pind - array - 1);
}

// static void BM_BinarySearchH12(benchmark::State &state)
// {
//    long long bin;
//    long long nvals = 50e6;
//    int nbins = state.range(0);
//    double val = state.range(1)/4.;

//    // Setup assumes binedges is in the cache
//    // TODO: try aligning
//    // TODO: try pinning
//    // TODO: try linear search? reuse distance
//    std::vector<double> binedges;
//    for (auto i = 0; i < nbins; i++)
//       binedges.emplace_back(i * 1./nbins);
//    auto a_binedges = binedges.data();

//    for (auto _ : state) {
//       for (long i = 0; i < nvals; i++) {
//          bin = BinarySearch(nbins, a_binedges, val);
//       }
//       static_cast<void>(bin); // prevent bin from being optimized away
//    }
// }
// BENCHMARK(BM_BinarySearchH12)
//    // ->ArgsProduct({{10, 1000, 100000, 10000000},
//    // ->ArgsProduct({{8, 1024, 131072, 16777216}, // powers of 2
//    ->ArgsProduct({{8, 1024, 4096, 131072, 65536, 3670016, 8388608, 16777216},
//                   {0, 1, 2, 3, 4}}) // Args only accepts integer, so this is a hacky way to get [0, 0.5, 1]
//    ->Unit(benchmark::kSecond);

// static void BM_BinarySearchH3(benchmark::State &state)
// {
//    long long bin;
//    int nbins = state.range(0);
//    long long nvals = state.range(1) * 1e6;

//    // Setup assumes binedges is in the cache
//    std::vector<double> binedges;
//    for (auto i = 0; i < nbins; i++)
//       binedges.emplace_back(i * 1./nbins);
//    auto a_binedges = binedges.data();


//    for (auto _ : state) {
//       for (long i = 0; i < nvals; i++) {
//          bin = BinarySearch(nbins, a_binedges, 0.);
//       }
//       static_cast<void>(bin); // prevent bin from being optimized away
//    }
// }
// BENCHMARK(BM_BinarySearchH3)
//    // ->ArgsProduct({{10, 1000, 100000, 10000000},
//    ->ArgsProduct({{8, 1024, 131072, 16777216}, // powers of 2
//                   {50, 100, 500, 1000}})
//    ->Unit(benchmark::kSecond);

// static void BM_BinarySearchH4(benchmark::State &state)
// {
//    long long bin;
//    int nbins = state.range(0);
//    double range = state.range(1)/4.;
//    int maxbin = nbins * range;
//    long long nvals = 50e6/maxbin;
//    double stride = 1./nbins;

//    // Setup assumes binedges is in the cache
//    std::vector<double> binedges;
//    for (auto i = 0; i < nbins; i++)
//       binedges.emplace_back(i * stride);
//    auto a_binedges = binedges.data();

//    for (auto _ : state) {
//       for (long i = 0; i < nvals; i++) {
//          for (long b = 0; b < maxbin; b++) {
//             bin = BinarySearch(nbins, a_binedges, b*stride);
//          }
//       }
//       static_cast<void>(bin); // prevent bin from being optimized away
//    }
// }
// BENCHMARK(BM_BinarySearchH4)
//    // ->ArgsProduct({{10, 1000, 100000, 10000000},
//    ->ArgsProduct({{8, 1024, 131072, 16777216},
//                   {1, 2, 3, 4}}) // Args only accepts integer, so this is a hacky way to get [0, 0.5, 1]
//    ->Unit(benchmark::kSecond);

// static void BM_BinarySearchStrided(benchmark::State &state)
// {
//    long long bin;
//    int stride = state.range(0);

//    // Setup assumes binedges is in the cache and the val is in thel register.
//    std::vector<double> binedges;
//    for (auto i = 0; i < nbins; i++)
//       binedges.emplace_back(i);
//    auto a_binedges = binedges.data();

//    for (auto _ : state) {
//       auto start = Clock::now();
//       for (long i = 0; i < repetitions; i++) {
//          bin = BinarySearch(nbins, a_binedges, double((i * stride) % nbins));
//       }
//       auto end = Clock::now();

//       auto elapsed_seconds = std::chrono::duration_cast<fsecs>(end - start);
//       state.SetIterationTime(elapsed_seconds.count());
//       static_cast<void>(bin); // prevent unused warnings
//    }
// }
// BENCHMARK(BM_BinarySearchStrided)->DenseRange(1, nbins, step)->UseManualTime()->Unit(benchmark::kMillisecond);

// prun -np 1 -v likwid-pin -c  C0:0 ./cpu-microbenchmarks --benchmark_repetitions=3 --benchmark_report_aggregates_only=yes --benchmark_perf_counters=INSTRUCTIONS,L1-dcache-load-misses,L1-dcache-loads,cache-misses,cache-references  --benchmark_counters_tabular=true --benchmark_format=json > das6-cpu/cpu_microbenchmarks.json
static void BM_BinarySearch(benchmark::State &state) {
   long long bin;
   long long repetitions = 100000000;
   int nbins = state.range(0) / sizeof(double);  // Increasing histogram size
   // double val = nbins; // Last element
   double val = nbins * (state.range(1) / 4.);

   AlignedVector<double, 64> binedges(nbins);
   auto data = binedges.data();
   for (auto i = 0; i < nbins; i++) binedges[i] = i;

   for (auto _ : state) {
      for (int n = 0; n < repetitions; n++) {
         bin = BinarySearch(nbins, data, val);
      }
   }

   state.counters["repetitions"] = repetitions;
   state.counters["nbins"] = nbins;
   state.counters["val"] = val;
   state.counters["bin"] = bin;
}
BENCHMARK(BM_BinarySearch)
   // ->RangeMultiplier(4)
   // ->Range(8, 33554432) // bytes
   ->ArgsProduct({benchmark::CreateRange(8, 33554432, /*multi=*/8),
                  {0, 1, 2, 3, 4}}) // Args only accepts integer, so this is a hacky way to get [0, 0.5, 1]
   ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
