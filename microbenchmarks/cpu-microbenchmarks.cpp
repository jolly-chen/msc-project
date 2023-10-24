#include <benchmark/benchmark.h>
#include <chrono>
#include <algorithm>
#include <random>

using Clock = std::chrono::steady_clock;
using fmsecs = std::chrono::duration<double, std::chrono::milliseconds::period>;

//______________________________________________________________________________
//
// Calibration of Model 1.1
//______________________________________________________________________________

static void BM_UpdateStats(benchmark::State &state)
{
   double sumw = 0, sumw2 = 0, sumwc = 0, sumwc2 = 0, sumccp = 0;

   for (auto _ : state) {
      auto w = rand();
      std::vector<double> coords;
      for (auto i = 0; i < state.range(0); i++)
         coords.emplace_back(rand());

      auto start = Clock::now();
      sumw += w;
      sumw2 += w * w;
      for (auto ci = 0U; ci < coords.size(); ci++) {
         sumwc += w * coords[ci];
         sumwc2 += w * coords[ci] * coords[ci];
         for (auto cpi = 0U; cpi < ci; cpi++) {
            sumccp += w * coords[ci] * coords[cpi];
         }
      }
      auto end = Clock::now();

      auto elapsed_seconds = std::chrono::duration_cast<fmsecs>(end - start);
      state.SetIterationTime(elapsed_seconds.count());
   }
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

static void BM_BinarySearch(benchmark::State &state)
{
   long long bin;
   auto nbins = state.range(0);

   // Setup assumes binedges is in the cache and the val is in the register.
   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(0.0, nbins);
   std::vector<double> binedges;
   for (auto i = 0; i < nbins; i++)
      binedges.emplace_back(i);
   auto a_binedges = binedges.data();

   for (auto _ : state) {
      double val = distribution(generator);

      auto start = Clock::now();
      bin = BinarySearch(nbins, a_binedges, val);
      auto end = Clock::now();

      auto elapsed_seconds = std::chrono::duration_cast<fmsecs>(end - start);
      state.SetIterationTime(elapsed_seconds.count());
      static_cast<void>(bin); // prevent unused warnings
   }
}
BENCHMARK(BM_BinarySearch)->RangeMultiplier(2)->Range(2, 2 << 16)->UseManualTime()->Unit(benchmark::kMillisecond);

static void BM_Histogram(benchmark::State &state)
{
   auto nbins = state.range(0);
   auto histogram = (double *)malloc(nbins * sizeof(double));

   std::default_random_engine generator;
   std::uniform_int_distribution<> distribution(0, nbins - 1);

   for (auto _ : state) {
      auto bin = distribution(generator);

      auto start = Clock::now();
      histogram[bin] += 1.0;
      auto end = Clock::now();

      auto elapsed_seconds = std::chrono::duration_cast<fmsecs>(end - start);
      state.SetIterationTime(elapsed_seconds.count());
   }

   free(histogram);
}

// BENCHMARK(BM_Histogram)->RangeMultiplier(2)->Range(2, 2 << 16)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_Histogram)->DenseRange(1, 32, 1)->UseManualTime()->Unit(benchmark::kMillisecond);


BENCHMARK_MAIN();
