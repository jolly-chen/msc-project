#include <benchmark/benchmark.h>
#include <chrono>
#include <algorithm>
#include <random>

using Clock = std::chrono::steady_clock;
using fsecs = std::chrono::duration<double>;
constexpr int repetitions = 1e4;

//______________________________________________________________________________
//
// Calibration of Model 1.1
//______________________________________________________________________________Me!

static void BM_UpdateStats(benchmark::State &state)
{
   double sumw = 0, sumw2 = 0, sumwc = 0, sumwc2 = 0, sumccp = 0;

   for (auto _ : state) {
      auto w = rand();
      std::vector<double> coords;
      for (auto i = 0; i < state.range(0); i++)
         coords.emplace_back(rand());

      auto start = Clock::now();
      for (int i = 0; i < repetitions; i++) {
         sumw += w;
         sumw2 += w * w;
         for (auto ci = 0U; ci < coords.size(); ci++) {
            sumwc += w * coords[ci];
            sumwc2 += w * coords[ci] * coords[ci];
            for (auto cpi = 0U; cpi < ci; cpi++) {
               sumccp += w * coords[ci] * coords[cpi];
            }
         }
      }
      auto end = Clock::now();

      auto elapsed_seconds = std::chrono::duration_cast<fsecs>(end - start);
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

int nbins = 1024;
int multiplier = 32;
int step = 32;
static void BM_BinarySearch(benchmark::State &state)
{
   long long bin;
   int n = state.range(0);

   // Setup assumes binedges is in the cache and the val is in thel register.
   std::vector<double> binedges;
   for (auto i = 0; i < nbins; i++)
      binedges.emplace_back(i);
   auto a_binedges = binedges.data();

   for (auto _ : state) {
      auto start = Clock::now();
      for (long i = 0; i < repetitions; i++) {
         bin = BinarySearch(nbins, a_binedges, double(i % n));
      }
      auto end = Clock::now();

      auto elapsed_seconds = std::chrono::duration_cast<fsecs>(end - start);
      state.SetIterationTime(elapsed_seconds.count());
      static_cast<void>(bin); // prevent unused warnings
   }
}
BENCHMARK(BM_BinarySearch)->DenseRange(1, nbins, step)->UseManualTime()->Unit(benchmark::kMillisecond);
// BENCHMARK(BM_BinarySearch)->Range(1, nbins)->RangeMultiplier(multiplier)->UseManualTime()->Unit(benchmark::kMillisecond);

static void BM_BinarySearchStrided(benchmark::State &state)
{
   long long bin;
   int stride = state.range(0);

   // Setup assumes binedges is in the cache and the val is in thel register.
   std::vector<double> binedges;
   for (auto i = 0; i < nbins; i++)
      binedges.emplace_back(i);
   auto a_binedges = binedges.data();

   for (auto _ : state) {
      auto start = Clock::now();
      for (long i = 0; i < repetitions; i++) {
         bin = BinarySearch(nbins, a_binedges, double((i * stride) % nbins));
      }
      auto end = Clock::now();

      auto elapsed_seconds = std::chrono::duration_cast<fsecs>(end - start);
      state.SetIterationTime(elapsed_seconds.count());
      static_cast<void>(bin); // prevent unused warnings
   }
}
BENCHMARK(BM_BinarySearchStrided)->DenseRange(1, nbins, step)->UseManualTime()->Unit(benchmark::kMillisecond);
// BENCHMARK(BM_BinarySearchStrided)->Range(1, nbins)->RangeMultiplier(multiplier)->UseManualTime()->Unit(benchmark::kMillisecond);

// static void BM_Histogram_BestCase(benchmark::State &state)
// {
//    auto nbins = state.range(0);
//    auto histogram = (double *)malloc(nbins * sizeof(double));

//    for (auto _ : state) {
//       auto bin = 0;

//       auto start = Clock::now();
//       for (int i = 0; i < repetitions; i++)
//          histogram[bin] += 1.0;
//       auto end = Clock::now();

//       auto elapsed_seconds = std::chrono::duration_cast<fsecs>(end - start);
//       state.SetIterationTime(elapsed_seconds.count());
//    }

//    free(histogram);
// }

// BENCHMARK(BM_Histogram_BestCase)->RangeMultiplier(2)->Range(2, 2 << 8)->UseManualTime()->Unit(benchmark::kMillisecond);

// static void BM_Histogram_AverageCase(benchmark::State &state)
// {
//    auto nbins = state.range(0);
//    auto histogram = (double *)malloc(nbins * sizeof(double));

//    std::default_random_engine generator;
//    std::uniform_int_distribution<> distribution(0, nbins - 1);

//    for (auto _ : state) {
//       auto bin = distribution(generator);

//       auto start = Clock::now();
//       histogram[bin] += 1.0;
//       auto end = Clock::now();

//       auto elapsed_seconds = std::chrono::duration_cast<fsecs>(end - start);
//       state.SetIterationTime(elapsed_seconds.count());
//    }

//    free(histogram);
// }

// BENCHMARK(BM_Histogram_AverageCase)->RangeMultiplier(2)->Range(2, 2 <<
// 16)->UseManualTime()->Unit(benchmark::kMillisecond); BENCHMARK(BM_Histogram_AverageCase)->DenseRange(1, 32,
// 1)->UseManualTime()->Unit(benchmark::kMillisecond);

// static void BM_Histogram_WorstCase(benchmark::State &state)
// {
//    auto nbins = state.range(0);
//    auto histogram = (double *)malloc(nbins * sizeof(double));
//    auto stride = 8;
//    int bin = 0;

//    // Load the maximum amount into the L1 cache
//    for (int i = 0; i < nbins; i++) {
//       histogram[bin] += 1.0;
//    }

//    for (auto _ : state) {
//       bin = (bin + stride) % nbins;
//       printf("stride: %d\n", bin);
//       auto start = Clock::now();
//       // for (int i = 0; i < repetitions; i++) {
//       histogram[bin] += 1.0;
//       // }
//       auto end = Clock::now();

//       auto elapsed_seconds = std::chrono::duration_cast<fsecs>(end - start);
//       state.SetIterationTime(elapsed_seconds.count());
//    }

//    free(histogram);
// }

// BENCHMARK(BM_Histogram_WorstCase)->RangeMultiplier(2)->Range(2, 2 << 8)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
