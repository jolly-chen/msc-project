#include <benchmark/benchmark.h>
#include <chrono>
#include <algorithm>
#include <random>
#include <thread>

using Clock = std::chrono::steady_clock;
using fsecs = std::chrono::duration<double>;
constexpr int repetitions = 1e4;

static void BM_Overhead(benchmark::State &state)
{
   for (auto _ : state) {
      auto start = Clock::now();
         std::this_thread::sleep_for(std::chrono::milliseconds(1));
      auto end = Clock::now();
      auto elapsed_seconds = std::chrono::duration_cast<fsecs>(end - start);
      state.SetIterationTime(elapsed_seconds.count());
   }
}
BENCHMARK(BM_Overhead)->UseManualTime()->Unit(benchmark::kMillisecond);

static void BM_OverheadSetup(benchmark::State &state)
{
   for (auto _ : state) {
      std::vector<double> coords;
      for (auto i = 0; i < 10; i++)
         coords.emplace_back(rand());

      auto start = Clock::now();
      for (int i = 0; i < repetitions; i++) {
         std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      auto end = Clock::now();
      auto elapsed_seconds = std::chrono::duration_cast<fsecs>(end - start);
      state.SetIterationTime(elapsed_seconds.count());
   }
}
BENCHMARK(BM_OverheadSetup)->UseManualTime()->Unit(benchmark::kMillisecond);

//______________________________________________________________________________
//
// Calibration of Model 1.1
//______________________________________________________________________________

static void BM_UpdateStats(benchmark::State &state)
{
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

static void BM_BinaryValidation(benchmark::State &state)
{
   long long bin;
   int nvals = 500e6;
   int nbins = state.range(0);

   // Setup assumes binedges is in the cache and the val is in thel register.
   std::vector<double> binedges;
   for (auto i = 0; i < nbins; i++)
      binedges.emplace_back(i * 1./nbins);
   auto a_binedges = binedges.data();


   for (auto _ : state) {
      double val = rand();
      auto start = Clock::now();
      for (long i = 0; i < nvals; i++) {
         bin = BinarySearch(nbins, a_binedges, val);
      }
      auto end = Clock::now();

      auto elapsed_seconds = std::chrono::duration_cast<fsecs>(end - start);
      state.SetIterationTime(elapsed_seconds.count());
      static_cast<void>(bin); // prevent unused warnings
   }
}
BENCHMARK(BM_BinaryValidation)
   ->Args({10})
   ->Args({1000})
   ->Args({1000000})
   ->Args({100000000})
   ->Unit(benchmark::kNanosecond);

static void BM_BinarySearchH12(benchmark::State &state)
{
   long long bin;
   long long nvals = 50e6;
   int nbins = state.range(0);
   double val = state.range(1)/4.;

   // Setup assumes binedges is in the cache
   // TODO: try aligning
   // TODO: try pinning
   // TODO: try linear search? reuse distance
   std::vector<double> binedges;
   for (auto i = 0; i < nbins; i++)
      binedges.emplace_back(i * 1./nbins);
   auto a_binedges = binedges.data();

   for (auto _ : state) {
      for (long i = 0; i < nvals; i++) {
         bin = BinarySearch(nbins, a_binedges, val);
      }
      static_cast<void>(bin); // prevent bin from being optimized away
   }
}
BENCHMARK(BM_BinarySearchH12)
   // ->ArgsProduct({{10, 1000, 100000, 10000000},
   // ->ArgsProduct({{8, 1024, 131072, 16777216}, // powers of 2
   ->ArgsProduct({{8, 1024, 4096, 131072, 65536, 3670016, 8388608, 16777216},
                  {0, 1, 2, 3, 4}}) // Args only accepts integer, so this is a hacky way to get [0, 0.5, 1]
   ->Unit(benchmark::kSecond);

static void BM_BinarySearchH3(benchmark::State &state)
{
   long long bin;
   int nbins = state.range(0);
   long long nvals = state.range(1) * 1e6;

   // Setup assumes binedges is in the cache
   std::vector<double> binedges;
   for (auto i = 0; i < nbins; i++)
      binedges.emplace_back(i * 1./nbins);
   auto a_binedges = binedges.data();


   for (auto _ : state) {
      for (long i = 0; i < nvals; i++) {
         bin = BinarySearch(nbins, a_binedges, 0.);
      }
      static_cast<void>(bin); // prevent bin from being optimized away
   }
}
BENCHMARK(BM_BinarySearchH3)
   // ->ArgsProduct({{10, 1000, 100000, 10000000},
   ->ArgsProduct({{8, 1024, 131072, 16777216}, // powers of 2
                  {50, 100, 500, 1000}})
   ->Unit(benchmark::kSecond);

static void BM_BinarySearchH4(benchmark::State &state)
{
   long long bin;
   int nbins = state.range(0);
   double range = state.range(1)/4.;
   int maxbin = nbins * range;
   long long nvals = 50e6/maxbin;
   double stride = 1./nbins;

   // Setup assumes binedges is in the cache
   std::vector<double> binedges;
   for (auto i = 0; i < nbins; i++)
      binedges.emplace_back(i * stride);
   auto a_binedges = binedges.data();

   for (auto _ : state) {
      for (long i = 0; i < nvals; i++) {
         for (long b = 0; b < maxbin; b++) {
            bin = BinarySearch(nbins, a_binedges, b*stride);
         }
      }
      static_cast<void>(bin); // prevent bin from being optimized away
   }
}
BENCHMARK(BM_BinarySearchH4)
   // ->ArgsProduct({{10, 1000, 100000, 10000000},
   ->ArgsProduct({{8, 1024, 131072, 16777216},
                  {1, 2, 3, 4}}) // Args only accepts integer, so this is a hacky way to get [0, 0.5, 1]
   ->Unit(benchmark::kSecond);


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
// BENCHMARK(BM_BinarySearch)->Range(1,
// nbins)->RangeMultiplier(multiplier)->UseManualTime()->Unit(benchmark::kMillisecond);

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
// BENCHMARK(BM_BinarySearchStrided)->Range(1,
// nbins)->RangeMultiplier(multiplier)->UseManualTime()->Unit(benchmark::kMillisecond);

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

// BENCHMARK(BM_Histogram_BestCase)->RangeMultiplier(2)->Range(2, 2 <<
// 8)->UseManualTime()->Unit(benchmark::kMillisecond);

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

// BENCHMARK(BM_Histogram_WorstCase)->RangeMultiplier(2)->Range(2, 2 <<
// 8)->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
