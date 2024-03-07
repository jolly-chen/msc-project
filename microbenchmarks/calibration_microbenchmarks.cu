#include <benchmark/benchmark.h>
#include "benchmark_kernels.cuh"
#include "utils.h"

using Clock = std::chrono::steady_clock;
using fsecs = std::chrono::duration<double>;

//
// CPU
//

static void BM_BinarySearch(benchmark::State &state) {
   long long bin;
   long long repetitions = 1000000;
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
   ->ArgsProduct({benchmark::CreateRange(8, 268435456, /*multi=*/2),
                  {0, 1, 2, 3, 4}}) // Args only accepts integer, so this is a hacky way to get [0, 0.5, 1]
   ->Unit(benchmark::kMillisecond);


BENCHMARK_MAIN();

static void BM_FixedSearch(benchmark::State &state) {
   long long bin;
   long long repetitions = 1e6;

   for (auto _ : state) {
      double elapsed_seconds = 0;
      for (int n = 0; n < repetitions; n++) {
         int nbins = rand();
         double x, xmin, xmax;
         x = xmin = xmax = rand();

         auto start = Clock::now();
         bin = 1 + int(nbins * (x - xmin) / (xmax - xmin));
         auto end = Clock::now();

         elapsed_seconds += std::chrono::duration_cast<fsecs>(end - start).count();
      }

      state.SetIterationTime(elapsed_seconds);
   }

   state.counters["repetitions"] = repetitions;
   state.counters["bin"] = bin;
}
BENCHMARK(BM_FixedSearch)
   ->UseManualTime()
   ->Unit(benchmark::kMillisecond);

static void BM_UpdateStats(benchmark::State &state)
{
   long long repetitions = 1e6;
   auto dim = state.range(0);
   std::vector<double> stats(2 + dim * 2 + dim * (dim - 1) / 2, 0);

   for (auto _ : state) {
      double elapsed_seconds = 0;
      for (int i = 0; i < repetitions; i++) {
         auto w = rand();
         std::vector<double> coords;
         for (auto i = 0; i < dim; i++)
            coords.emplace_back(rand());

         auto start = Clock::now();
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
         auto end = Clock::now();
         elapsed_seconds += std::chrono::duration_cast<fsecs>(end - start).count();
      }

      state.SetIterationTime(elapsed_seconds);
   }

   state.counters["repetitions"] = repetitions;
   state.counters["dim"] = dim;
}
BENCHMARK(BM_UpdateStats)
   ->Args({1})
   ->UseManualTime()
   ->Unit(benchmark::kMillisecond);

//
// GPU
//

