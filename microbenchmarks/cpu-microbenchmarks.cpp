#include <benchmark/benchmark.h>
#include <chrono>
#include <algorithm>

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
      for (auto ci = 0; ci < coords.size(); ci++) {
         sumwc += w * coords[ci];
         sumwc2 += w * coords[ci] * coords[ci];
         for (auto cpi = 0; cpi < ci; cpi++) {
            sumccp += w * coords[ci] * coords[cpi];
         }
      }

      auto end = Clock::now();
      auto elapsed_seconds = std::chrono::duration_cast<fmsecs>(end - start);
      state.SetIterationTime(elapsed_seconds.count());
   }
}
BENCHMARK(BM_UpdateStats)->RangeMultiplier(2)->Range(2, 2 << 6)->UseManualTime()->Unit(benchmark::kMillisecond);;

// static void BM_Histogram() {}
// BENCHMARK(BM_Histogram);

BENCHMARK_MAIN();
