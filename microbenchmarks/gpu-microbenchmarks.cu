#include <benchmark/benchmark.h>
#include <chrono>
#include <algorithm>
#include <random>

using Clock = std::chrono::steady_clock;
using fmsecs = std::chrono::duration<double, std::chrono::milliseconds::period>;

static void BM_Malloc(benchmark::State &state)
{
   cudaDeviceSynchronize(); // dummy operation to start the CUDA runtime

   auto nbytes = state.range(0);
   void *ptr;

   for (auto _ : state) {
      auto start = Clock::now();
      cudaMalloc((void **)&ptr, nbytes);
      auto end = Clock::now();

      cudaFree(ptr);
      auto elapsed_seconds = std::chrono::duration_cast<fmsecs>(end - start);
      state.SetIterationTime(elapsed_seconds.count());
   }
}
BENCHMARK(BM_Malloc)->RangeMultiplier(2)->Range(8, 8 << 16)->UseManualTime()->Unit(benchmark::kMillisecond);

static void BM_HToD(benchmark::State &state)
{
   auto nbytes = state.range(0);
   auto data = malloc(nbytes);
   void *ptr;
   cudaMalloc((void **)&ptr, nbytes);

   for (auto _ : state) {
      auto start = Clock::now();
      cudaMemcpy(ptr, data, nbytes, cudaMemcpyHostToDevice);
      auto end = Clock::now();

      auto elapsed_seconds = std::chrono::duration_cast<fmsecs>(end - start);
      state.SetIterationTime(elapsed_seconds.count());
   }

   free(data);
   cudaFree(ptr);
}

BENCHMARK(BM_HToD)->RangeMultiplier(2)->Range(8, 8 << 16)->UseManualTime()->Unit(benchmark::kMillisecond);

static void BM_DToH(benchmark::State &state)
{
   auto nbytes = state.range(0);
   auto data = malloc(nbytes);
   void *ptr;
   cudaMalloc((void **)&ptr, nbytes);

   for (auto _ : state) {
      auto start = Clock::now();
      cudaMemcpy(data, ptr, nbytes, cudaMemcpyDeviceToHost);
      auto end = Clock::now();

      auto elapsed_seconds = std::chrono::duration_cast<fmsecs>(end - start);
      state.SetIterationTime(elapsed_seconds.count());
   }

   free(data);
   cudaFree(ptr);
}
BENCHMARK(BM_DToH)->RangeMultiplier(2)->Range(8, 8 << 16)->UseManualTime()->Unit(benchmark::kMillisecond);


BENCHMARK_MAIN();
