#include <benchmark/benchmark.h>
#include "benchmark_kernels.cuh"
#include "utils.h"

using Clock = std::chrono::steady_clock;
using fsecs = std::chrono::duration<double>;

//
// CPU
//

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
// prun -np 1 -v -native '-C gpunode,A4000 --gres=gpu:1' calibration_microbenchmarks --benchmark_filter=GPU --benchmark_repetitions=3 --benchmark_report_aggregates_only=yes --benchmark_counters_tabular=true  --benchmark_format=json > das6/gpu_calibration.json
//

static void BM_BinarySearchGPU(benchmark::State &state) {
   constexpr long long repetitions = 1000;
   long nbins = state.range(0) / sizeof(double);  // Increasing histogram size
   size_t bulkSize = state.range(1);
   int blockSize = 256;
   int numBlocks = bulkSize % blockSize == 0 ? bulkSize / blockSize : bulkSize / blockSize + 1;

   std::random_device rd;  // a seed source for the random number engine
   std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
   std::uniform_int_distribution<> distrib(0, nbins);
   AlignedVector<double, 64> vals(bulkSize);
   for (auto i = 0; i < bulkSize; i++) vals[i] = distrib(gen);
   double *d_vals;
   ERRCHECK(cudaMalloc((void **)&d_vals, bulkSize * sizeof(double)));
   ERRCHECK(cudaMemcpy(d_vals, vals.data(), bulkSize * sizeof(double), cudaMemcpyHostToDevice));

   AlignedVector<double, 64> binedges(nbins);
   for (auto i = 0; i < nbins; i++) binedges[i] = i;
   double *d_binedges;
   ERRCHECK(cudaMalloc((void **)&d_binedges, nbins * sizeof(double)));
   ERRCHECK(cudaMemcpy(d_binedges, binedges.data(), nbins * sizeof(double), cudaMemcpyHostToDevice));

   //warmup
   BinarySearchGPU<<<numBlocks, blockSize>>>(nbins, d_binedges, d_vals, static_cast<double*>(0));

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   for (auto _ : state) {
      cudaEventRecord(start);
      for (auto i = 0; i < repetitions; i++) {
         BinarySearchGPU<<<numBlocks, blockSize>>>(nbins, d_binedges, d_vals, static_cast<double*>(0));
      }
      cudaEventRecord(stop);

      cudaEventSynchronize(stop);
      float elapsed_milliseconds;
      cudaEventElapsedTime(&elapsed_milliseconds, start, stop);
      state.SetIterationTime(elapsed_milliseconds/1e3);  // Iteration time needs to be set in seconds
   }

   state.counters["repetitions"] = repetitions;
   state.counters["nbins"] = nbins;
   state.counters["bulksize"] = bulkSize;
   state.counters["numblocks"] = numBlocks;
   ERRCHECK(cudaFree(d_binedges));
   ERRCHECK(cudaFree(d_vals));
   ERRCHECK(cudaEventDestroy(start));
   ERRCHECK(cudaEventDestroy(stop));
}
BENCHMARK(BM_BinarySearchGPU)
   ->ArgsProduct({benchmark::CreateRange(8, 268435456, /*multi=*/2), // Array size
                  benchmark::CreateRange(32, 262144, /*multi=*/4), // Bulksize
   })
   ->Unit(benchmark::kMicrosecond)
   ->UseManualTime()
   ->MinTime(1e-3); // repeat until at least a millisecond since the resolution of cudaEventRecord is 0.5 us

// prun -np 1 -v -native '-C gpunode,A4000 --gres=gpu:1' investigational_benchmarks --benchmark_filter=Histogram --benchmark_repetitions=3 --benchmark_report_aggregates_only=yes --benchmark_counters_tabular=true  --benchmark_format=json > das6/addbincontent_gpu.json
static void BM_HistogramGPU(benchmark::State &state) {
   constexpr long long repetitions = 1000;
   long nbins = state.range(0) / sizeof(double);  // Increasing histogram size
   size_t bulkSize = state.range(1);
   bool global = state.range(2) == 1 ? true : false;
   int blockSize = 256;
   int numBlocks = bulkSize % blockSize == 0 ? bulkSize / blockSize : bulkSize / blockSize + 1;
   auto smemSize = nbins * sizeof(double);

   int maxSmemSize;
   cudaDeviceGetAttribute(&maxSmemSize, cudaDevAttrMaxSharedMemoryPerBlock, 0);

   double *d_histogram;
   ERRCHECK(cudaMalloc((void **)&d_histogram, nbins * sizeof(double)));

   AlignedVector<int, 64> coords(bulkSize);
   for (auto i = 0; i < bulkSize; i++) coords[i] = 0;

   int *d_coords;
   ERRCHECK(cudaMalloc((void **)&d_coords, bulkSize * sizeof(int)));
   ERRCHECK(cudaMemcpy(d_coords, coords.data(), bulkSize * sizeof(int), cudaMemcpyHostToDevice));

   AlignedVector<double, 64> weights(bulkSize, 1);
   double *d_weights;
   ERRCHECK(cudaMalloc((void **)&d_weights, bulkSize * sizeof(double)));
   ERRCHECK(cudaMemcpy(d_weights, weights.data(), bulkSize * sizeof(double), cudaMemcpyHostToDevice));

   // Warmup to load the kernel
   if (global)
      HistogramGlobal<<<numBlocks, blockSize>>>(d_histogram, d_coords, d_weights, 0);
   else
      HistogramLocal<<<numBlocks, blockSize>>>(d_histogram, 0, d_coords, d_weights, 0);
      ERRCHECK(cudaPeekAtLastError());

   float elapsed_milliseconds;
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   if (global) {
      for (auto _ : state) {
         cudaEventRecord(start);
         for (auto i = 0; i < repetitions; i++)
            HistogramGlobal<<<numBlocks, blockSize>>>(d_histogram, d_coords, d_weights, bulkSize);
         cudaEventRecord(stop);
         ERRCHECK(cudaPeekAtLastError());

         cudaEventSynchronize(stop);
         cudaEventElapsedTime(&elapsed_milliseconds, start, stop);
         state.SetIterationTime(elapsed_milliseconds/1e3);  // Iteration time needs to be set in seconds
      }
   } else {
      if (smemSize < maxSmemSize) {
         for (auto _ : state) {
            cudaEventRecord(start);
            for (auto i = 0; i < repetitions; i++)
               HistogramLocal<<<numBlocks, blockSize, smemSize>>>(d_histogram, nbins, d_coords, d_weights, bulkSize);
            cudaEventRecord(stop);
            ERRCHECK(cudaPeekAtLastError());

            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed_milliseconds, start, stop);
            state.SetIterationTime(elapsed_milliseconds/1e3);  // Iteration time needs to be set in seconds
         }
      } else {
         state.SkipWithError("Does not fit in shared memory");
      }
   }

   state.counters["repetitions"] = repetitions;
   state.counters["nbins"] = nbins;
   state.counters["bulksize"] = bulkSize;
   state.counters["numblocks"] = numBlocks;
   state.counters["global"] = global ? 1 : 0;
   ERRCHECK(cudaFree(d_histogram));
   ERRCHECK(cudaFree(d_coords));
   ERRCHECK(cudaFree(d_weights));
   ERRCHECK(cudaEventDestroy(start));
   ERRCHECK(cudaEventDestroy(stop));
}
BENCHMARK(BM_HistogramGPU)
   ->ArgsProduct({benchmark::CreateRange(8, 268435456, /*multi=*/2), // Array size
                  benchmark::CreateRange(32, 262144, /*multi=*/4), // Bulksize
                  {1, 0},  // global, local
   })
   ->Unit(benchmark::kMicrosecond)
   ->UseManualTime()
   ->MinTime(1e-3); // repeat until at least a millisecond since the resolution of cudaEventRecord is 0.5 us

// prun -np 1 -v -native '-C gpunode,A4000 --gres=gpu:1' ./investigational_benchmarks --benchmark_filter=TransformReduceGPU --benchmark_repetitions=3 --benchmark_report_aggregates_only=yes --benchmark_counters_tabular=true --benchmark_format=json > das6/transformreduce_gpu.json
static void BM_TransformReduceGPU(benchmark::State &state) {
   constexpr long long repetitions = 10000;
   size_t numElements = state.range(0);
   int blockSize = state.range(1);
   int numThreads = (numElements < blockSize * 2) ? nextPow2((numElements + 1) / 2) : blockSize;
   int numBlocks = (numElements + (numThreads * 2 - 1)) / (numThreads * 2);

   AlignedVector<double, 64> data(numElements);
   for (auto i = 0; i < numElements; i++) data[i] = i;
   double *d_data;
   ERRCHECK(cudaMalloc((void **)&d_data, numElements * sizeof(double)));
   ERRCHECK(cudaMemcpy(d_data, data.data(), numElements * sizeof(double), cudaMemcpyHostToDevice));

   double *d_out;
   ERRCHECK(cudaMalloc((void **)&d_out, sizeof(double)));

   // warmup
   TransformReduce(numBlocks, blockSize, numElements, d_out, 0., true, Plus{}, Identity{}, d_data);

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   for (auto _ : state) {
      cudaEventRecord(start);
      for (auto i = 0; i < repetitions; i++) {
         TransformReduce(numBlocks, blockSize, numElements, d_out, 0., true, Plus{}, Identity{}, d_data);
      }
      cudaEventRecord(stop);

      cudaEventSynchronize(stop);
      float elapsed_milliseconds;
      cudaEventElapsedTime(&elapsed_milliseconds, start, stop);
      state.SetIterationTime(elapsed_milliseconds/1e3);
   }

   state.counters["repetitions"] = repetitions;
   state.counters["numelements"] = numElements;
   state.counters["numblocks"] = numBlocks;
   state.counters["numthreads"] = numThreads;
   state.counters["blocksize"] = blockSize;
   ERRCHECK(cudaFree(d_data));
   ERRCHECK(cudaFree(d_out));
   ERRCHECK(cudaEventDestroy(start));
   ERRCHECK(cudaEventDestroy(stop));
}
BENCHMARK(BM_TransformReduceGPU)
   ->ArgsProduct({
      benchmark::CreateRange(32, 262144, /*multi=*/8), // array size
      {256}, // blockSize
   })
   ->Unit(benchmark::kMicrosecond)
   ->UseManualTime()
   ->MinTime(1e-3); // repeat until at least a millisecond since the resolution of cudaEventRecord is 0.5 us

static void BM_MemcpyCPUToGPU(benchmark::State &state)
{
   constexpr long long repetitions = 300;
   auto nbytes = state.range(0);
   bool pinned = state.range(1) == 1 ? true : false;

   void *data;
   if (pinned)
      ERRCHECK(cudaMallocHost((void **)&data, nbytes));
   else
      data = malloc(nbytes);

   void *ptr;
   ERRCHECK(cudaMalloc((void **)&ptr, nbytes));

   // Warmup
   cudaMemcpy(ptr, data, nbytes, cudaMemcpyHostToDevice);

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   for (auto _ : state) {
      cudaEventRecord(start);
      // auto start = Clock::now();
      for (auto i = 0; i < repetitions; i++)
        cudaMemcpy(ptr, data, nbytes, cudaMemcpyHostToDevice);
      cudaEventRecord(stop);
      // auto end = Clock::now();

      cudaEventSynchronize(stop);
      float elapsed_milliseconds;
      cudaEventElapsedTime(&elapsed_milliseconds, start, stop);
      state.SetIterationTime(elapsed_milliseconds/1e3);

      // auto elapsed_seconds = std::chrono::duration_cast<fsecs>(end - start);
      // state.SetIterationTime(elapsed_seconds.count());
   }

   state.counters["nbytes"] = nbytes;
   state.counters["pinned"] = pinned ? 1 : 0;
   state.counters["repetitions"] = repetitions;
   cudaFreeHost(data);
   cudaFree(ptr);
   ERRCHECK(cudaEventDestroy(start));
   ERRCHECK(cudaEventDestroy(stop));
}
BENCHMARK(BM_MemcpyCPUToGPU)
   ->ArgsProduct({benchmark::CreateRange(1, 33554432, /*multi=*/2), // Array size
                  {1, 0},  // pinned, pageable
   })
   ->ArgsProduct({benchmark::CreateDenseRange(33554432, 268435456, /*step=*/int(268435456-33554432)/10), // Array size
                  {1, 0},  // pinned, pageable
   })
   ->MinTime(1e-3) // repeat until at least a millisecond since the resolution of cudaEventRecord is 0.5 us
   ->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
