#include <benchmark/benchmark.h>
#include "benchmark_kernels.cuh"
#include "utils.h"
#include <thread>

//
// CPU
//

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

// prun -np 1 -v  likwid-pin -c C0:0  ./investigational_benchmarks --benchmark_filter=Linear --benchmark_repetitions=3 --benchmark_report_aggregates_only=yes --benchmark_perf_counters=INSTRUCTIONS,L1-dcache-load-misses,L1-dcache-loads,cache-misses,cache-references  --benchmark_counters_tabular=true --benchmark_format=json > das6-cpu/gbenchmark_evaluation.json
// prun -np 1 -v  likwid-pin -c C0:0  ./investigational_benchmarks --benchmark_filteLinear --benchmark_repetitions=3 --benchmark_report_aggregates_only=yes --benchmark_perf_counters=INSTRUCTIONS,L1-dcache-load-misses,L1-dcache-loads,cache-misses,cache-references  --benchmark_counters_tabular=true  &>> out
static void BM_LinearSearch(benchmark::State &state) {
    long long bin;
    constexpr long long repetitions = 100;
    int nbins = state.range(0) / sizeof(double);  // Increasing histogram size
    double val = nbins * (state.range(1) / 4.);

    AlignedVector<double, 64> binedges(nbins);
    auto data = binedges.data();
    for (auto i = 0; i < nbins; i++) binedges[i] = i;

    for (auto _ : state) {
        for (int n = 0; n < repetitions; n++) {
            bin = LinearSearch(data, nbins, val);
        }
    }

    state.counters["repetitions"] = repetitions;
    state.counters["nbins"] = nbins;
    state.counters["val"] = val;
    state.counters["bin"] = bin;
}
BENCHMARK(BM_LinearSearch)
    ->ArgsProduct({benchmark::CreateRange(8, 33554432, /*multi=*/8),
                   {0, 1, 2, 3, 4}})
    ->Unit(benchmark::kMillisecond);


// prun -np 1 -v likwid-pin -c  C0:0 ./investigational_benchmarks --benchmark_filter=BM_BinarySearchCPU --benchmark_repetitions=3 --benchmark_report_aggregates_only=yes --benchmark_perf_counters=INSTRUCTIONS,L1-dcache-load-misses,L1-dcache-loads,cache-misses,cache-references  --benchmark_counters_tabular=true --benchmark_format=json > das6/binsearch_cpu.json
static void BM_BinarySearchCPU(benchmark::State &state) {
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
BENCHMARK(BM_BinarySearchCPU)
   ->ArgsProduct({benchmark::CreateRange(8, 268435456, /*multi=*/2),
                  {0, 1, 2, 3, 4}}) // Args only accepts integer, so this is a hacky way to get [0, 0.5, 1]
   ->Unit(benchmark::kMillisecond);

//
// GPU
//

//  prun -np 1 -v -native '-C gpunode,A4000 --gres=gpu:1' investigational_benchmarks --benchmark_filter=BinarySearchGPU --benchmark_repetitions=3 --benchmark_report_aggregates_only=yes --benchmark_counters_tabular=true  --benchmark_format=json > das6/binarysearch_gpu.json
static void BM_BinarySearchGPUConstant(benchmark::State &state) {
   constexpr long long repetitions = 1;
   long nbins = state.range(0) / sizeof(double);  // Increasing histogram size
   size_t bulkSize = state.range(1);
   int blockSize = state.range(2);
   int numBlocks = bulkSize % blockSize == 0 ? bulkSize / blockSize : bulkSize / blockSize + 1;
   double val = nbins * (state.range(3) / 4.);

   AlignedVector<double, 64> vals(bulkSize);
   for (auto i = 0; i < bulkSize; i++) vals[i] = val;
   double *d_vals;
   ERRCHECK(cudaMalloc((void **)&d_vals, bulkSize * sizeof(double)));
   ERRCHECK(cudaMemcpy(d_vals, vals.data(), bulkSize * sizeof(double), cudaMemcpyHostToDevice));

   AlignedVector<double, 64> binedges(nbins);
   auto data = binedges.data();
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
      state.SetIterationTime(elapsed_milliseconds/1e3); // Iteration time needs to be set in seconds
   }

   state.counters["repetitions"] = repetitions;
   state.counters["nbins"] = nbins;
   state.counters["val"] = val;
   state.counters["bulksize"] = bulkSize;
   state.counters["numblocks"] = numBlocks;
   state.counters["blocksize"] = blockSize;
   ERRCHECK(cudaFree(d_binedges));
   ERRCHECK(cudaFree(d_vals));
   ERRCHECK(cudaEventDestroy(start));
   ERRCHECK(cudaEventDestroy(stop));
}
BENCHMARK(BM_BinarySearchGPUConstant)
   ->ArgsProduct({benchmark::CreateRange(8, 268435456, /*multi=*/2), // Array size
                  benchmark::CreateRange(32, 262144, /*multi=*/2), // Bulkszie
                  benchmark::CreateRange(32, 1024, /*multi=*/2), // blockSize
                  {0, 2, 4},  // Args only accepts integer, so this is a hacky way to get [0, 0.25, 0.5, 0.75, 1]
   })
   ->Unit(benchmark::kMicrosecond)
   ->UseManualTime()
   ->MinTime(1e-3); // repeat until at least a millisecond since the resolution of cudaEventRecord is 0.5 us


static void BM_BinarySearchGPURandom(benchmark::State &state) {
   constexpr long long repetitions = 1;
   long nbins = state.range(0) / sizeof(double);  // Increasing histogram size
   size_t bulkSize = state.range(1);
   int blockSize = state.range(2);
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
   state.counters["blocksize"] = blockSize;
   ERRCHECK(cudaFree(d_binedges));
   ERRCHECK(cudaFree(d_vals));
   ERRCHECK(cudaEventDestroy(start));
   ERRCHECK(cudaEventDestroy(stop));
}
BENCHMARK(BM_BinarySearchGPURandom)
   ->ArgsProduct({benchmark::CreateRange(8, 268435456, /*multi=*/2), // Array size
                  benchmark::CreateRange(32, 262144, /*multi=*/2), // Bulkszie
                  benchmark::CreateRange(32, 1024, /*multi=*/2), // blockSize
   })
   ->Unit(benchmark::kMicrosecond)
   ->UseManualTime()
   ->MinTime(1e-3); // repeat until at least a millisecond since the resolution of cudaEventRecord is 0.5 us

// prun -np 1 -v -native '-C gpunode,A4000 --gres=gpu:1' investigational_benchmarks --benchmark_filter=Histogram --benchmark_repetitions=3 --benchmark_report_aggregates_only=yes --benchmark_counters_tabular=true  --benchmark_format=json > das6/addbincontent_gpu.json
static void BM_HistogramGPU(benchmark::State &state) {
   constexpr long long repetitions = 1000;
   long nbins = state.range(0) / sizeof(double);  // Increasing histogram size
   size_t bulkSize = state.range(1);
   int blockSize = state.range(2);
   bool gen_random = state.range(3) == 1 ? true : false;
   bool global = state.range(4) == 1 ? true : false;
   int numBlocks = bulkSize % blockSize == 0 ? bulkSize / blockSize : bulkSize / blockSize + 1;
   auto smemSize = nbins * sizeof(double);

   int maxSmemSize;
   cudaDeviceGetAttribute(&maxSmemSize, cudaDevAttrMaxSharedMemoryPerBlock, 0);

   double *d_histogram;
   ERRCHECK(cudaMalloc((void **)&d_histogram, nbins * sizeof(double)));

   AlignedVector<int, 64> coords(bulkSize);
   if (gen_random) {
      std::random_device rd;  // a seed source for the random number engine
      std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
      std::uniform_int_distribution<> distrib(0, nbins-1);
      for (auto i = 0; i < bulkSize; i++) coords[i] = distrib(gen);
   } else {
      for (auto i = 0; i < bulkSize; i++) coords[i] = 0;
   }
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
   state.counters["blocksize"] = blockSize;
   state.counters["random"] = gen_random ? 1 : 0;
   state.counters["global"] = global ? 1 : 0;
   ERRCHECK(cudaFree(d_histogram));
   ERRCHECK(cudaFree(d_coords));
   ERRCHECK(cudaFree(d_weights));
   ERRCHECK(cudaEventDestroy(start));
   ERRCHECK(cudaEventDestroy(stop));
}
BENCHMARK(BM_HistogramGPU)
   ->ArgsProduct({benchmark::CreateRange(8, 268435456, /*multi=*/2), // Array size
                  benchmark::CreateRange(32, 262144, /*multi=*/2), // Bulksize
                  benchmark::CreateRange(32, 1024, /*multi=*/2), // blockSize
                  {1, 0},  // 1 = random, 0 = constant
                  {1, 0},  // global, local
   })
   ->Unit(benchmark::kMicrosecond)
   ->UseManualTime()
   ->MinTime(1e-3); // repeat until at least a millisecond since the resolution of cudaEventRecord is 0.5 us

// prun -np 1 -v -native '-C gpunode,A4000 --gres=gpu:1' ./investigational_benchmarks --benchmark_filter=TransformReduceGPU --benchmark_repetitions=3 --benchmark_report_aggregates_only=yes --benchmark_counters_tabular=true --benchmark_format=json > das6/transformreduce_gpu.json
static void BM_TransformReduceGPU(benchmark::State &state) {
   constexpr long long repetitions = 10000;
   size_t bulkSize = state.range(0);
   int blockSize = state.range(1);
   int numThreads = (bulkSize < blockSize * 2) ? nextPow2((bulkSize + 1) / 2) : blockSize;
   int numBlocks = (bulkSize + (numThreads * 2 - 1)) / (numThreads * 2);

   AlignedVector<double, 64> data(bulkSize);
   for (auto i = 0; i < bulkSize; i++) data[i] = i;
   double *d_data;
   ERRCHECK(cudaMalloc((void **)&d_data, bulkSize * sizeof(double)));
   ERRCHECK(cudaMemcpy(d_data, data.data(), bulkSize * sizeof(double), cudaMemcpyHostToDevice));

   double *d_out;
   ERRCHECK(cudaMalloc((void **)&d_out, sizeof(double)));

   // warmup
   TransformReduce(numBlocks, blockSize, bulkSize, d_out, 0., true, Plus{}, Identity{}, d_data);

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   for (auto _ : state) {
      cudaEventRecord(start);
      for (auto i = 0; i < repetitions; i++) {
         TransformReduce(numBlocks, blockSize, bulkSize, d_out, 0., true, Plus{}, Identity{}, d_data);
      }
      cudaEventRecord(stop);

      cudaEventSynchronize(stop);
      float elapsed_milliseconds;
      cudaEventElapsedTime(&elapsed_milliseconds, start, stop);
      state.SetIterationTime(elapsed_milliseconds/1e3);
   }

   state.counters["repetitions"] = repetitions;
   state.counters["bulksize"] = bulkSize;
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
      benchmark::CreateRange(32, 262144, /*multi=*/2), // Bulkszie
      benchmark::CreateRange(32, 1024, /*multi=*/2), // blockSize
   })
   ->Unit(benchmark::kMicrosecond)
   ->UseManualTime()
   ->MinTime(1e-3); // repeat until at least a millisecond since the resolution of cudaEventRecord is 0.5 us

static void BM_DToH(benchmark::State &state)
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
   cudaMemcpy(data, ptr, nbytes, cudaMemcpyDeviceToHost);

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   for (auto _ : state) {
      // auto start = Clock::now();
      cudaEventRecord(start);
      for (auto i = 0; i < repetitions; i++)
        cudaMemcpy(data, ptr, nbytes, cudaMemcpyDeviceToHost);
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
BENCHMARK(BM_DToH)
   ->ArgsProduct({benchmark::CreateRange(1, 33554432, /*multi=*/2), // Array size
                  {1, 0},  // pinned, pageable
   })
   ->ArgsProduct({benchmark::CreateDenseRange(33554432, 268435456, /*step=*/int(268435456-33554432)/10), // Array size
                  {1, 0},  // pinned, pageable
   })
   ->MinTime(1e-3) // repeat until at least a millisecond since the resolution of cudaEventRecord is 0.5 us
   ->UseManualTime()->Unit(benchmark::kMicrosecond);

static void BM_HToD(benchmark::State &state)
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
BENCHMARK(BM_HToD)
   ->ArgsProduct({benchmark::CreateRange(1, 33554432, /*multi=*/2), // Array size
                  {1, 0},  // pinned, pageable
   })
   ->ArgsProduct({benchmark::CreateDenseRange(33554432, 268435456, /*step=*/int(268435456-33554432)/10), // Array size
                  {1, 0},  // pinned, pageable
   })
   ->MinTime(1e-3) // repeat until at least a millisecond since the resolution of cudaEventRecord is 0.5 us
   ->UseManualTime()->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();

