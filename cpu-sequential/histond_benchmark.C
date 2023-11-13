#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/Utils.hxx"
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleDS.hxx>
#include <ROOT/RLogger.hxx>
#include "TCanvas.h"

#include <benchmark/benchmark.h>

// this increases RDF's verbosity level as long as the `verbosity` variable is in scope
// auto verbosity =
//    ROOT::Experimental::RLogScopedVerbosity(ROOT::Detail::RDF::RDFLogChannel(), ROOT::Experimental::ELogLevel::kInfo);

int main(int argc, char **argv)
{
   size_t bulkSize = -1;
   int nbins = -1;
   bool verbose = false;
   bool edges = false;
   char *file;

   int c;
   while ((c = getopt(argc, argv, "b:h:f:ve")) != -1) {
      switch (c) {
      case 'b': bulkSize = std::stoul(optarg); break;
      case 'h': nbins = atoi(optarg); break;
      case 'f': file = optarg; break;
      case 'v': verbose = true; break;
      case 'e': edges = true; break;
      default: std::cout << "Ignoring unknown parse returns: " << char(c) << std::endl;
      }
   }

   if (argc < 3) {
      printf("The parameters are not set!!!\n");
      return 1;
   }

   TCanvas *cv;
   if (verbose) {
      printf("Bulksize: %lu Bins: %d File: %s\n", bulkSize, nbins, file);
      cv = new TCanvas("c", "", 200, 10, 700, 500);
   }

   ROOT::GetROOT();

   auto pageSource = ROOT::Experimental::Detail::RPageSource::Create("Data", file);
   auto df = ROOT::RDataFrame(std::make_unique<ROOT::Experimental::RNTupleDS>(std::move(pageSource)), {}, bulkSize);

   ROOT::RDF::TH1DModel mdl;
   if (edges) {
      std::vector<double> e(nbins + 1);
      for (int i = 0; i <= nbins; i++)
         e[i] = i * 1. / nbins;

      if (verbose) {
         printf("Edges:\n");
         for (int i = 0; i < e.size(); i++) {
            printf("%f ", e[i]);
         }
         printf("\n");
      }
      mdl = ROOT::RDF::TH1DModel("h1", "h1", nbins, e.data());
   } else {
      mdl = ROOT::RDF::TH1DModel("h1", "h1", nbins, 0, 1);
   }

   auto start = Clock::now();
   auto h1 = df.Histo1D<double>(mdl, "Doubles");
   h1.GetValue();
   auto end = Clock::now();

   if (verbose) {
      h1->Draw();
      cv->SaveAs("test.png");
      // free(cv);
   }

   printf("total:%f\n", std::chrono::duration_cast<fsecs>(end - start).count());

   return 0;
}
