#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>

#include "NBodySystem.h"
#include "NBodySimulator.h"
#include "Benchmark.h"

int main(int argc, char** argv) {
    if (argc > 1) {
        std::string arg = argv[1];

        if (arg == "--benchmark") {
            int N = 2000;
            int steps = 500;
            int reps = 10;

            std::ofstream out("benchmark_results.dat");
            out << "# threads mean stddev speedup efficiency\n";

            double T1 = 0.0;

            for (int threads : {1, 2, 4, 8}) {
                omp_set_num_threads(threads);

                auto result = Benchmark::measureSimulation(N, steps, reps);

                if (threads == 1) {
                    T1 = result.mean;
                }

                double Sp = Benchmark::speedup(T1, result.mean);
                double Ep = Benchmark::efficiency(Sp, threads);

                out << threads << " "
                    << result.mean << " "
                    << result.stddev << " "
                    << Sp << " "
                    << Ep << "\n";

                std::cout << "Threads: " << threads
                          << " | Mean: " << result.mean
                          << " | Stddev: " << result.stddev
                          << " | Speedup: " << Sp
                          << " | Efficiency: " << Ep
                          << std::endl;
            }

            out.close();
            return 0;
        }

        if (arg == "--analysis") {
            std::cout << "Modo analysis aun no implementado.\n";
            return 0;
        }
    }

    std::cout << "lab1_distri: build OK" << std::endl;
    return 0;
}