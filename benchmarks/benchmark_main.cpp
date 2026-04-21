#include "Benchmark.h"
#include "NBodySystem.h"
#include "NBodySimulator.h"
#include <omp.h>
#include <fstream>
#include <iostream>

int main() {
    int N = 1000;
    int steps = 100;
    int reps = 10;

    std::ofstream out("benchmark_results.dat");

    // Cabecera para que el archivo se entienda mejor
    out << "# threads mean stddev speedup efficiency\n";

    double T1 = 0.0;

    for (int threads : {1, 2, 4, 8}) {
        omp_set_num_threads(threads);

        // Benchmark crea y reinicia el sistema en cada repetición
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