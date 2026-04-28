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

        // =========================================
        // MODO BENCHMARK
        // =========================================
        if (arg == "--benchmark") {
            int N = 2000;
            int steps = 500;
            int reps = 10;

            // -----------------------------------------
            // BENCHMARK 1: simulacion completa
            // -----------------------------------------
            std::ofstream sim_out("benchmark_results.dat");
            sim_out << "# threads mean stddev speedup efficiency\n";

            double T1_sim = 0.0;

            for (int threads : {1, 2, 4, 8}) {
                omp_set_num_threads(threads);

                auto result = Benchmark::measureSimulation(N, steps, reps);

                if (threads == 1) {
                    T1_sim = result.mean;
                }

                double Sp = Benchmark::speedup(T1_sim, result.mean);
                double Ep = Benchmark::efficiency(Sp, threads);

                sim_out << threads << " "
                        << result.mean << " "
                        << result.stddev << " "
                        << Sp << " "
                        << Ep << "\n";

                std::cout << "[simulate] Threads: " << threads
                          << " | Mean: " << result.mean
                          << " | Stddev: " << result.stddev
                          << " | Speedup: " << Sp
                          << " | Efficiency: " << Ep
                          << std::endl;
            }

            sim_out.close();

            // -----------------------------------------
            // BENCHMARK 2: solo aceleraciones
            // -----------------------------------------
            std::ofstream acc_out("accelerations_results.dat");
            acc_out << "# threads mean stddev speedup efficiency\n";

            int schedule_type = 0; // 0=static, 1=dynamic, 2=guided
            int chunk_size = 32;

            double T1_acc = 0.0;

            for (int threads : {1, 2, 4, 8}) {
                omp_set_num_threads(threads);

                auto result = Benchmark::measureAccelerationsOnly(
                    N, steps, reps, schedule_type, chunk_size
                );

                if (threads == 1) {
                    T1_acc = result.mean;
                }

                double Sp = Benchmark::speedup(T1_acc, result.mean);
                double Ep = Benchmark::efficiency(Sp, threads);

                acc_out << threads << " "
                        << result.mean << " "
                        << result.stddev << " "
                        << Sp << " "
                        << Ep << "\n";

                std::cout << "[accelerations] Threads: " << threads
                          << " | Mean: " << result.mean
                          << " | Stddev: " << result.stddev
                          << " | Speedup: " << Sp
                          << " | Efficiency: " << Ep
                          << std::endl;
            }

            acc_out.close();

            return 0;
        }

        // =========================================
        // MODO ANALYSIS
        // =========================================
        if (arg == "--analysis") {
            std::cout << "Modo analysis aun no implementado.\n";
            return 0;
        }
    }

    // Ejecucion normal
    std::cout << "lab1_distri: build OK" << std::endl;
    return 0;
}