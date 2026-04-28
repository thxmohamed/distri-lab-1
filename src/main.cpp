#include <iostream>
#include <fstream>
#include <string>
#include <vector>
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

            std::vector<int> threads_list = {1, 2, 4, 8};

            // =====================================================
            // 1) Benchmark de simulacion completa
            // =====================================================
            std::ofstream sim_out("benchmark_results.dat");
            sim_out << "# threads mean stddev speedup speedup_error efficiency efficiency_error\n";

            double T1_sim = 0.0;
            double sigma_T1_sim = 0.0;

            for (int threads : threads_list) {
                omp_set_num_threads(threads);

                Result result = Benchmark::measureSimulation(N, steps, reps);

                if (threads == 1) {
                    T1_sim = result.mean;
                    sigma_T1_sim = result.stddev;
                }

                double Sp = Benchmark::speedup(T1_sim, result.mean);
                double sigmaSp = Benchmark::speedupError(
                    T1_sim, sigma_T1_sim,
                    result.mean, result.stddev
                );

                double Ep = Benchmark::efficiency(Sp, threads);
                double sigmaEp = Benchmark::efficiencyError(sigmaSp, threads);

                sim_out << threads << " "
                        << result.mean << " "
                        << result.stddev << " "
                        << Sp << " "
                        << sigmaSp << " "
                        << Ep << " "
                        << sigmaEp << "\n";

                std::cout << "[simulate] Threads: " << threads
                          << " | Mean: " << result.mean
                          << " | Speedup: " << Sp
                          << " ± " << sigmaSp
                          << " | Efficiency: " << Ep
                          << " ± " << sigmaEp
                          << std::endl;
            }

            sim_out.close();

            // =====================================================
            // 2) Benchmark solo de aceleraciones
            // =====================================================
            std::ofstream acc_out("accelerations_results.dat");
            acc_out << "# threads mean stddev speedup speedup_error efficiency efficiency_error\n";

            int schedule_type = 0; // static
            int chunk_size = 32;

            double T1_acc = 0.0;
            double sigma_T1_acc = 0.0;

            std::vector<double> acc_speedups;

            for (int threads : threads_list) {
                omp_set_num_threads(threads);

                Result result = Benchmark::measureAccelerationsOnly(
                    N, steps, reps, schedule_type, chunk_size
                );

                if (threads == 1) {
                    T1_acc = result.mean;
                    sigma_T1_acc = result.stddev;
                }

                double Sp = Benchmark::speedup(T1_acc, result.mean);
                double sigmaSp = Benchmark::speedupError(
                    T1_acc, sigma_T1_acc,
                    result.mean, result.stddev
                );

                double Ep = Benchmark::efficiency(Sp, threads);
                double sigmaEp = Benchmark::efficiencyError(sigmaSp, threads);

                acc_speedups.push_back(Sp);

                acc_out << threads << " "
                        << result.mean << " "
                        << result.stddev << " "
                        << Sp << " "
                        << sigmaSp << " "
                        << Ep << " "
                        << sigmaEp << "\n";

                std::cout << "[accelerations] Threads: " << threads
                          << " | Mean: " << result.mean
                          << " | Speedup: " << Sp
                          << " ± " << sigmaSp
                          << " | Efficiency: " << Ep
                          << " ± " << sigmaEp
                          << std::endl;
            }

            acc_out.close();

            // =====================================================
            // 3) Comparacion de schedules
            // schedule_type: 0=static, 1=dynamic, 2=guided
            // =====================================================
            std::ofstream sched_out("schedule_results.dat");
            sched_out << "# schedule_type chunk threads mean stddev\n";

            int fixed_threads = 8;
            omp_set_num_threads(fixed_threads);

            for (int sched = 0; sched <= 2; sched++) {
                for (int chunk : {1, 8, 16, 32, 64, 128}) {
                    Result result = Benchmark::measureAccelerationsOnly(
                        N, steps, reps, sched, chunk
                    );

                    sched_out << sched << " "
                              << chunk << " "
                              << fixed_threads << " "
                              << result.mean << " "
                              << result.stddev << "\n";

                    std::cout << "[schedule] sched=" << sched
                              << " chunk=" << chunk
                              << " | Mean: " << result.mean
                              << std::endl;
                }
            }

            sched_out.close();

            // =====================================================
            // 4) Analisis de Amdahl
            // Usamos el speedup de 8 hilos para estimar f
            // =====================================================
            std::ofstream amdahl_out("scaling_analysis.dat");
            amdahl_out << "# threads measured_speedup estimated_f amdahl_prediction\n";

            double S8 = acc_speedups.back();
            double f_est = Benchmark::amdahlSerialFraction(S8, 8);

            for (size_t i = 0; i < threads_list.size(); i++) {
                int p = threads_list[i];
                double measured = acc_speedups[i];
                double predicted = Benchmark::amdahlSpeedup(f_est, p);

                amdahl_out << p << " "
                           << measured << " "
                           << f_est << " "
                           << predicted << "\n";
            }

            amdahl_out.close();

            std::cout << "\nArchivos generados:\n";
            std::cout << " - benchmark_results.dat\n";
            std::cout << " - accelerations_results.dat\n";
            std::cout << " - schedule_results.dat\n";
            std::cout << " - scaling_analysis.dat\n";

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