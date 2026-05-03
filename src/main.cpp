#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <omp.h>

#include "NBodySystem.h"
#include "NBodySimulator.h"
#include "Benchmark.h"
#include "MetricsCalculator.h"
#include "Visualizer.h"

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
            const int    N     = 50;
            const int    steps = 200;
            const double dt    = 0.001;

            NBodySystem sys(1.0, 0.05);
            sys.initBinary(N, 42);

            NBodySimulator sim(&sys, dt);
            Visualizer viz("energy_timeseries.dat", "snapshots.dat", "global_metrics.dat");

            for (int s = 0; s < steps; s++) {
                double t = s * dt;

                double K = MetricsCalculator::kineticEnergy(sys);
                double U = MetricsCalculator::potentialEnergy(sys);
                viz.recordEnergy(t, K, U);

                viz.recordMetrics(t,
                    MetricsCalculator::centerOfMassX(sys),
                    MetricsCalculator::centerOfMassY(sys),
                    MetricsCalculator::rmsRadius(sys),
                    MetricsCalculator::momentum(sys),
                    MetricsCalculator::minPairDistance(sys));

                if (s % 5 == 0)
                    viz.recordSnapshot(t, sys);

                sim.integrateEuler();
            }

            // =====================================================
            // Comparación de deriva energética para distintos Δt
            // Sistema: binario circular estable (2 cuerpos, M=20, d=1)
            //   v_circ = sqrt(G*M / (2*d)) = sqrt(10) ≈ 3.162
            //   T_orb  ≈ π / sqrt(10)      ≈ 0.993 ≈ 1.0
            // dt elegidos como fracciones del período: T/20, T/100, T/500
            // =====================================================
            const double t_fin_drift = 5.0;
            const double M_orb = 20.0;
            const double v_orb = std::sqrt(M_orb / 2.0);

            struct DtConfig { double dt; const char* filename; };
            DtConfig dt_configs[] = {
                {0.05,  "energy_drift_dt05.dat"},
                {0.01,  "energy_drift_dt01.dat"},
                {0.002, "energy_drift_dt002.dat"}
            };

            for (auto& cfg : dt_configs) {
                NBodySystem sys_dt(1.0, 0.05);
                sys_dt.addParticle(Particle(M_orb,  0.5, 0.0, 0.0,  v_orb));
                sys_dt.addParticle(Particle(M_orb, -0.5, 0.0, 0.0, -v_orb));

                NBodySimulator sim_dt(&sys_dt, cfg.dt);

                double E0 = MetricsCalculator::totalEnergy(sys_dt);

                std::ofstream drift_out(cfg.filename);
                drift_out << "# t E_rel  (dt=" << cfg.dt << ")\n";
                drift_out << std::scientific << std::setprecision(8);

                int n_steps = static_cast<int>(std::round(t_fin_drift / cfg.dt));
                for (int s = 0; s <= n_steps; s++) {
                    double t_s = s * cfg.dt;
                    double E   = MetricsCalculator::totalEnergy(sys_dt);
                    double E_rel = (E0 != 0.0) ? std::abs(E - E0) / std::abs(E0) : 0.0;
                    drift_out << t_s << " " << E_rel << "\n";
                    if (s < n_steps)
                        sim_dt.integrateEuler();
                }
            }

            std::cout << "Archivos generados:\n";
            std::cout << " - energy_timeseries.dat\n";
            std::cout << " - snapshots.dat\n";
            std::cout << " - global_metrics.dat\n";
            std::cout << " - energy_drift_dt05.dat\n";
            std::cout << " - energy_drift_dt01.dat\n";
            std::cout << " - energy_drift_dt002.dat\n";
            return 0;
        }
    }

    std::cout << "lab1_distri: build OK" << std::endl;
    return 0;
}