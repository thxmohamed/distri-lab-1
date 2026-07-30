#include "Benchmark.h"
#include "NBodySystem.h"

#include <omp.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/**
 * Driver de la matriz de benchmarks GPU (sección 8 del enunciado Lab 2).
 *
 * Recorre N x variante x blockDim.x, midiendo tiempo de kernel-only y
 * end-to-end (blockdim_study.dat), y compara GPU contra CPU para el mismo N
 * (benchmark_results.dat). Pensado para correr UNA sola vez en el clúster
 * DIINF: cada combinación implica kRepetitions * kSteps llamadas al kernel.
 *
 * No forma parte del binario principal (lab1_distri, compilado con g++):
 * se compila aparte con nvcc via `make benchmark-gpu`, porque enlaza
 * directamente con código CUDA.
 */

namespace {

constexpr int kSteps = 100;         // Pasos por corrida (minimo del enunciado: >=100)
constexpr int kRepetitions = 10;    // Repeticiones por punto (minimo del enunciado: >=10)
constexpr int kDefaultBlockSize = 256;

const std::vector<int> kBodyCounts = {256, 512, 1024, 2000};
const std::vector<int> kVariants = {0, 1}; // 0=basico, 1=shared
const std::vector<int> kBlockSizes = {64, 128, 256, 512, 1024};

// Ejecuta un comando de sistema y agrega su salida a cluster_run.log, para
// documentar nodo/GPU/driver como pide la sección 8 del enunciado.
void appendCommandOutput(const std::string& label, const std::string& command) {
    std::ofstream log("cluster_run.log", std::ios::app);
    log << "\n# " << label << "\n$ " << command << "\n";
    log.close();

    if (std::system((command + " >> cluster_run.log 2>&1").c_str()) != 0) {
        std::ofstream errLog("cluster_run.log", std::ios::app);
        errLog << "(comando '" << label << "' termino con error, ver arriba)\n";
    }
}

} // namespace

int main() {
    const int cpuThreads = omp_get_max_threads();
    omp_set_num_threads(cpuThreads);

    std::ofstream header("cluster_run.log", std::ios::trunc);
    header << "# Matriz de benchmarks GPU - Lab 2\n"
           << "# seed=" << Benchmark::kSimulationSeed << "\n"
           << "# N in {256,512,1024,2000}, variant in {0=basico,1=shared}, "
              "blockDim.x in {64,128,256,512,1024}\n"
           << "# steps=" << kSteps << " repetitions=" << kRepetitions
           << " cpu_threads=" << cpuThreads << "\n";
    header.close();

    appendCommandOutput("nvidia-smi", "nvidia-smi");
    appendCommandOutput("nvcc --version", "nvcc --version");

    // =====================================================
    // 1) Estudio blockDim.x: N x variante x blockDim -> kernel-only y end-to-end
    // =====================================================
    std::ofstream blockdim_out("blockdim_study.dat");
    blockdim_out << "# seed=" << Benchmark::kSimulationSeed << "\n";
    blockdim_out << "# N variant block_size kernel_mean kernel_stddev endtoend_mean endtoend_stddev\n";

    for (int N : kBodyCounts) {
        for (int variant : kVariants) {
            for (int blockSize : kBlockSizes) {
                Result kernelOnly = Benchmark::benchmarkKernelOnly(
                    N, variant, blockSize, kSteps, kRepetitions
                );
                Result endToEnd = Benchmark::benchmarkEndToEnd(
                    N, variant, blockSize, kSteps, kRepetitions
                );

                blockdim_out << N << " " << variant << " " << blockSize << " "
                             << kernelOnly.mean << " " << kernelOnly.stddev << " "
                             << endToEnd.mean << " " << endToEnd.stddev << "\n";

                std::cout << "[blockdim] N=" << N << " variant=" << variant
                          << " block=" << blockSize
                          << " | kernel-only: " << kernelOnly.mean << "s"
                          << " | end-to-end: " << endToEnd.mean << "s"
                          << std::endl;
            }
        }
    }

    blockdim_out.close();

    // =====================================================
    // 2) Speedup GPU vs CPU vs N (block size por defecto, ambas variantes)
    // =====================================================
    std::ofstream results_out("benchmark_results.dat");
    results_out << "# seed=" << Benchmark::kSimulationSeed << "\n";
    results_out << "# cpu_threads=" << cpuThreads << "\n";
    results_out << "# N variant block_size cpu_mean cpu_stddev gpu_mean gpu_stddev speedup speedup_error\n";

    for (int N : kBodyCounts) {
        for (int variant : kVariants) {
            CpuGpuComparison comparison = Benchmark::compareCpuGpu(
                N, variant, kDefaultBlockSize, kSteps, kRepetitions
            );

            results_out << N << " " << variant << " " << kDefaultBlockSize << " "
                        << comparison.cpu.mean << " " << comparison.cpu.stddev << " "
                        << comparison.gpu.mean << " " << comparison.gpu.stddev << " "
                        << comparison.speedup << " " << comparison.speedupError << "\n";

            std::cout << "[speedup] N=" << N << " variant=" << variant
                      << " | CPU: " << comparison.cpu.mean << "s"
                      << " | GPU: " << comparison.gpu.mean << "s"
                      << " | Speedup: " << comparison.speedup
                      << " +/- " << comparison.speedupError
                      << std::endl;
        }
    }

    results_out.close();

    std::cout << "\nArchivos generados:\n"
              << " - blockdim_study.dat\n"
              << " - benchmark_results.dat\n"
              << " - cluster_run.log\n";

    return 0;
}
