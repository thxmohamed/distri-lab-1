#include "Benchmark.h"

#include "CudaCheck.cuh"
#include "NBodyDeviceState.h"
#include "accelerations.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

/**
 * Implementa los benchmarks GPU del simulador N-cuerpos: kernel-only
 * (sin transferencias), end-to-end (con transferencias) y comparación
 * directa contra la ruta CPU (OpenMP) para el mismo N.
 *
 * Temporización según el enunciado: std::chrono::steady_clock en host +
 * cudaDeviceSynchronize() antes de cerrar el cronómetro. No se usa
 * cudaEvent_t.
 */

namespace {

// Misma fórmula que Benchmark::calculateStats (Benchmark.cpp); se duplica
// aquí porque esa función tiene enlace interno y este archivo se compila
// por separado con nvcc.
Result calculateStatsGpu(const std::vector<double>& times) {
    double sum = 0.0;
    for (double t : times) {
        sum += t;
    }

    double mean = sum / times.size();

    double var = 0.0;
    for (double t : times) {
        double diff = t - mean;
        var += diff * diff;
    }

    if (times.size() > 1) {
        var /= (times.size() - 1);
    }

    return {mean, std::sqrt(var)};
}

AccelerationKernelVariant toKernelVariant(int variant) {
    if (variant < 0 || variant > 1) {
        throw std::invalid_argument("Benchmark: variant GPU invalido.");
    }

    return variant == 0
        ? AccelerationKernelVariant::Basic
        : AccelerationKernelVariant::Shared;
}

} // namespace

/**
 * ---------------------------------------------------------------
 * Benchmark::benchmarkKernelOnly
 * ---------------------------------------------------------------
 * Sube el estado a device una sola vez (fuera del cronómetro), lanza el
 * kernel de aceleraciones "steps" veces seguidas y sincroniza una sola vez
 * al final. Así se mide el costo puro de cómputo, sin transferencias.
 */
Result Benchmark::benchmarkKernelOnly(int N, int variant, int block_size,
                                      int steps, int repetitions) {
    const AccelerationKernelVariant kernelVariant = toKernelVariant(variant);

    std::vector<double> times;

    for (int r = 0; r < repetitions; r++) {
        NBodySystem system(1.0, 0.05);
        system.initDisk(N, 1.0, Benchmark::kSimulationSeed);

        NBodyDeviceState deviceState;
        deviceState.uploadInitialState(system.getBodies());
        deviceState.initializeAccelerationOutputs(0.0);

        auto start = std::chrono::steady_clock::now();

        for (int s = 0; s < steps; s++) {
            launchComputeAccelerations(
                deviceState.massData(),
                deviceState.positionXData(),
                deviceState.positionYData(),
                deviceState.accelerationXData(),
                deviceState.accelerationYData(),
                N,
                system.getG(),
                system.getEpsilon(),
                kernelVariant,
                block_size
            );
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        auto end = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double>(end - start).count());
    }

    return calculateStatsGpu(times);
}

/**
 * ---------------------------------------------------------------
 * Benchmark::benchmarkEndToEnd
 * ---------------------------------------------------------------
 * Cronometra NBodySystem::computeAccelerationsGpu() directamente: esa
 * función ya sube el estado, ejecuta el kernel, sincroniza y descarga el
 * resultado en cada llamada, por lo que el tiempo incluye transferencias.
 */
Result Benchmark::benchmarkEndToEnd(int N, int variant, int block_size,
                                    int steps, int repetitions) {
    std::vector<double> times;

    for (int r = 0; r < repetitions; r++) {
        NBodySystem system(1.0, 0.05);
        system.initDisk(N, 1.0, Benchmark::kSimulationSeed);

        auto start = std::chrono::steady_clock::now();

        for (int s = 0; s < steps; s++) {
            system.computeAccelerationsGpu(variant, block_size);
        }

        auto end = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double>(end - start).count());
    }

    return calculateStatsGpu(times);
}

/**
 * ---------------------------------------------------------------
 * Benchmark::compareCpuGpu
 * ---------------------------------------------------------------
 * Referencia CPU: misma ruta paralela usada en el resto de los benchmarks
 * (static, chunk=32). Referencia GPU: benchmarkEndToEnd, para comparar
 * tiempos "reales" de uso (con transferencias) y no solo de cómputo.
 */
CpuGpuComparison Benchmark::compareCpuGpu(int N, int variant, int block_size,
                                          int steps, int repetitions) {
    Result cpu = Benchmark::measureAccelerationsOnly(N, steps, repetitions, 0, 32);
    Result gpu = Benchmark::benchmarkEndToEnd(N, variant, block_size, steps, repetitions);

    double sp = Benchmark::speedup(cpu.mean, gpu.mean);
    double sigmaSp = Benchmark::speedupError(cpu.mean, cpu.stddev, gpu.mean, gpu.stddev);

    return {cpu, gpu, sp, sigmaSp};
}
