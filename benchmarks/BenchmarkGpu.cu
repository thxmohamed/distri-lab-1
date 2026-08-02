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
 * Implementa los benchmarks GPU del simulador N-cuerpos: kernel-only (sin
 * transferencias), aceleraciones con transferencias (sin integrar Euler),
 * un paso de simulación completo (end-to-end real), y comparación directa
 * contra la referencia CPU serial para el mismo N.
 *
 * Temporización con std::chrono::steady_clock en host + cudaDeviceSynchronize()
 * antes de cerrar el cronómetro. No se usa cudaEvent_t.
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
 * Benchmark::benchmarkAccelerationsWithTransfers
 * ---------------------------------------------------------------
 * Cronometra NBodySystem::computeAccelerationsGpu() directamente: esa
 * función ya sube el estado, ejecuta el kernel, sincroniza y descarga el
 * resultado en cada llamada, por lo que el tiempo incluye transferencias.
 * No integra Euler: las posiciones no cambian entre llamadas. Sirve para
 * aislar el costo de las transferencias frente a benchmarkKernelOnly.
 */
Result Benchmark::benchmarkAccelerationsWithTransfers(int N, int variant, int block_size,
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
 * Benchmark::benchmarkEndToEnd
 * ---------------------------------------------------------------
 * Cronometra NBodySimulator::stepEulerGpu(): aceleraciones en GPU (con
 * transferencias) + integración de Euler en host (kick + drift). A
 * diferencia de benchmarkAccelerationsWithTransfers, el estado sí avanza
 * entre llamadas, por lo que representa un paso de simulación real.
 */
Result Benchmark::benchmarkEndToEnd(int N, int variant, int block_size,
                                    int steps, int repetitions) {
    std::vector<double> times;

    for (int r = 0; r < repetitions; r++) {
        NBodySystem system(1.0, 0.05);
        system.initDisk(N, 1.0, Benchmark::kSimulationSeed);
        NBodySimulator simulator(&system, 0.01);

        auto start = std::chrono::steady_clock::now();

        for (int s = 0; s < steps; s++) {
            simulator.stepEulerGpu(variant, block_size);
        }

        auto end = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double>(end - start).count());
    }

    return calculateStatsGpu(times);
}

/**
 * ---------------------------------------------------------------
 * Benchmark::measureAccelerationsSerial
 * ---------------------------------------------------------------
 * Referencia CPU serial (Lab 1): computeAccelerationsSerial(), sin OpenMP.
 * Es la referencia de correctitud/desempeño exigida para comparar contra GPU.
 */
Result Benchmark::measureAccelerationsSerial(int N, int steps, int repetitions) {
    std::vector<double> times;

    for (int r = 0; r < repetitions; r++) {
        NBodySystem system(1.0, 0.05);
        system.initDisk(N, 1.0, Benchmark::kSimulationSeed);

        auto start = std::chrono::steady_clock::now();

        for (int s = 0; s < steps; s++) {
            system.computeAccelerationsSerial();
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
 * Referencia CPU: serial (measureAccelerationsSerial), sin OpenMP.
 * Referencia GPU: benchmarkEndToEnd, un paso de simulación completo.
 */
CpuGpuComparison Benchmark::compareCpuGpu(int N, int variant, int block_size,
                                          int steps, int repetitions) {
    Result cpu = Benchmark::measureAccelerationsSerial(N, steps, repetitions);
    Result gpu = Benchmark::benchmarkEndToEnd(N, variant, block_size, steps, repetitions);

    double sp = Benchmark::speedup(cpu.mean, gpu.mean);
    double sigmaSp = Benchmark::speedupError(cpu.mean, cpu.stddev, gpu.mean, gpu.stddev);

    return {cpu, gpu, sp, sigmaSp};
}
