#include "metrics.cuh"

#include "CudaBuffer.h"
#include "CudaCheck.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

/**
 * @brief Kernel CUDA para calcular energía usando reducción por bloque.
 */
__global__ void calculateEnergyReductionKernel(
    const double* d_mass,
    const double* d_x,
    const double* d_y,
    const double* d_vx,
    const double* d_vy,
    double* d_partialK,
    double* d_partialU,
    int n,
    double gravitationalConstant,
    double epsilonSquared
) {
    extern __shared__ double sharedMemory[];

    double* sharedK = sharedMemory;
    double* sharedU = sharedMemory + blockDim.x;

    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    double kinetic = 0.0;
    double potential = 0.0;

    if (i < n) {
        const double vx = d_vx[i];
        const double vy = d_vy[i];

        kinetic =
            0.5 * d_mass[i] * (vx * vx + vy * vy);

        for (int j = i + 1; j < n; ++j) {
            const double dx = d_x[j] - d_x[i];
            const double dy = d_y[j] - d_y[i];

            const double distance =
                sqrt(dx * dx + dy * dy + epsilonSquared);

            potential +=
                -gravitationalConstant *
                d_mass[i] *
                d_mass[j] /
                distance;
        }
    }

    sharedK[tid] = kinetic;
    sharedU[tid] = potential;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedK[tid] += sharedK[tid + stride];
            sharedU[tid] += sharedU[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        d_partialK[blockIdx.x] = sharedK[0];
        d_partialU[blockIdx.x] = sharedU[0];
    }
}

/**
 * @brief Kernel CUDA para calcular energía usando atomicAdd.
 */
__global__ void calculateEnergyAtomicKernel(
    const double* d_mass,
    const double* d_x,
    const double* d_y,
    const double* d_vx,
    const double* d_vy,
    double* d_K,
    double* d_U,
    int n,
    double gravitationalConstant,
    double epsilonSquared
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    const double vx = d_vx[i];
    const double vy = d_vy[i];

    const double kinetic =
        0.5 * d_mass[i] * (vx * vx + vy * vy);

    double potential = 0.0;

    for (int j = i + 1; j < n; ++j) {
        const double dx = d_x[j] - d_x[i];
        const double dy = d_y[j] - d_y[i];

        const double distance =
            sqrt(dx * dx + dy * dy + epsilonSquared);

        potential +=
            -gravitationalConstant *
            d_mass[i] *
            d_mass[j] /
            distance;
    }

    atomicAdd(d_K, kinetic);
    atomicAdd(d_U, potential);
}

/**
 * @brief Ejecuta el cálculo de energía total en GPU.
 */
void launchCalculateEnergyGpu(
    const double* d_mass,
    const double* d_x,
    const double* d_y,
    const double* d_vx,
    const double* d_vy,
    int n,
    double gravitationalConstant,
    double epsilon,
    EnergyKernelMethod method,
    int blockSize,
    double* h_kineticEnergy,
    double* h_potentialEnergy
) {
    if (n < 0) {
        throw std::invalid_argument(
            "launchCalculateEnergyGpu: n no puede ser negativo.");
    }

    if (blockSize <= 0) {
        throw std::invalid_argument(
            "launchCalculateEnergyGpu: blockSize debe ser positivo.");
    }

    if (epsilon <= 0.0) {
        throw std::invalid_argument(
            "launchCalculateEnergyGpu: epsilon debe ser positivo.");
    }

    if (h_kineticEnergy == nullptr ||
        h_potentialEnergy == nullptr) {
        throw std::invalid_argument(
            "launchCalculateEnergyGpu: los punteros host de salida no pueden ser nulos.");
    }

    *h_kineticEnergy = 0.0;
    *h_potentialEnergy = 0.0;

    if (n == 0) {
        return;
    }

    if (d_mass == nullptr ||
        d_x == nullptr ||
        d_y == nullptr ||
        d_vx == nullptr ||
        d_vy == nullptr) {
        throw std::invalid_argument(
            "launchCalculateEnergyGpu: los punteros device no pueden ser nulos.");
    }

    const int gridSize =
        (n + blockSize - 1) / blockSize;

    const double epsilonSquared =
        epsilon * epsilon;

    if (method == EnergyKernelMethod::Reduction) {
        CudaBuffer<double> dPartialK(
            static_cast<std::size_t>(gridSize)
        );
        CudaBuffer<double> dPartialU(
            static_cast<std::size_t>(gridSize)
        );

        const std::size_t sharedMemoryBytes =
            2ULL *
            static_cast<std::size_t>(blockSize) *
            sizeof(double);

        calculateEnergyReductionKernel
            <<<gridSize, blockSize, sharedMemoryBytes>>>(
                d_mass,
                d_x,
                d_y,
                d_vx,
                d_vy,
                dPartialK.data(),
                dPartialU.data(),
                n,
                gravitationalConstant,
                epsilonSquared
            );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> partialK(
            static_cast<std::size_t>(gridSize),
            0.0
        );
        std::vector<double> partialU(
            static_cast<std::size_t>(gridSize),
            0.0
        );

        dPartialK.copyToHost(partialK.data(), partialK.size());
        dPartialU.copyToHost(partialU.data(), partialU.size());

        for (int i = 0; i < gridSize; ++i) {
            *h_kineticEnergy +=
                partialK[static_cast<std::size_t>(i)];

            *h_potentialEnergy +=
                partialU[static_cast<std::size_t>(i)];
        }

    } else if (method == EnergyKernelMethod::Atomic) {
        CudaBuffer<double> dK(1);
        CudaBuffer<double> dU(1);

        const double zero = 0.0;

        dK.copyFromHost(&zero, 1);
        dU.copyFromHost(&zero, 1);

        calculateEnergyAtomicKernel<<<gridSize, blockSize>>>(
            d_mass,
            d_x,
            d_y,
            d_vx,
            d_vy,
            dK.data(),
            dU.data(),
            n,
            gravitationalConstant,
            epsilonSquared
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        dK.copyToHost(h_kineticEnergy, 1);
        dU.copyToHost(h_potentialEnergy, 1);

    } else {
        throw std::invalid_argument(
            "launchCalculateEnergyGpu: metodo invalido.");
    }
}
