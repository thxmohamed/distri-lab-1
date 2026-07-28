#include "NBodySystem.h"

#include "CudaCheck.cuh"
#include "NBodyDeviceState.h"
#include "accelerations.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

// ================================================================
// Cálculo de aceleraciones GPU / CUDA
// ================================================================

/**
 * ---------------------------------------------------------------
 * computeAccelerationsGpu
 * ---------------------------------------------------------------
 * Entrada:
 *  - No recibe parámetros.
 *
 * Salida:
 *  - No retorna valor. Actualiza ax y ay de cada partícula.
 *
 * Descripción:
 *  Ruta CUDA por defecto. Usa kernel básico y block size 256.
 */
void NBodySystem::computeAccelerationsGpu() {
    computeAccelerationsGpu(0, 256);
}

/**
 * ---------------------------------------------------------------
 * computeAccelerationsGpu (variant)
 * ---------------------------------------------------------------
 * Entrada:
 *  - variant: 0=kernel básico, 1=kernel con shared memory.
 *
 * Salida:
 *  - No retorna valor. Actualiza ax y ay de cada partícula.
 *
 * Descripción:
 *  Sobrecarga que permite seleccionar variante CUDA manteniendo
 *  un block size por defecto de 256 hilos.
 */
void NBodySystem::computeAccelerationsGpu(int variant) {
    computeAccelerationsGpu(variant, 256);
}

/**
 * ---------------------------------------------------------------
 * computeAccelerationsGpu (variant, block_size)
 * ---------------------------------------------------------------
 * Entrada:
 *  - variant: 0=kernel básico, 1=kernel con shared memory.
 *  - block_size: cantidad de hilos CUDA por bloque.
 *
 * Salida:
 *  - No retorna valor. Deja las aceleraciones copiadas en host.
 *
 * Descripción:
 *  Copia el estado del sistema hacia device, ejecuta el kernel CUDA
 *  correspondiente, sincroniza la GPU y descarga ax/ay hacia las
 *  partículas del sistema.
 */
void NBodySystem::computeAccelerationsGpu(int variant, int block_size) {
    if (variant < 0 || variant > 1) {
        throw std::invalid_argument(
            "NBodySystem::computeAccelerationsGpu: variant invalido.");
    }

    if (block_size <= 0) {
        throw std::invalid_argument(
            "NBodySystem::computeAccelerationsGpu: block_size debe ser positivo.");
    }

    auto& bodies = getBodies();

    NBodyDeviceState deviceState;
    deviceState.uploadInitialState(bodies);
    deviceState.initializeAccelerationOutputs(0.0);

    const AccelerationKernelVariant kernelVariant =
        variant == 0
            ? AccelerationKernelVariant::Basic
            : AccelerationKernelVariant::Shared;

    launchComputeAccelerations(
        deviceState.massData(),
        deviceState.positionXData(),
        deviceState.positionYData(),
        deviceState.accelerationXData(),
        deviceState.accelerationYData(),
        getCount(),
        getG(),
        getEpsilon(),
        kernelVariant,
        block_size
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    deviceState.downloadAccelerations(bodies);
}
