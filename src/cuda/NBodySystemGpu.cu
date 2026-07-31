#include "NBodySystem.h"

#include "CudaCheck.cuh"
#include "NBodyDeviceState.h"
#include "accelerations.cuh"

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>

// ================================================================
// Preparación y reutilización del estado CUDA
// ================================================================

/**
 * ---------------------------------------------------------------
 * prepareDeviceState
 * ---------------------------------------------------------------
 * Entrada:
 *  - include_velocities:
 *      true  = también actualiza vx y vy en device.
 *      false = actualiza solamente las posiciones.
 *
 * Salida:
 *  - Referencia al estado SoA persistente almacenado en GPU.
 *
 * Descripción:
 *  Prepara los buffers CUDA utilizados por el sistema.
 *
 *  En la primera operación GPU, o cuando cambia la composición del
 *  sistema, realiza una carga completa de masas, posiciones y
 *  velocidades.
 *
 *  En operaciones posteriores reutiliza los mismos buffers:
 *   - actualiza siempre las posiciones;
 *   - actualiza las velocidades solamente cuando la operación las
 *     necesita, por ejemplo, para calcular energía cinética.
 *
 *  De esta forma se evitan nuevas reservas y liberaciones de memoria
 *  CUDA en cada paso temporal.
 */
NBodyDeviceState& NBodySystem::prepareDeviceState(
    bool include_velocities
) {
    auto& bodies = getBodies();

    /*
     * El estado CUDA se crea de forma diferida. Esto permite que el
     * programa CPU del Lab 1 utilice NBodySystem sin reservar memoria
     * GPU mientras no se invoque una operación CUDA.
     */
    if (!device_state_) {
        device_state_ = std::make_shared<NBodyDeviceState>();
    }

    const bool requiresFullUpload =
        device_state_needs_full_upload_ ||
        !device_state_->isInitialized() ||
        device_state_->size() != bodies.size();

    if (requiresFullUpload) {
        device_state_->uploadInitialState(bodies);
        device_state_needs_full_upload_ = false;
    } else {
        device_state_->uploadPositions(bodies);
        if (include_velocities) {
            device_state_->uploadVelocities(bodies);
        }
    }

    return *device_state_;
}

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
 *  Sobrecarga que permite seleccionar la variante CUDA manteniendo
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
 *  Reutiliza el estado SoA persistente del sistema, actualiza las
 *  posiciones necesarias, ejecuta el kernel CUDA, sincroniza la GPU
 *  y descarga ax/ay hacia las partículas almacenadas en host.
 */
void NBodySystem::computeAccelerationsGpu(
    int variant,
    int block_size
) {
    if (variant < 0 || variant > 1) {
        throw std::invalid_argument(
            "NBodySystem::computeAccelerationsGpu: "
            "variant invalido."
        );
    }

    if (block_size <= 0) {
        throw std::invalid_argument(
            "NBodySystem::computeAccelerationsGpu: "
            "block_size debe ser positivo."
        );
    }

    auto& bodies = getBodies();

    /*
     * Un sistema vacío no requiere lanzamiento CUDA. Además, evita
     * intentar lanzar una grilla con cero bloques.
     */
    if (bodies.empty()) {
        return;
    }

    /*
     * El kernel de aceleraciones solamente necesita:
     *  - masas;
     *  - posiciones x/y;
     *  - buffers de salida ax/ay.
     *
     * Las velocidades no se actualizan porque este kernel no las usa.
     */
    NBodyDeviceState& deviceState =
        prepareDeviceState(false);

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

    /*
     * Una vez finalizado el kernel, se descargan únicamente las
     * aceleraciones. No se descargan masas, posiciones ni velocidades.
     */
    deviceState.downloadAccelerations(bodies);
}