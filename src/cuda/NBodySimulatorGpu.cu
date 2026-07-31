#include "NBodySimulator.h"

#include "Integrator.h"
#include "NBodyDeviceState.h"
#include "metrics.cuh"

#include <stdexcept>

// ================================================================
// Integración temporal GPU / CUDA
// ================================================================

/**
 * ---------------------------------------------------------------
 * stepEulerGpu
 * ---------------------------------------------------------------
 * Entrada:
 *  - No recibe parámetros.
 *
 * Salida:
 *  - No retorna valor. Avanza el sistema un paso temporal.
 *
 * Descripción:
 *  Ruta CUDA por defecto. Calcula aceleraciones con kernel básico
 *  y aplica Euler explícito en host.
 */
void NBodySimulator::stepEulerGpu() {
    stepEulerGpu(0, 256);
}

/**
 * ---------------------------------------------------------------
 * stepEulerGpu (variant, block_size)
 * ---------------------------------------------------------------
 * Entrada:
 *  - variant: 0=kernel básico, 1=kernel con shared memory.
 *  - block_size: cantidad de hilos CUDA por bloque.
 *
 * Salida:
 *  - No retorna valor. Actualiza velocidades, posiciones y tiempo.
 *
 * Descripción:
 *  Mantiene la integración Euler en host:
 *   1. calcula aceleraciones en GPU,
 *   2. sincroniza dentro de computeAccelerationsGpu(),
 *   3. aplica kick,
 *   4. aplica drift.
 */
void NBodySimulator::stepEulerGpu(
    int variant,
    int block_size
) {
    if (variant < 0 || variant > 1) {
        throw std::invalid_argument(
            "NBodySimulator::stepEulerGpu: variant invalido."
        );
    }

    if (block_size <= 0) {
        throw std::invalid_argument(
            "NBodySimulator::stepEulerGpu: "
            "block_size debe ser positivo."
        );
    }

    /*
     * computeAccelerationsGpu():
     *  - actualiza las posiciones necesarias en device;
     *  - ejecuta el kernel CUDA;
     *  - sincroniza;
     *  - descarga ax y ay hacia host.
     */
    system_->computeAccelerationsGpu(
        variant,
        block_size
    );

    auto& bodies = system_->getBodies();

    /*
     * Euler explícito permanece completamente en host,
     * respetando el orden físico del Lab 1:
     *
     * v <- v + a * dt
     * r <- r + v * dt
     */
    Integrator::applyKick(
        bodies,
        time_step_
    );

    Integrator::applyDrift(
        bodies,
        time_step_
    );

    current_time_ += time_step_;
}

// ================================================================
// Energía GPU / CUDA
// ================================================================

/**
 * ---------------------------------------------------------------
 * calculateEnergyGpu
 * ---------------------------------------------------------------
 * Entrada:
 *  - No recibe parámetros.
 *
 * Salida:
 *  - Retorna la energía total.
 *
 * Descripción:
 *  Ruta GPU por defecto. Usa reducción en shared memory.
 */
double NBodySimulator::calculateEnergyGpu() {
    return calculateEnergyGpu(0);
}

/**
 * ---------------------------------------------------------------
 * calculateEnergyGpu (method)
 * ---------------------------------------------------------------
 * Entrada:
 *  - method: 0=reducción en shared memory, 1=atomicAdd.
 *
 * Salida:
 *  - Retorna la energía total calculada como K + U.
 *
 * Descripción:
 *  Reutiliza el estado SoA persistente perteneciente a NBodySystem.
 *  Antes de ejecutar los kernels de energía actualiza posiciones y
 *  velocidades, pero no vuelve a transferir las masas si la
 *  composición del sistema no ha cambiado.
 */
double NBodySimulator::calculateEnergyGpu(int method) {
    if (method < 0 || method > 1) {
        throw std::invalid_argument(
            "NBodySimulator::calculateEnergyGpu: "
            "method invalido."
        );
    }

    /*
     * Para un sistema vacío:
     *  - K = 0;
     *  - U = 0;
     *  - E = 0.
     *
     * También se evita preparar buffers o lanzar una grilla vacía.
     */
    if (system_->getCount() == 0) {
        kinetic_energy_ = 0.0;
        potential_energy_ = 0.0;
        total_energy_ = 0.0;

        return total_energy_;
    }

    /*
     * Se reutiliza el estado device almacenado en NBodySystem.
     *
     * include_velocities = true porque el cálculo de energía requiere:
     *  - masas;
     *  - posiciones para U;
     *  - velocidades para K.
     *
     * En la primera llamada se realiza una carga completa.
     * En llamadas posteriores se actualizan solamente x, y, vx y vy.
     */
    NBodyDeviceState& deviceState =
        system_->prepareDeviceState(true);

    double kineticEnergy = 0.0;
    double potentialEnergy = 0.0;

    const EnergyKernelMethod energyMethod =
        method == 0
            ? EnergyKernelMethod::Reduction
            : EnergyKernelMethod::Atomic;

    /*
     * El block size fijo de 256 es válido para la reducción actual,
     * ya que es una potencia de dos.
     *
     * launchCalculateEnergyGpu() se encarga de:
     *  - lanzar el kernel correspondiente;
     *  - comprobar cudaGetLastError();
     *  - sincronizar el device;
     *  - recuperar K y U hacia host.
     */
    launchCalculateEnergyGpu(
        deviceState.massData(),
        deviceState.positionXData(),
        deviceState.positionYData(),
        deviceState.velocityXData(),
        deviceState.velocityYData(),
        system_->getCount(),
        system_->getG(),
        system_->getEpsilon(),
        energyMethod,
        256,
        &kineticEnergy,
        &potentialEnergy
    );

    kinetic_energy_ = kineticEnergy;
    potential_energy_ = potentialEnergy;
    total_energy_ =
        kinetic_energy_ + potential_energy_;

    return total_energy_;
}