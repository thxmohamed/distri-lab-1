#include "NBodySimulator.h"

#include "Integrator.h"

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
void NBodySimulator::stepEulerGpu(int variant, int block_size) {
    if (variant < 0 || variant > 1) {
        throw std::invalid_argument(
            "NBodySimulator::stepEulerGpu: variant invalido.");
    }

    if (block_size <= 0) {
        throw std::invalid_argument(
            "NBodySimulator::stepEulerGpu: block_size debe ser positivo.");
    }

    system_->computeAccelerationsGpu(variant, block_size);

    auto& bodies = system_->getBodies();

    Integrator::applyKick(bodies, time_step_);
    Integrator::applyDrift(bodies, time_step_);

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
 *  Ruta GPU por defecto. En este commit delega temporalmente en CPU;
 *  el cálculo CUDA real se agrega en el commit de métricas GPU.
 */
double NBodySimulator::calculateEnergyGpu() {
    return calculateEnergyGpu(0);
}

/**
 * ---------------------------------------------------------------
 * calculateEnergyGpu (method)
 * ---------------------------------------------------------------
 * Entrada:
 *  - method: 0=reducción, 1=atomicAdd.
 *
 * Salida:
 *  - Retorna la energía total.
 *
 * Descripción:
 *  Implementación temporal para mantener la interfaz compilable antes
 *  de agregar kernels CUDA de energía.
 */
double NBodySimulator::calculateEnergyGpu(int method) {
    if (method < 0 || method > 1) {
        throw std::invalid_argument(
            "NBodySimulator::calculateEnergyGpu: method invalido.");
    }

    return calculateEnergy(method);
}
