#pragma once

#include "CudaBuffer.h"
#include "Particle.h"

#include <cstddef>
#include <vector>

/**
 * @brief Mantiene el estado SoA del sistema N-cuerpos en memoria GPU.
 *
 * Cada atributo físico se almacena en un arreglo separado:
 * masa, posición, velocidad y aceleración.
 */
class NBodyDeviceState {
private:
    std::size_t count_ = 0;
    bool initialized_ = false;

    // Buffers device SoA.
    CudaBuffer<double> dMass_;
    CudaBuffer<double> dX_;
    CudaBuffer<double> dY_;
    CudaBuffer<double> dVx_;
    CudaBuffer<double> dVy_;
    CudaBuffer<double> dAx_;
    CudaBuffer<double> dAy_;

    // Arreglos host auxiliares para preparar y recuperar datos.
    std::vector<double> hMass_;
    std::vector<double> hX_;
    std::vector<double> hY_;
    std::vector<double> hVx_;
    std::vector<double> hVy_;
    std::vector<double> hAx_;
    std::vector<double> hAy_;

    void validateReady(
        const std::vector<Particle>& bodies,
        const char* operation
    ) const;

public:
    NBodyDeviceState() = default;

    explicit NBodyDeviceState(std::size_t count);

    /**
     * @brief Reserva todos los buffers para count partículas.
     */
    void resize(std::size_t count);

    /**
     * @brief Copia el estado inicial completo desde CPU hacia GPU.
     *
     * Esta operación incluye las masas, por lo que debe ejecutarse
     * al inicializar el sistema o cuando cambia la cantidad de cuerpos.
     */
    void uploadInitialState(const std::vector<Particle>& bodies);

    /**
     * @brief Copia solamente las posiciones actualizadas.
     */
    void uploadPositions(const std::vector<Particle>& bodies);

    /**
     * @brief Copia solamente las velocidades.
     *
     * Será utilizada por las métricas GPU cuando necesiten vx y vy.
     */
    void uploadVelocities(const std::vector<Particle>& bodies);

    /**
     * @brief Inicializa ax y ay con un valor conocido.
     *
     * Es útil en pruebas para detectar si un kernel dejó elementos sin escribir.
     */
    void initializeAccelerationOutputs(double value);

    /**
     * @brief Recupera ax y ay desde GPU y las guarda en las partículas CPU.
     *
     * El llamador debe sincronizar el device antes de ejecutar esta función.
     */
    void downloadAccelerations(std::vector<Particle>& bodies);

    std::size_t size() const noexcept {
        return count_;
    }

    bool isInitialized() const noexcept {
        return initialized_;
    }

    const double* massData() const noexcept {
        return dMass_.data();
    }

    const double* positionXData() const noexcept {
        return dX_.data();
    }

    const double* positionYData() const noexcept {
        return dY_.data();
    }

    const double* velocityXData() const noexcept {
        return dVx_.data();
    }

    const double* velocityYData() const noexcept {
        return dVy_.data();
    }

    double* accelerationXData() noexcept {
        return dAx_.data();
    }

    double* accelerationYData() noexcept {
        return dAy_.data();
    }
};