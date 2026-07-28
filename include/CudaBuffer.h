#pragma once
#include <string>
#include "CudaCheck.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

/**
 * @brief Administra un arreglo almacenado en memoria device mediante RAII.
 *
 * La memoria se reserva al construir o llamar allocate() y se libera
 * automáticamente cuando el objeto deja de existir.
 */
template <typename T>
class CudaBuffer {
private:
    T* devicePtr_ = nullptr;
    std::size_t size_ = 0;

    void validateHostCopy(
        const void* hostPtr,
        std::size_t count,
        const char* operation
    ) const {
        if (count > size_) {
            throw std::out_of_range(
                std::string(operation) +
                ": la cantidad solicitada supera el tamaño del buffer."
            );
        }

        if (count > 0 && hostPtr == nullptr) {
            throw std::invalid_argument(
                std::string(operation) +
                ": el puntero host no puede ser nulo."
            );
        }
    }

public:
    CudaBuffer() noexcept = default;

    explicit CudaBuffer(std::size_t count) {
        allocate(count);
    }

    ~CudaBuffer() {
        release();
    }

    // Un buffer CUDA no debe copiarse porque dos objetos terminarían
    // intentando liberar el mismo puntero device.
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // Se permite transferir la propiedad mediante movimiento.
    CudaBuffer(CudaBuffer&& other) noexcept
        : devicePtr_(std::exchange(other.devicePtr_, nullptr)),
          size_(std::exchange(other.size_, 0)) {
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            release();

            devicePtr_ = std::exchange(other.devicePtr_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }

        return *this;
    }

    /**
     * @brief Reserva espacio para count elementos.
     *
     * Si el buffer ya tenía otro tamaño, primero libera la memoria anterior.
     */
    void allocate(std::size_t count) {
        if (count == size_) {
            return;
        }

        release();

        if (count == 0) {
            return;
        }

        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&devicePtr_),
            count * sizeof(T)
        ));

        size_ = count;
    }

    /**
     * @brief Libera la memoria device.
     */
    void release() {
        if (devicePtr_ != nullptr) {
            CUDA_CHECK(cudaFree(devicePtr_));
            devicePtr_ = nullptr;
        }

        size_ = 0;
    }

    /**
     * @brief Copia datos desde memoria host hacia memoria device.
     */
    void copyFromHost(const T* hostData, std::size_t count) {
        validateHostCopy(hostData, count, "CudaBuffer::copyFromHost");

        if (count == 0) {
            return;
        }

        CUDA_CHECK(cudaMemcpy(
            devicePtr_,
            hostData,
            count * sizeof(T),
            cudaMemcpyHostToDevice
        ));
    }

    /**
     * @brief Copia datos desde memoria device hacia memoria host.
     */
    void copyToHost(T* hostData, std::size_t count) const {
        validateHostCopy(hostData, count, "CudaBuffer::copyToHost");

        if (count == 0) {
            return;
        }

        CUDA_CHECK(cudaMemcpy(
            hostData,
            devicePtr_,
            count * sizeof(T),
            cudaMemcpyDeviceToHost
        ));
    }

    T* data() noexcept {
        return devicePtr_;
    }

    const T* data() const noexcept {
        return devicePtr_;
    }

    std::size_t size() const noexcept {
        return size_;
    }

    std::size_t bytes() const noexcept {
        return size_ * sizeof(T);
    }

    bool empty() const noexcept {
        return size_ == 0;
    }

    ~CudaBuffer() noexcept {
    release();
    }
};