#include "accelerations.cuh"
#include "CudaCheck.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <stdexcept>

/**
 * @brief Kernel CUDA básico para calcular aceleraciones gravitatorias.
 *
 * Asigna un hilo CUDA a cada cuerpo del sistema. Cada hilo calcula la
 * aceleración del cuerpo i recorriendo secuencialmente todos los cuerpos j.
 *
 * La contribución gravitatoria de cada cuerpo se acumula en registros locales
 * y finalmente se escribe en memoria global en las posiciones correspondientes.
 *
 * @param d_mass Masas de los cuerpos almacenadas en memoria device.
 * @param d_x Coordenadas X de los cuerpos almacenadas en memoria device.
 * @param d_y Coordenadas Y de los cuerpos almacenadas en memoria device.
 * @param d_ax Vector de salida con aceleraciones X calculadas.
 * @param d_ay Vector de salida con aceleraciones Y calculadas.
 * @param n Cantidad de cuerpos del sistema.
 * @param gravitationalConstant Constante gravitatoria utilizada.
 * @param epsilonSquared Termino de suavizado para evitar singularidades
 *                       cuando dos cuerpos están muy cercanos.
 */
__global__ void computeAccelerationsKernel(
    const double* d_mass,
    const double* d_x,
    const double* d_y,
    double* d_ax,
    double* d_ay,
    int n,
    double gravitationalConstant,
    double epsilonSquared
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Protección para N no múltiplo del tamaño de bloque.
    if (i >= n) {
        return;
    }

    const double xi = d_x[i];
    const double yi = d_y[i];

    double accelerationX = 0.0;
    double accelerationY = 0.0;

    // El bucle interno sobre j se mantiene serial dentro del hilo.
    for (int j = 0; j < n; ++j) {
        if (i == j) {
            continue;
        }

        const double dx = d_x[j] - xi;
        const double dy = d_y[j] - yi;

        const double distanceSquared =
            dx * dx + dy * dy + epsilonSquared;

        const double distanceCubed =
            distanceSquared * sqrt(distanceSquared);

        const double factor =
            gravitationalConstant * d_mass[j] / distanceCubed;

        accelerationX += factor * dx;
        accelerationY += factor * dy;
    }

    d_ax[i] = accelerationX;
    d_ay[i] = accelerationY;
}

/**
 * @brief Kernel CUDA optimizado para calcular aceleraciones usando memoria compartida.
 *
 * Divide los cuerpos en tiles que son cargados desde memoria global hacia
 * memoria compartida. Los hilos de cada bloque reutilizan estos datos para
 * reducir accesos repetidos a memoria global.
 *
 * Cada bloque procesa una porción del conjunto de cuerpos y sincroniza sus
 * hilos mediante __syncthreads() antes y después del uso de los datos compartidos.
 *
 * @param d_mass Masas de los cuerpos en memoria device.
 * @param d_x Posiciones X de los cuerpos en memoria device.
 * @param d_y Posiciones Y de los cuerpos en memoria device.
 * @param d_ax Salida de aceleraciones en X.
 * @param d_ay Salida de aceleraciones en Y.
 * @param n Número total de cuerpos.
 * @param gravitationalConstant Constante gravitatoria.
 * @param epsilonSquared Suavizado aplicado al cálculo de distancia.
 */
__global__ void computeAccelerationsKernelShared(
    const double* d_mass,
    const double* d_x,
    const double* d_y,
    double* d_ax,
    double* d_ay,
    int n,
    double gravitationalConstant,
    double epsilonSquared
) {
    extern __shared__ double sharedMemory[];

    double* sharedMass = sharedMemory;
    double* sharedX = sharedMemory + blockDim.x;
    double* sharedY = sharedMemory + 2 * blockDim.x;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool activeBody = i < n;

    /*
     * Los hilos fuera de rango no pueden abandonar el kernel aqui,
     * porque todos deben alcanzar los mismos __syncthreads().
     */
    const double xi = activeBody ? d_x[i] : 0.0;
    const double yi = activeBody ? d_y[i] : 0.0;

    double accelerationX = 0.0;
    double accelerationY = 0.0;

    const int numberOfTiles =
        (n + static_cast<int>(blockDim.x) - 1) /
        static_cast<int>(blockDim.x);

    for (int tile = 0; tile < numberOfTiles; ++tile) {
        const int jGlobal =
            tile * static_cast<int>(blockDim.x) +
            static_cast<int>(threadIdx.x);

        if (jGlobal < n) {
            sharedMass[threadIdx.x] = d_mass[jGlobal];
            sharedX[threadIdx.x] = d_x[jGlobal];
            sharedY[threadIdx.x] = d_y[jGlobal];
        } else {
            sharedMass[threadIdx.x] = 0.0;
            sharedX[threadIdx.x] = 0.0;
            sharedY[threadIdx.x] = 0.0;
        }

        __syncthreads();

        if (activeBody) {
            const int remainingBodies =
                n - tile * static_cast<int>(blockDim.x);

            const int bodiesInTile =
                remainingBodies < static_cast<int>(blockDim.x)
                    ? remainingBodies
                    : static_cast<int>(blockDim.x);

            for (int jLocal = 0; jLocal < bodiesInTile; ++jLocal) {
                const int j =
                    tile * static_cast<int>(blockDim.x) + jLocal;

                if (i == j) {
                    continue;
                }

                const double dx = sharedX[jLocal] - xi;
                const double dy = sharedY[jLocal] - yi;

                const double distanceSquared =
                    dx * dx + dy * dy + epsilonSquared;

                const double distanceCubed =
                    distanceSquared * sqrt(distanceSquared);

                const double factor =
                    gravitationalConstant *
                    sharedMass[jLocal] /
                    distanceCubed;

                accelerationX += factor * dx;
                accelerationY += factor * dy;
            }
        }

        __syncthreads();
    }

    if (activeBody) {
        d_ax[i] = accelerationX;
        d_ay[i] = accelerationY;
    }
}

/**
 * @brief Ejecuta una variante del cálculo de aceleraciones en GPU.
 *
 * Selecciona entre el kernel básico y el kernel optimizado con memoria
 * compartida según la variante recibida. También valida los parámetros
 * de entrada antes de realizar el lanzamiento CUDA.
 *
 * La memoria device debe haber sido reservada previamente por el llamador.
 *
 * @param d_mass Puntero device con masas.
 * @param d_x Puntero device con posiciones X.
 * @param d_y Puntero device con posiciones Y.
 * @param d_ax Puntero device donde se almacenan aceleraciones X.
 * @param d_ay Puntero device donde se almacenan aceleraciones Y.
 * @param n Número de cuerpos.
 * @param gravitationalConstant Constante gravitatoria.
 * @param epsilon Parámetro de suavizado.
 * @param variant Variante de kernel a ejecutar (Basic o Shared).
 * @param blockSize Número de hilos por bloque CUDA.
 *
 * @throws std::invalid_argument Si algún parámetro de entrada es inválido.
 */
void launchComputeAccelerations(
    const double* d_mass,
    const double* d_x,
    const double* d_y,
    double* d_ax,
    double* d_ay,
    int n,
    double gravitationalConstant,
    double epsilon,
    AccelerationKernelVariant variant,
    int blockSize
) {
    if (n < 0) {
        throw std::invalid_argument(
            "launchComputeAccelerations: n no puede ser negativo.");
    }

    if (n == 0) {
        return;
    }

    if (blockSize <= 0) {
        throw std::invalid_argument(
            "launchComputeAccelerations: blockSize debe ser positivo.");
    }

    if (epsilon <= 0.0) {
        throw std::invalid_argument(
            "launchComputeAccelerations: epsilon debe ser positivo.");
    }

    if (d_mass == nullptr ||
        d_x == nullptr ||
        d_y == nullptr ||
        d_ax == nullptr ||
        d_ay == nullptr) {
        throw std::invalid_argument(
            "launchComputeAccelerations: los punteros device no pueden ser nulos.");
    }

    const int gridSize =
        (n + blockSize - 1) / blockSize;

    const double epsilonSquared =
        epsilon * epsilon;

    switch (variant) {
        case AccelerationKernelVariant::Basic:
            computeAccelerationsKernel<<<gridSize, blockSize>>>(
                d_mass,
                d_x,
                d_y,
                d_ax,
                d_ay,
                n,
                gravitationalConstant,
                epsilonSquared
            );
            break;

        case AccelerationKernelVariant::Shared: {
            const std::size_t sharedMemoryBytes =
                3ULL *
                static_cast<std::size_t>(blockSize) *
                sizeof(double);

            computeAccelerationsKernelShared
                <<<gridSize, blockSize, sharedMemoryBytes>>>(
                    d_mass,
                    d_x,
                    d_y,
                    d_ax,
                    d_ay,
                    n,
                    gravitationalConstant,
                    epsilonSquared
                );
            break;
        }

        default:
            throw std::invalid_argument(
                "launchComputeAccelerations: variante de kernel invalida.");
    }

    CUDA_CHECK(cudaGetLastError());
}
