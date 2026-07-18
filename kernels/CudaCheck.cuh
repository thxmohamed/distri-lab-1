#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

/**
 * @brief Comprueba el resultado de una operación ejecutada mediante la API CUDA.
 *
 * Recibe el código de error retornado por una función CUDA y verifica si
 * la operación fue ejecutada correctamente. En caso de error, muestra
 * información de depuración incluyendo archivo, línea, expresión ejecutada
 * y descripción del error, finalizando posteriormente la ejecución del programa.
 *
 * @param result Código de retorno entregado por una función de CUDA.
 * @param expression Expresión CUDA evaluada que generó el resultado.
 * @param file Archivo fuente donde fue realizada la llamada CUDA.
 * @param line Línea del archivo donde se ejecutó la llamada CUDA.
 */
inline void cudaCheck(
    cudaError_t result,
    const char* expression,
    const char* file,
    int line
) {
    if (result != cudaSuccess) {
        std::cerr
            << "CUDA error en " << file << ":" << line << '\n'
            << "Expresion: " << expression << '\n'
            << "Detalle: " << cudaGetErrorString(result)
            << std::endl;

        std::exit(EXIT_FAILURE);
    }
}

/**
 * @brief Ejecuta una llamada CUDA y verifica automáticamente su resultado.
 *
 * Envuelve una expresión CUDA dentro de la función cudaCheck, capturando
 * automáticamente la expresión ejecutada, el archivo fuente y la línea
 * donde ocurrió la llamada.
 *
 * Permite detectar errores de ejecución CUDA inmediatamente después del
 * lanzamiento de una operación en GPU.
 *
 * @param expression Llamada o expresión perteneciente a la API CUDA.
 *
 * @note Ejemplo de uso:
 * @code
 * CUDA_CHECK(cudaMalloc(&ptr, size));
 * @endcode
 */
#define CUDA_CHECK(expression) \
    cudaCheck((expression), #expression, __FILE__, __LINE__)
