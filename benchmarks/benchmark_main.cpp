#include "Benchmark.h"
#include "NBodySystem.h"
#include "NBodySimulator.h"
#include <omp.h>
#include <fstream>
#include <iostream>

/**
 * Función: main
 * Entrada: no recibe argumentos.
 * Salida: retorna 0 si el benchmark termina correctamente.
 * Descripción:
 * Ejecuta benchmarks de escalabilidad para la simulación N-cuerpos,
 * variando el número de hilos OpenMP y registrando los resultados
 * en el archivo benchmark_results_lab1.dat.
 */
int main() {
    // Cantidad de cuerpos usados en la simulación.
    int N = 1000;

    // Cantidad de pasos temporales ejecutados por simulación.
    int steps = 100;

    // Cantidad de repeticiones por configuración para calcular estadística.
    int reps = 10;

    // Archivo de salida donde se guardan los resultados del benchmark.
    std::ofstream out("benchmark_results_lab1.dat");

    // Cabecera del archivo .dat:
    // threads  -> número de hilos usados
    // mean     -> tiempo promedio de ejecución
    // stddev   -> desviación estándar de los tiempos
    // speedup  -> aceleración respecto a 1 hilo
    // efficiency -> eficiencia paralela
    out << "# threads mean stddev speedup efficiency\n";

    // Tiempo promedio de referencia con 1 hilo.
    // Se usa como base para calcular el speedup.
    double T1 = 0.0;

    /*
     * Se evalúa la simulación usando distintas cantidades de hilos.
     * Para cada configuración:
     *  1. Se fija el número de hilos OpenMP.
     *  2. Se ejecuta el benchmark con varias repeticiones.
     *  3. Se calcula speedup y eficiencia.
     *  4. Se guarda el resultado en benchmark_results_lab1.dat.
     */
    for (int threads : {1, 2, 4, 8}) {
        // Configura la cantidad de hilos que usará OpenMP.
        omp_set_num_threads(threads);

        /*
         * Ejecuta la medición de la simulación.
         * Internamente, Benchmark::measureSimulation crea el sistema,
         * reinicia las condiciones iniciales y repite la ejecución 'reps' veces.
         */
        auto result = Benchmark::measureSimulation(N, steps, reps);

        /*
         * El caso con 1 hilo se toma como tiempo base T1.
         * Este valor permite comparar las ejecuciones paralelas contra
         * una referencia equivalente usando el mismo problema.
         */
        if (threads == 1) {
            T1 = result.mean;
        }

        // Speedup: cuánto más rápido es el caso con p hilos respecto a 1 hilo.
        double Sp = Benchmark::speedup(T1, result.mean);

        // Eficiencia: proporción de aprovechamiento del paralelismo disponible.
        double Ep = Benchmark::efficiency(Sp, threads);

        // Guarda los resultados en formato de columnas para facilitar graficación.
        out << threads << " "
            << result.mean << " "
            << result.stddev << " "
            << Sp << " "
            << Ep << "\n";

        // Muestra los resultados por consola 
        std::cout << "Threads: " << threads
                  << " | Mean: " << result.mean
                  << " | Stddev: " << result.stddev
                  << " | Speedup: " << Sp
                  << " | Efficiency: " << Ep
                  << std::endl;
    }

    out.close();

    return 0;
}