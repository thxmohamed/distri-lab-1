#pragma once

#include "Particle.h"

#include <vector>
#include <string>
#include <iosfwd>
#include <memory>

class NBodyDeviceState;
class NBodySimulator;

/**
 * =================================================================
 * NBodySystem
 * =================================================================
 *
 * Clase central del simulador gravitacional N-cuerpos en 2D.
 *
 * Actúa como contenedor principal de todas las partículas del sistema
 * y encapsula los parámetros físicos globales: la constante gravitacional
 * G y el parámetro de suavizado ε (epsilon). Estos parámetros son
 * compartidos por todas las interacciones del sistema.
 *
 * Su responsabilidad principal es el cálculo de aceleraciones todo-pares
 * O(N²), implementando la ley de gravitación newtoniana con suavizado
 * de Plummer para evitar singularidades numéricas cuando dos partículas
 * se acercan demasiado.
 *
 * Para permitir el análisis de rendimiento con OpenMP, expone múltiples
 * variantes del cálculo de aceleraciones: serial, paralela simple, con
 * distintos schedules (static, dynamic, guided) y con collapse(2).
 * Todas las variantes producen el mismo resultado físico y pueden
 * compararse en benchmarks sin modificar el resto del simulador.
 *
 * Uso típico:
 *   NBodySystem sys(1.0, 0.05);   // G=1, epsilon=0.05
 *   sys.initDisk(1000, 1.0, 42);  // condición inicial
 *   sys.computeAccelerations(0, 16); // schedule static, chunk=16
 * =================================================================
 */
class NBodySystem {
private:
    std::vector<Particle> bodies_; // vector que almacena todas las partículas del sistema
    double G_;                     // constante gravitacional (G = 1.0 en unidades adimensionales)
    double epsilon_;               // parámetro de suavizado de Plummer (ε > 0), evita singularidades

    /**
     * Estado SoA persistente en memoria GPU.
     *
     * Se crea de forma diferida durante la primera operación CUDA y se
     * conserva entre llamadas para evitar reservar y liberar memoria
     * device en cada paso temporal.
     *
     * Aunque se utiliza shared_ptr para permitir la declaración adelantada
     * del tipo CUDA, las copias de NBodySystem no compartirán este estado.
     */
    std::shared_ptr<NBodyDeviceState> device_state_;

    /**
     * Indica que el estado device requiere una carga completa.
     *
     * Se activa al agregar, eliminar o reinicializar partículas, ya que
     * en esos casos pueden cambiar la cantidad de cuerpos y sus masas.
     */
    bool device_state_needs_full_upload_ = true;

    /**
     * Prepara y retorna el estado CUDA asociado al sistema.
     *
     * Primera utilización o sistema modificado:
     *  - reserva/redimensiona los buffers;
     *  - copia masas, posiciones y velocidades.
     *
     * Utilizaciones posteriores:
     *  - actualiza posiciones;
     *  - actualiza velocidades solo cuando include_velocities es true.
     *
     * @param include_velocities Indica si también deben actualizarse vx y vy.
     */
    NBodyDeviceState& prepareDeviceState(bool include_velocities);

    /**
     * Marca el estado CUDA como desactualizado.
     *
     * No libera inmediatamente la memoria: la siguiente operación GPU
     * decidirá si debe reutilizar o redimensionar los buffers.
     */
    void invalidateDeviceState() noexcept {
        device_state_needs_full_upload_ = true;
    }

    /**
     * Permite que NBodySimulator reutilice el mismo estado CUDA para
     * calcular energía, sin crear otro NBodyDeviceState temporal.
     */
    friend class NBodySimulator;

public:
    // ----------------------------------------------------------------
    // Constructor
    // ----------------------------------------------------------------

    /**
     * @param G       Constante gravitacional (p.ej. 1.0).
     * @param epsilon Parámetro de suavizado (p.ej. 0.05). Debe ser > 0.
     */
    NBodySystem(double G, double epsilon);

    /**
     * Copia el estado físico almacenado en host.
     *
     * El estado CUDA no se comparte con la copia. El nuevo sistema creará
     * sus propios buffers cuando ejecute por primera vez una operación GPU.
     */
    NBodySystem(const NBodySystem& other);

    /**
     * Asigna el estado físico almacenado en host.
     *
     * Cualquier estado CUDA anterior se descarta y deberá inicializarse
     * nuevamente en la siguiente operación GPU.
     */
    NBodySystem& operator=(const NBodySystem& other);

    /**
     * Transfiere tanto el estado host como la propiedad del estado CUDA.
     */
    NBodySystem(NBodySystem&& other) noexcept = default;
    NBodySystem& operator=(NBodySystem&& other) noexcept = default;

    // ----------------------------------------------------------------
    // Gestión de partículas
    // ----------------------------------------------------------------

    /** Agrega una partícula al sistema. */
    void addParticle(const Particle& p);

    /** Elimina todas las partículas del sistema. */
    void clear();

    /** Retorna el número total de partículas almacenadas. */
    int  getCount() const;

    /** Retorna referencia constante al vector de partículas (solo lectura). */
    const std::vector<Particle>& getBodies() const;

    /** Retorna referencia mutable al vector de partículas (para uso del Integrator). */
    std::vector<Particle>&       getBodies();

    /** Retorna la constante gravitacional G del sistema. */
    double getG()       const { return G_;       }

    /** Retorna el parámetro de suavizado epsilon del sistema. */
    double getEpsilon() const { return epsilon_; }

    // ----------------------------------------------------------------
    // Preproceso de aceleraciones
    // ----------------------------------------------------------------

    /** Pone a cero ax y ay de todas las partículas antes de cada paso de cálculo. */
    void zeroAccelerations();

    // ----------------------------------------------------------------
    // Cálculo de aceleraciones (todo-pares, ecuación 1 del enunciado)
    // ----------------------------------------------------------------

    /** Versión serial de referencia; delega en computeAccelerationsSerial(). */
    void computeAccelerations();

    /** Implementación secuencial O(N²); útil como referencia y para benchmarks de un hilo. */
    void computeAccelerationsSerial();

    /**
     * Versión paralela con schedule configurable y chunk_size = 1 por defecto.
     * @param schedule_type  0 = static, 1 = dynamic, 2 = guided
     */
    void computeAccelerations(int schedule_type);

    /**
     * Versión paralela con schedule y chunk explícito; es la implementación base de las sobrecargas.
     * @param schedule_type  0 = static, 1 = dynamic, 2 = guided
     * @param chunk_size     Tamaño de chunk para el schedule. Debe ser > 0.
     */
    void computeAccelerations(int schedule_type, int chunk_size);

    /**
     * Variante con collapse(2) sobre el doble bucle (i, j); usa arreglos auxiliares
     * y atomic para evitar condiciones de carrera al acumular sobre el mismo índice i.
     */
    void computeAccelerationsCollapse();

    /** Versión paralela básica sin schedule explícito; útil como primer paso de validación paralela. */
    void computeAccelerationsParallelSimple();

    /**
     * Dispatcher que selecciona la variante de cálculo según el modo indicado.
     * 0 = serial, 1 = paralelo simple, 2 = static, 3 = dynamic, 4 = guided, 5 = collapse.
     */
    void computeAccelerationsMode(int mode);

    // ----------------------------------------------------------------
    // Cálculo de aceleraciones GPU / CUDA
    // ----------------------------------------------------------------

    /**
     * Calcula aceleraciones en GPU usando kernel básico y block size 256.
     */
    void computeAccelerationsGpu();

    /**
     * Calcula aceleraciones en GPU usando variante configurable.
     * @param variant 0 = básico, 1 = shared memory
     */
    void computeAccelerationsGpu(int variant);

    /**
     * Calcula aceleraciones en GPU usando variante y block size configurables.
     * @param variant    0 = básico, 1 = shared memory
     * @param block_size hilos CUDA por bloque
     */
    void computeAccelerationsGpu(int variant, int block_size);

    // ----------------------------------------------------------------
    // Inicialización de condiciones iniciales
    // ----------------------------------------------------------------

    /**
     * Inicializa dos masas dominantes simétricas con N-2 cuerpos ligeros perturbados.
     * @param N      Número total de cuerpos (mínimo 2).
     * @param seed   Semilla para reproducibilidad.
     */
    void initBinary(int N, unsigned int seed = 42);

    /**
     * Inicializa una masa central y N-1 partículas en disco con velocidades tangenciales circulares.
     * @param N      Número total de cuerpos (mínimo 1).
     * @param radius Radio máximo del disco. Debe ser > 0.
     * @param seed   Semilla para reproducibilidad.
     */
    void initDisk(int N, double radius = 1.0, unsigned int seed = 42);

    /**
     * Genera partículas con distribución tipo Plummer proyectada a 2D por muestreo inverso.
     * @param N      Número total de cuerpos (mínimo 1).
     * @param a      Escala de Plummer (radio característico). Debe ser > 0.
     * @param seed   Semilla para reproducibilidad.
     */
    void initPlummer(int N, double a = 0.5, unsigned int seed = 42);
};