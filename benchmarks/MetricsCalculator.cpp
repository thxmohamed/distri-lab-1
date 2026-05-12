#include "MetricsCalculator.h"
#include <cmath>
#include <omp.h>


/**
 * Implementa métricas físicas globales para el sistema N-cuerpos.
 *
 * Esta clase permite calcular energía cinética, energía potencial,
 * energía total, momento lineal, centro de masa, radio RMS y distancia
 * mínima entre pares de partículas.
 *
 * Estas métricas sirven para analizar la estabilidad física del sistema,
 * la conservación aproximada de energía y momento, y la dispersión espacial
 * de las partículas durante la simulación.
 */



 /**
 * Función: MetricsCalculator::kineticEnergy
 * Entrada:
 *  - system: sistema N-cuerpos desde el cual se leen masas y velocidades.
 * Salida:
 *  - Energía cinética total K del sistema.
 * Descripción:
 *  Calcula K = 1/2 * Σ m_i * |v_i|² usando reducción paralela.
 */
double MetricsCalculator::kineticEnergy(const NBodySystem& system) {
    const auto& bodies = system.getBodies();
    double K = 0.0;

    /*
     * Cada hilo acumula una parte de la energía cinética.
     * La cláusula reduction combina los acumulados parciales de forma segura.
     */
    #pragma omp parallel for reduction(+:K)
    for (int i = 0; i < static_cast<int>(bodies.size()); i++) {
        double vx = bodies[i].getVx();
        double vy = bodies[i].getVy();
        double m  = bodies[i].getMass();

        K += 0.5 * m * (vx * vx + vy * vy);
    }

    return K;
}

/**
 * Función: MetricsCalculator::potentialEnergy
 * Entrada:
 *  - system: sistema N-cuerpos desde el cual se leen masas y posiciones.
 * Salida:
 *  - Energía potencial gravitacional total U.
 * Descripción:
 *  Calcula la energía potencial sumando pares únicos i < j,
 *  aplicando suavizado epsilon para evitar singularidades.
 */
double MetricsCalculator::potentialEnergy(const NBodySystem& system) {
    const auto& bodies = system.getBodies();
    double G = system.getG();
    double eps2 = system.getEpsilon() * system.getEpsilon();

    double U = 0.0;

     /*
     * Se recorren pares únicos i<j para no contar dos veces la misma interacción.
     * La reducción evita condiciones de carrera sobre la variable compartida U.
     */
    #pragma omp parallel for reduction(+:U)
    for (int i = 0; i < static_cast<int>(bodies.size()); i++) {
        for (int j = i + 1; j < static_cast<int>(bodies.size()); j++) {
            double dx = bodies[j].getX() - bodies[i].getX();
            double dy = bodies[j].getY() - bodies[i].getY();
            double dist = std::sqrt(dx * dx + dy * dy + eps2);

            U += -G * bodies[i].getMass() * bodies[j].getMass() / dist;
        }
    }

    return U;
}

/**
 * Función: MetricsCalculator::totalEnergy
 * Entrada:
 *  - system: sistema N-cuerpos.
 * Salida:
 *  - Energía total E = K + U.
 * Descripción:
 *  Calcula la energía total del sistema sumando energía cinética
 *  y energía potencial.
 */
double MetricsCalculator::totalEnergy(const NBodySystem& system) {
    return kineticEnergy(system) + potentialEnergy(system);
}


/**
 * Función: MetricsCalculator::momentum
 * Entrada:
 *  - system: sistema N-cuerpos desde el cual se leen masas y velocidades.
 * Salida:
 *  - Magnitud del momento lineal total.
 * Descripción:
 *  Calcula el momento lineal total P = Σ m_i * v_i y retorna su magnitud.
 */
double MetricsCalculator::momentum(const NBodySystem& system) {
    const auto& bodies = system.getBodies();
    double px = 0.0;
    double py = 0.0;

    /*
     * Se reducen simultáneamente las componentes px y py.
     * Cada hilo acumula contribuciones de distintas partículas.
     */
    #pragma omp parallel for reduction(+:px,py)
    for (int i = 0; i < static_cast<int>(bodies.size()); i++) {
        px += bodies[i].getMass() * bodies[i].getVx();
        py += bodies[i].getMass() * bodies[i].getVy();
    }

    return std::sqrt(px * px + py * py);
}

/**
 * Función: MetricsCalculator::centerOfMassX
 * Entrada:
 *  - system: sistema N-cuerpos desde el cual se leen masas y posiciones.
 * Salida:
 *  - Coordenada X del centro de masa.
 * Descripción:
 *  Calcula Rcm_x = Σ(m_i * x_i) / Σm_i.
 */
double MetricsCalculator::centerOfMassX(const NBodySystem& system) {
    const auto& bodies = system.getBodies();

    double weighted_x = 0.0;
    double total_mass = 0.0;

    /*
     * Se acumulan en paralelo la masa total y la suma ponderada en X.
     * La reducción mantiene ambas acumulaciones libres de carreras.
     */
    #pragma omp parallel for reduction(+:weighted_x,total_mass)
    for (int i = 0; i < static_cast<int>(bodies.size()); i++) {
        double m = bodies[i].getMass();
        weighted_x += m * bodies[i].getX();
        total_mass += m;
    }

    if (total_mass == 0.0) {
        return 0.0;
    }

    return weighted_x / total_mass;
}


/**
 * Función: MetricsCalculator::centerOfMassY
 * Entrada:
 *  - system: sistema N-cuerpos desde el cual se leen masas y posiciones.
 * Salida:
 *  - Coordenada Y del centro de masa.
 * Descripción:
 *  Calcula Rcm_y = Σ(m_i * y_i) / Σm_i.
 */
double MetricsCalculator::centerOfMassY(const NBodySystem& system) {
    const auto& bodies = system.getBodies();

    double weighted_y = 0.0;
    double total_mass = 0.0;

    /*
     * Misma lógica que centerOfMassX, pero aplicada a la coordenada Y.
     */
    #pragma omp parallel for reduction(+:weighted_y,total_mass)
    for (int i = 0; i < static_cast<int>(bodies.size()); i++) {
        double m = bodies[i].getMass();
        weighted_y += m * bodies[i].getY();
        total_mass += m;
    }

    if (total_mass == 0.0) {
        return 0.0;
    }

    return weighted_y / total_mass;
}

/**
 * Función: MetricsCalculator::rmsRadius
 * Entrada:
 *  - system: sistema N-cuerpos desde el cual se leen masas y posiciones.
 * Salida:
 *  - Radio RMS respecto al centro de masa.
 * Descripción:
 *  Calcula una medida global de dispersión espacial del sistema respecto
 *  al centro de masa.
 */
double MetricsCalculator::rmsRadius(const NBodySystem& system) {
    const auto& bodies = system.getBodies();

    double cmx = centerOfMassX(system);
    double cmy = centerOfMassY(system);

    double sum = 0.0;
    double total_mass = 0.0;

    /*
     * Se calcula la distancia cuadrática de cada partícula al centro de masa,
     * ponderada por su masa. Luego se normaliza por la masa total.
     */
    #pragma omp parallel for reduction(+:sum,total_mass)
    for (int i = 0; i < static_cast<int>(bodies.size()); i++) {
        double m = bodies[i].getMass();

        double dx = bodies[i].getX() - cmx;
        double dy = bodies[i].getY() - cmy;

        sum += m * (dx * dx + dy * dy);
        total_mass += m;
    }

    if (total_mass == 0.0) {
        return 0.0;
    }

    return std::sqrt(sum / total_mass);
}


/**
 * Función: MetricsCalculator::minPairDistance
 * Entrada:
 *  - system: sistema N-cuerpos desde el cual se leen posiciones.
 * Salida:
 *  - Distancia mínima entre dos partículas distintas.
 * Descripción:
 *  Recorre todos los pares i<j y retorna la menor distancia encontrada.
 *  Esta métrica permite detectar encuentros cercanos entre cuerpos.
 */
double MetricsCalculator::minPairDistance(const NBodySystem& system) {
    const auto& bodies = system.getBodies();
    int N = static_cast<int>(bodies.size());

    if (N < 2) {
        return 0.0;
    }

    double min_dist = 1e300;

    /*
     * Cada hilo calcula una distancia mínima local para evitar actualizar
     * una variable compartida dentro del doble bucle. Al final, cada mínimo
     * local se compara con el mínimo global usando critical.
     */
    #pragma omp parallel
    {
        double local_min = 1e300;

        #pragma omp for nowait
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                double dx = bodies[j].getX() - bodies[i].getX();
                double dy = bodies[j].getY() - bodies[i].getY();

                double dist = std::sqrt(dx * dx + dy * dy);

                if (dist < local_min) {
                    local_min = dist;
                }
            }
        }

        /*
         * Sección crítica breve: solo se protege la actualización final
         * del mínimo global, reduciendo la contención entre hilos.
         */
        #pragma omp critical
        {
            if (local_min < min_dist) {
                min_dist = local_min;
            }
        }
    }

    return min_dist;
}