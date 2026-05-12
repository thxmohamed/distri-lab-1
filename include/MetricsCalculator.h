#pragma once

#include "NBodySystem.h"

/*
 * MetricsCalculator.h
 * -------------------
 * Define la interfaz del módulo encargado del cálculo de métricas físicas
 * globales del sistema N-cuerpos.
 *
 * Esta clase permite analizar el comportamiento físico y numérico de la
 * simulación mediante métricas como energía, momentum, centro de masa,
 * radio RMS y distancia mínima entre partículas.
 *
 * Todas las funciones son estáticas, ya que las métricas se calculan a partir
 * del estado actual del sistema sin necesidad de almacenar información interna.
 */

class MetricsCalculator {
public:

    // Calcula la energía cinética total del sistema.
    static double kineticEnergy(const NBodySystem& system);

    // Calcula la energía potencial gravitacional total del sistema.
    static double potentialEnergy(const NBodySystem& system);

    // Calcula la energía total E = K + U.
    static double totalEnergy(const NBodySystem& system);

    // Calcula la magnitud del momentum lineal total.
    static double momentum(const NBodySystem& system);

    // Calcula la coordenada X del centro de masa.
    static double centerOfMassX(const NBodySystem& system);

    // Calcula la coordenada Y del centro de masa.
    static double centerOfMassY(const NBodySystem& system);

    // Calcula el radio RMS respecto al centro de masa.
    static double rmsRadius(const NBodySystem& system);

    // Calcula la distancia mínima entre pares de partículas.
    static double minPairDistance(const NBodySystem& system);
};