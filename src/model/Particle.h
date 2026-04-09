#pragma once

#include <iosfwd>   // forward declaration de ostream

/**
 * Particle
 * --------
 * Representa un cuerpo puntual en 2D con masa, posición,
 * velocidad y aceleración. Es la unidad atómica del simulador.
 *
 * Responsabilidades:
 *  - Almacenar y exponer el estado cinemático del cuerpo.
 *  - Aplicar kick (v += a*dt) y drift (r += v*dt) por separado,
 *    de modo que el integrador pueda respetar el orden correcto.
 *  - NO calcula fuerzas: eso es responsabilidad de NBodySystem.
 */
class Particle {
private:
    double mass_;           // masa (debe ser > 0)
    double x_,   y_;       // posición
    double vx_,  vy_;      // velocidad
    double ax_,  ay_;      // aceleración acumulada en el paso actual

public:
    // ----------------------------------------------------------------
    // Constructor / destructor
    // ----------------------------------------------------------------

    /**
     * @param m   Masa del cuerpo (debe ser > 0, lanza std::invalid_argument si no).
     * @param x0  Posición inicial x.
     * @param y0  Posición inicial y.
     * @param vx0 Velocidad inicial en x (por defecto 0).
     * @param vy0 Velocidad inicial en y (por defecto 0).
     */
    Particle(double m, double x0, double y0,
             double vx0 = 0.0, double vy0 = 0.0);

    // ----------------------------------------------------------------
    // Integración (llamadas desde Integrator / NBodySimulator)
    // ----------------------------------------------------------------

    /** v += a * dt  (fase "kick") */
    void kick(double dt);

    /** r += v * dt  (fase "drift") */
    void drift(double dt);

    // ----------------------------------------------------------------
    // Manejo de aceleración (llamadas desde NBodySystem)
    // ----------------------------------------------------------------

    /** Pone ax = ay = 0. Debe llamarse antes de acumular contribuciones. */
    void zeroAcceleration();

    /** Sobreescribe la aceleración completa. */
    void setAcceleration(double ax, double ay);

    /**
     * Suma un incremento a la aceleración acumulada.
     * ATENCIÓN: si se llama desde múltiples hilos sobre la misma partícula
     * se requiere sincronización (atomic / critical) en NBodySystem.
     */
    void addAcceleration(double dax, double day);

    // ----------------------------------------------------------------
    // Getters
    // ----------------------------------------------------------------
    double getMass() const { return mass_; }
    double getX()    const { return x_;    }
    double getY()    const { return y_;    }
    double getVx()   const { return vx_;   }
    double getVy()   const { return vy_;   }
    double getAx()   const { return ax_;   }
    double getAy()   const { return ay_;   }

    // ----------------------------------------------------------------
    // Setters de posición y velocidad (para inicialización)
    // ----------------------------------------------------------------
    void setPosition(double x, double y)   { x_  = x;  y_  = y;  }
    void setVelocity(double vx, double vy) { vx_ = vx; vy_ = vy; }

    // ----------------------------------------------------------------
    // I/O
    // ----------------------------------------------------------------

    /** Imprime estado por consola (debug). */
    void print() const;

    /**
     * Escribe una línea con: x y vx vy ax ay
     * Formato compatible con los archivos .dat del proyecto.
     */
    void writeToStream(std::ostream& out) const;
};