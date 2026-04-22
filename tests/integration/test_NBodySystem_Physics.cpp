#include <gtest/gtest.h>
#include "NBodySystem.h"
#include "Particle.h"
#include <cmath>

// ============================================================
// VALIDACIÓN ANALÍTICA (FÍSICA REAL)
// ============================================================
// Este test verifica que la implementación de gravedad cumple:
//
// ax = G * m * dx / (r^2 + eps^2)^(3/2)
//
// OBJETIVOS:
// -Validar fórmula física exacta
// -Detectar errores en la ecuación gravitacional
// -Evitar hardcodeo de resultados
// -Usar geometría dinámica del sistema
// ============================================================

// ============================================================
// TEST: Analytical Validation (2 cuerpos en 2D)
// ============================================================
TEST(NBodySystem_Physics, AnalyticalValidation) {

    double G = 1.0;
    double eps = 0.1;

    NBodySystem sys(G, eps);

    // ---------------- configuración dinámica ----------------
    double x1 = 0.0, y1 = 0.0;
    double x2 = 1.0, y2 = 0.0;
    double m1 = 1.0, m2 = 1.0;

    sys.addParticle(Particle(m1, x1, y1, 0, 0));
    sys.addParticle(Particle(m2, x2, y2, 0, 0));

    // ejecución del sistema
    sys.computeAccelerations();
    auto bodies = sys.getBodies();

    ASSERT_EQ(sys.getCount(), 2);

    // ========================================================
    // CÁLCULO ANALÍTICO (referencia física)
    // ========================================================
    double dx = x2 - x1;
    double dy = y2 - y1;

    double r2 = dx * dx + dy * dy;
    double dist2 = r2 + eps * eps;
    double dist3 = dist2 * std::sqrt(dist2);

    double expected_ax = G * m2 * dx / dist3;
    double expected_ay = G * m2 * dy / dist3;

    double ax = bodies[0].getAx();
    double ay = bodies[0].getAy();

    // ========================================================
    // VALIDACIÓN
    // ========================================================
    EXPECT_NEAR(ax, expected_ax, 1e-6);
    EXPECT_NEAR(ay, expected_ay, 1e-6);
}