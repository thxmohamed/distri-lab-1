#include <gtest/gtest.h>

#include "NBodySystem.h"
#include "Particle.h"

#include <cmath>

// ============================================================
// VALIDACIÓN FÍSICA DE NBODYSYSTEM
// ============================================================
// Este archivo verifica que la implementación de gravedad cumple:
//
// a_i = G * sum_{j != i} m_j * (r_j - r_i)
//       / (|r_j - r_i|^2 + eps^2)^(3/2)
//
// OBJETIVOS:
// - Validar fórmula física con casos analíticos.
// - Validar dirección de aceleraciones.
// - Validar simetría acción-reacción en términos de fuerza.
// - Validar equilibrio por simetría.
// - Detectar errores en dx, dy, masa, G o epsilon.
// ============================================================


// ------------------------------------------------------------
// TEST 1: Validación analítica 2 cuerpos en eje X
// ------------------------------------------------------------
TEST(NBodySystem_Physics, AnalyticalValidation_TwoBodies_XAxis) {
    const double G   = 1.0;
    const double eps = 0.1;

    const double x1 = 0.0;
    const double y1 = 0.0;

    const double x2 = 1.0;
    const double y2 = 0.0;

    const double m1 = 1.0;
    const double m2 = 1.0;

    NBodySystem sys(G, eps);

    sys.addParticle(Particle(m1, x1, y1, 0.0, 0.0));
    sys.addParticle(Particle(m2, x2, y2, 0.0, 0.0));

    sys.computeAccelerationsSerial();

    const auto& bodies = sys.getBodies();

    ASSERT_EQ(sys.getCount(), 2);

    const double dx = x2 - x1;
    const double dy = y2 - y1;

    const double dist2 = dx * dx + dy * dy + eps * eps;
    const double dist3 = dist2 * std::sqrt(dist2);

    const double expected_ax0 = G * m2 * dx / dist3;
    const double expected_ay0 = G * m2 * dy / dist3;

    const double expected_ax1 = -G * m1 * dx / dist3;
    const double expected_ay1 = -G * m1 * dy / dist3;

    EXPECT_NEAR(bodies[0].getAx(), expected_ax0, 1e-9);
    EXPECT_NEAR(bodies[0].getAy(), expected_ay0, 1e-9);

    EXPECT_NEAR(bodies[1].getAx(), expected_ax1, 1e-9);
    EXPECT_NEAR(bodies[1].getAy(), expected_ay1, 1e-9);

    // Dirección esperada: el cuerpo 0 acelera hacia +x,
    // el cuerpo 1 acelera hacia -x.
    EXPECT_GT(bodies[0].getAx(), 0.0);
    EXPECT_LT(bodies[1].getAx(), 0.0);
}


// ------------------------------------------------------------
// TEST 2: Validación analítica 2 cuerpos en 2D real
// ------------------------------------------------------------
// Este test detecta errores donde la implementación calcula bien
// dx pero mal dy, o ignora la componente y.
TEST(NBodySystem_Physics, AnalyticalValidation_TwoBodies_2D) {
    const double G   = 2.0;
    const double eps = 0.05;

    const double x1 = 0.0;
    const double y1 = 0.0;

    const double x2 = 3.0;
    const double y2 = 4.0;

    const double m1 = 2.0;
    const double m2 = 5.0;

    NBodySystem sys(G, eps);

    sys.addParticle(Particle(m1, x1, y1));
    sys.addParticle(Particle(m2, x2, y2));

    sys.computeAccelerationsSerial();

    const auto& bodies = sys.getBodies();

    ASSERT_EQ(sys.getCount(), 2);

    const double dx = x2 - x1;
    const double dy = y2 - y1;

    const double dist2 = dx * dx + dy * dy + eps * eps;
    const double dist3 = dist2 * std::sqrt(dist2);

    // Aceleración sobre cuerpo 0 causada por cuerpo 1.
    const double expected_ax0 = G * m2 * dx / dist3;
    const double expected_ay0 = G * m2 * dy / dist3;

    // Aceleración sobre cuerpo 1 causada por cuerpo 0.
    const double expected_ax1 = -G * m1 * dx / dist3;
    const double expected_ay1 = -G * m1 * dy / dist3;

    EXPECT_NEAR(bodies[0].getAx(), expected_ax0, 1e-9);
    EXPECT_NEAR(bodies[0].getAy(), expected_ay0, 1e-9);

    EXPECT_NEAR(bodies[1].getAx(), expected_ax1, 1e-9);
    EXPECT_NEAR(bodies[1].getAy(), expected_ay1, 1e-9);
}


// ------------------------------------------------------------
// TEST 3: Acción-reacción en fuerzas, no solo aceleraciones
// ------------------------------------------------------------
// Para masas distintas, las aceleraciones no tienen por qué tener
// la misma magnitud. Lo que debe ser opuesto es:
//
// F_01 = m0 * a0
// F_10 = m1 * a1
//
// Por lo tanto:
// m0 * a0 = -m1 * a1
// ------------------------------------------------------------
TEST(NBodySystem_Physics, TwoBody_ActionReaction_Forces) {
    const double G   = 1.0;
    const double eps = 0.01;

    const double m0 = 2.0;
    const double m1 = 5.0;

    NBodySystem sys(G, eps);

    sys.addParticle(Particle(m0, -1.0, 0.0));
    sys.addParticle(Particle(m1,  1.0, 0.0));

    sys.computeAccelerationsSerial();

    const auto& bodies = sys.getBodies();

    ASSERT_EQ(sys.getCount(), 2);

    const double Fx0 = bodies[0].getMass() * bodies[0].getAx();
    const double Fy0 = bodies[0].getMass() * bodies[0].getAy();

    const double Fx1 = bodies[1].getMass() * bodies[1].getAx();
    const double Fy1 = bodies[1].getMass() * bodies[1].getAy();

    EXPECT_NEAR(Fx0, -Fx1, 1e-9);
    EXPECT_NEAR(Fy0, -Fy1, 1e-9);
}


// ------------------------------------------------------------
// TEST 4: Tres cuerpos simétricos, equilibrio del cuerpo central
// ------------------------------------------------------------
TEST(NBodySystem_Physics, ThreeBody_CenterEquilibrium) {
    NBodySystem sys(1.0, 0.01);

    // Sistema simétrico: izquierda - centro - derecha
    sys.addParticle(Particle(1.0, -1.0, 0.0));
    sys.addParticle(Particle(1.0,  0.0, 0.0));
    sys.addParticle(Particle(1.0,  1.0, 0.0));

    sys.computeAccelerationsSerial();

    const auto& bodies = sys.getBodies();

    ASSERT_EQ(sys.getCount(), 3);

    // En configuración simétrica, el centro debe estar en equilibrio.
    EXPECT_NEAR(bodies[1].getAx(), 0.0, 1e-12);
    EXPECT_NEAR(bodies[1].getAy(), 0.0, 1e-12);
}


// ------------------------------------------------------------
// TEST 5: Dos cuerpos iguales simétricos, aceleraciones opuestas
// ------------------------------------------------------------
// Este test es similar al de acción-reacción, pero para masas iguales,
// por lo que también se cumple ax0 = -ax1.
TEST(NBodySystem_Physics, TwoBody_EqualMasses_AccelerationSymmetry) {
    const double G   = 1.0;
    const double eps = 0.01;

    NBodySystem sys(G, eps);

    sys.addParticle(Particle(1.0, -1.0, 0.0));
    sys.addParticle(Particle(1.0,  1.0, 0.0));

    sys.computeAccelerationsSerial();

    const auto& bodies = sys.getBodies();

    ASSERT_EQ(sys.getCount(), 2);

    EXPECT_NEAR(bodies[0].getAx(), -bodies[1].getAx(), 1e-9);
    EXPECT_NEAR(bodies[0].getAy(), -bodies[1].getAy(), 1e-9);

    EXPECT_NEAR(bodies[0].getAx() + bodies[1].getAx(), 0.0, 1e-9);
    EXPECT_NEAR(bodies[0].getAy() + bodies[1].getAy(), 0.0, 1e-9);
}