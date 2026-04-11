#include <gtest/gtest.h>
#include "model/NBodySystem.h"
#include "model/Particle.h"
#include <cmath>

// ============================================================
// TEST BÁSICO NBodySystem (FÍSICA)
// ============================================================
// Valida:
// -Simetría física (acción-reacción)
// -Dirección de fuerzas
// -Casos simples controlados
// ============================================================


// ------------------------------------------------------------
// TEST 1: Two Body (Simetría física)
// ------------------------------------------------------------
TEST(NBodySystem_Physics, TwoBody_Symmetry) {

    double G = 1.0;
    double eps = 0.01;

    NBodySystem sys(G, eps);

    // Configuración simétrica en eje X
    double x1 = -1.0;
    double x2 =  1.0;

    sys.addParticle(Particle(1.0, x1, 0.0, 0.0, 0.0));
    sys.addParticle(Particle(1.0, x2, 0.0, 0.0, 0.0));

    sys.computeAccelerations();
    auto bodies = sys.getBodies();

    ASSERT_EQ(sys.getCount(), 2);

    double ax0 = bodies[0].getAx();
    double ax1 = bodies[1].getAx();

    //acción-reacción: fuerzas opuestas exactas
    EXPECT_NEAR(ax0, -ax1, 1e-9);

    //consistencia global del sistema
    EXPECT_NEAR(ax0 + ax1, 0.0, 1e-9);
}


// ------------------------------------------------------------
// TEST 2: Three Body (Equilibrio central)
// ------------------------------------------------------------
TEST(NBodySystem_Physics, ThreeBody_CenterEquilibrium) {

    NBodySystem sys(1.0, 0.01);

    // Sistema simétrico: izquierda - centro - derecha
    double xL = -1.0;
    double xC =  0.0;
    double xR =  1.0;

    sys.addParticle(Particle(1.0, xL, 0.0, 0.0, 0.0));
    sys.addParticle(Particle(1.0, xC, 0.0, 0.0, 0.0));
    sys.addParticle(Particle(1.0, xR, 0.0, 0.0, 0.0));

    sys.computeAccelerations();
    auto bodies = sys.getBodies();

    ASSERT_EQ(sys.getCount(), 3);

    double ax_center = bodies[1].getAx();

    //en configuración simétrica, el centro debe estar en equilibrio
    EXPECT_NEAR(ax_center, 0.0, 1e-9);
}