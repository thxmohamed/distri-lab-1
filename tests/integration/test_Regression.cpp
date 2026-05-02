#include <gtest/gtest.h>

#include "NBodySystem.h"
#include "Particle.h"

#include <cmath>

// ============================================================
// TESTS DE REGRESIÓN
// ============================================================
// Se valida:
// - La masa afecta la aceleración calculada
// - Un solo cuerpo tiene aceleración cero
// - Recomputar aceleraciones sin mover partículas da el mismo resultado
// - Las variantes paralelas no alteran el número de cuerpos
// - No hay autointeracción (j == i excluido del bucle interno)
// ============================================================


// ------------------------------------------------------------
// TEST 1: Cambiar la masa del atractor cambia la aceleración
// ------------------------------------------------------------
TEST(Regression, AccelerationIsSensitiveToMass) {
    const double G   = 1.0;
    const double eps = 0.01;

    // Sistema A: m1=1, m2=1
    NBodySystem sysA(G, eps);
    sysA.addParticle(Particle(1.0, 0.0, 0.0));
    sysA.addParticle(Particle(1.0, 1.0, 0.0));
    sysA.computeAccelerationsSerial();
    double axA = sysA.getBodies()[0].getAx();

    // Sistema B: m1=1, m2=10 (masa del segundo cuerpo 10x mayor)
    NBodySystem sysB(G, eps);
    sysB.addParticle(Particle(1.0,  0.0, 0.0));
    sysB.addParticle(Particle(10.0, 1.0, 0.0));
    sysB.computeAccelerationsSerial();
    double axB = sysB.getBodies()[0].getAx();

    // La aceleración sobre el cuerpo 0 debe ser 10x mayor en B.
    EXPECT_NEAR(axB, 10.0 * axA, 1e-9);
}


// ------------------------------------------------------------
// TEST 2: Un solo cuerpo tiene aceleración cero (no hay par j≠i)
// ------------------------------------------------------------
TEST(Regression, SingleBodyHasZeroAcceleration) {
    NBodySystem sys(1.0, 0.01);
    sys.addParticle(Particle(5.0, 3.0, -2.0, 1.0, 0.5));

    sys.computeAccelerationsSerial();

    const auto& bodies = sys.getBodies();
    ASSERT_EQ(bodies.size(), 1u);

    EXPECT_DOUBLE_EQ(bodies[0].getAx(), 0.0);
    EXPECT_DOUBLE_EQ(bodies[0].getAy(), 0.0);
}


// ------------------------------------------------------------
// TEST 3: Recomputar sin mover partículas da el mismo resultado
// ------------------------------------------------------------
TEST(Regression, RecomputeGivesSameResult) {
    NBodySystem sys(1.0, 0.01);
    sys.initBinary(10, 99);

    sys.computeAccelerationsSerial();
    std::vector<double> ax_first, ay_first;
    for (const auto& b : sys.getBodies()) {
        ax_first.push_back(b.getAx());
        ay_first.push_back(b.getAy());
    }

    // Segunda llamada sin modificar posiciones
    sys.computeAccelerationsSerial();

    const auto& bodies = sys.getBodies();
    ASSERT_EQ(bodies.size(), ax_first.size());

    for (size_t i = 0; i < bodies.size(); ++i) {
        EXPECT_DOUBLE_EQ(bodies[i].getAx(), ax_first[i])
            << "Aceleración ax residual en cuerpo i=" << i;
        EXPECT_DOUBLE_EQ(bodies[i].getAy(), ay_first[i])
            << "Aceleración ay residual en cuerpo i=" << i;
    }
}


// ------------------------------------------------------------
// TEST 4: El cómputo paralelo no altera el número de cuerpos
// ------------------------------------------------------------
TEST(Regression, ParallelComputeDoesNotCorruptBodyCount) {
    const int N = 20;

    NBodySystem sys(1.0, 0.01);
    sys.initBinary(N, 7);

    ASSERT_EQ(sys.getCount(), N);

    sys.computeAccelerationsParallelSimple();
    EXPECT_EQ(sys.getCount(), N);

    sys.computeAccelerations(0);   // static
    EXPECT_EQ(sys.getCount(), N);

    sys.computeAccelerations(1);   // dynamic
    EXPECT_EQ(sys.getCount(), N);

    sys.computeAccelerationsCollapse();
    EXPECT_EQ(sys.getCount(), N);
}


// ------------------------------------------------------------
// TEST 5: Sin autointeracción — cuatro cuerpos en cuadrado simétrico
// ------------------------------------------------------------
TEST(Regression, NoSelfInteraction) {
    // Cuatro masas iguales en los vértices de un cuadrado unitario.
    // Por simetría, la resultante sobre el cuerpo 0 (en (+,+))
    // debe apuntar hacia el origen: ax < 0, ay < 0, |ax| == |ay|.
    NBodySystem sys(1.0, 0.001);

    sys.addParticle(Particle(1.0,  1.0,  1.0));
    sys.addParticle(Particle(1.0, -1.0,  1.0));
    sys.addParticle(Particle(1.0, -1.0, -1.0));
    sys.addParticle(Particle(1.0,  1.0, -1.0));

    sys.computeAccelerationsSerial();

    const auto& bodies = sys.getBodies();

    EXPECT_LT(bodies[0].getAx(), 0.0);
    EXPECT_LT(bodies[0].getAy(), 0.0);
    EXPECT_NEAR(std::abs(bodies[0].getAx()), std::abs(bodies[0].getAy()), 1e-9);
}
