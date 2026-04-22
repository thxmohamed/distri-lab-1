#include <gtest/gtest.h>
#include "NBodySystem.h"
#include "Particle.h"
#include "NBodySimulator.h"
#include <cmath>

// ============================================================
// TEST DE INTEGRACIÓN N-BODY SIMULATOR
// ============================================================
// OBJETIVO:
// Verificar que el lazo temporal, el cálculo de energía y las
// variantes instrumentadas de NBodySimulator se comportan de
// forma correcta sobre sistemas físicos simples.
//
// Se valida:
// -Integración temporal básica con un solo cuerpo
// -Cálculo de energía cinética
// -Cálculo de energía potencial
// -Consistencia entre calculateEnergy con reduce y atomic
// -Ejecución coherente de integrateEuler instrumentado
// ============================================================

// ------------------------------------------------------------
// TEST 1: Un cuerpo sin interacción
// ------------------------------------------------------------
TEST(NBodySimulator, SingleBody_ConstantVelocity) {
    NBodySystem sys(1.0, 0.01);
    sys.addParticle(Particle(1.0, 0.0, 0.0, 1.0, 2.0));

    NBodySimulator sim(&sys, 0.1);
    sim.integrateEuler();

    const auto& bodies = sys.getBodies();
    ASSERT_EQ(bodies.size(), 1u);

    EXPECT_NEAR(bodies[0].getVx(), 1.0, 1e-12);
    EXPECT_NEAR(bodies[0].getVy(), 2.0, 1e-12);
    EXPECT_NEAR(bodies[0].getX(), 0.1, 1e-12);
    EXPECT_NEAR(bodies[0].getY(), 0.2, 1e-12);
}

// ------------------------------------------------------------
// TEST 2: Energía cinética conocida
// ------------------------------------------------------------
TEST(NBodySimulator, KineticEnergy) {
    NBodySystem sys(1.0, 0.01);
    sys.addParticle(Particle(2.0, 0.0, 0.0, 3.0, 4.0));

    NBodySimulator sim(&sys, 0.1);

    double K = sim.calculateKineticEnergy();
    EXPECT_NEAR(K, 25.0, 1e-12);
}

// ------------------------------------------------------------
// TEST 3: Energía potencial en 2 cuerpos
// ------------------------------------------------------------
TEST(NBodySimulator, PotentialEnergy_TwoBody) {
    double G = 1.0;
    double eps = 0.1;

    NBodySystem sys(G, eps);
    sys.addParticle(Particle(1.0, 0.0, 0.0, 0.0, 0.0));
    sys.addParticle(Particle(1.0, 1.0, 0.0, 0.0, 0.0));

    NBodySimulator sim(&sys, 0.1);

    double expected = -G / std::sqrt(1.0 + eps * eps);
    EXPECT_NEAR(sim.calculatePotentialEnergy(), expected, 1e-12);
}

// ------------------------------------------------------------
// TEST 4: reduce vs atomic
// ------------------------------------------------------------
TEST(NBodySimulator, EnergyReduceVsAtomic) {
    NBodySystem sys(1.0, 0.01);
    sys.initBinary(12, 42);

    NBodySimulator sim(&sys, 0.001);

    double E_reduce = sim.calculateEnergy(0, true);
    double E_atomic = sim.calculateEnergy(1, true);

    EXPECT_NEAR(E_reduce, E_atomic, 1e-9);
}

// ------------------------------------------------------------
// TEST 5: integrateEuler instrumentado
// ------------------------------------------------------------
TEST(NBodySimulator, IntegrateEulerVariantsRun) {
    NBodySystem sys0(1.0, 0.01);
    sys0.initBinary(8, 42);

    NBodySystem sys1 = sys0;
    NBodySystem sys2 = sys0;
    NBodySystem sys3 = sys0;

    NBodySimulator sim_ref(&sys0, 0.001);
    NBodySimulator sim_atomic(&sys1, 0.001);
    NBodySimulator sim_critical(&sys2, 0.001);
    NBodySimulator sim_nowait(&sys3, 0.001);

    sim_ref.integrateEuler();
    sim_atomic.integrateEuler(0);
    sim_critical.integrateEuler(1);
    sim_nowait.integrateEuler(2);

    const auto& ref = sys0.getBodies();
    const auto& a   = sys1.getBodies();
    const auto& c   = sys2.getBodies();
    const auto& n   = sys3.getBodies();

    ASSERT_EQ(ref.size(), a.size());
    ASSERT_EQ(ref.size(), c.size());
    ASSERT_EQ(ref.size(), n.size());

    for (size_t i = 0; i < ref.size(); ++i) {
        EXPECT_NEAR(ref[i].getX(), a[i].getX(), 1e-9);
        EXPECT_NEAR(ref[i].getY(), a[i].getY(), 1e-9);
        EXPECT_NEAR(ref[i].getX(), c[i].getX(), 1e-9);
        EXPECT_NEAR(ref[i].getY(), c[i].getY(), 1e-9);
        EXPECT_NEAR(ref[i].getX(), n[i].getX(), 1e-9);
        EXPECT_NEAR(ref[i].getY(), n[i].getY(), 1e-9);
    }
}