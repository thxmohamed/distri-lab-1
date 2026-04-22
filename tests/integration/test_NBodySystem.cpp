#include <gtest/gtest.h>
#include "NBodySystem.h"
#include "Particle.h"
#include <cmath>

// ============================================================
// TEST 1: 2 cuerpos (caso base simple)
// ============================================================

TEST(NBodySystem, TwoBody) {
    NBodySystem sys(1.0, 0.01);

    sys.addParticle(Particle(1.0, -1.0, 0.0, 0.0, 0.0));
    sys.addParticle(Particle(1.0,  1.0, 0.0, 0.0, 0.0));

    sys.computeAccelerations();
    auto serial = sys.getBodies();

    sys.computeAccelerationsParallelSimple();
    auto parallel = sys.getBodies();

    for (int i = 0; i < sys.getCount(); ++i) {
        EXPECT_NEAR(serial[i].getAx(), parallel[i].getAx(), 1e-9);
        EXPECT_NEAR(serial[i].getAy(), parallel[i].getAy(), 1e-9);
    }
}

// ============================================================
// TEST 2: 3 cuerpos (verificable a mano)
// ============================================================

TEST(NBodySystem, ThreeBody) {
    NBodySystem sys(1.0, 0.01);

    sys.addParticle(Particle(1.0, -1.0, 0.0, 0.0, 0.0));
    sys.addParticle(Particle(1.0,  0.0, 0.0, 0.0, 0.0));
    sys.addParticle(Particle(1.0,  1.0, 0.0, 0.0, 0.0));

    sys.computeAccelerations();
    auto serial = sys.getBodies();

    sys.computeAccelerationsParallelSimple();
    auto parallel = sys.getBodies();

    for (int i = 0; i < sys.getCount(); ++i) {
        EXPECT_NEAR(serial[i].getAx(), parallel[i].getAx(), 1e-9);
        EXPECT_NEAR(serial[i].getAy(), parallel[i].getAy(), 1e-9);
    }
}

// ============================================================
// TEST 3: Schedules OpenMP
// ============================================================

TEST(NBodySystem, Schedules) {
    NBodySystem sys(1.0, 0.01);
    sys.initBinary(10, 42);

    sys.computeAccelerations();
    auto ref = sys.getBodies();

    for (int mode = 2; mode <= 5; ++mode) {
        sys.computeAccelerationsMode(mode);
        auto result = sys.getBodies();

        for (int i = 0; i < sys.getCount(); ++i) {
            EXPECT_NEAR(ref[i].getAx(), result[i].getAx(), 1e-6);
            EXPECT_NEAR(ref[i].getAy(), result[i].getAy(), 1e-6);
        }
    }
}