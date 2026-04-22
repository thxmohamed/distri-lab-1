#include <gtest/gtest.h>
#include "NBodySystem.h"
#include "Particle.h"
#include <cmath>

// ============================================================
// TEST DE PARALELISMO N-BODY SYSTEM
// ============================================================
// OBJETIVO:
// Verificar que todas las implementaciones OpenMP producen
// el mismo resultado que la versión SERIAL.
//
// Se valida:
// -Parallel simple vs serial
// -OpenMP schedules (static, dynamic, guided)
// -collapse(2)
// -computeAccelerationsMode()
// ============================================================

// ------------------------------------------------------------
// TEST 1: Parallel Simple vs Serial
// ------------------------------------------------------------
TEST(NBodySystem_Parallel, ParallelSimple_vs_Serial) {

    NBodySystem sys(1.0, 0.01);
    sys.initBinary(10, 42);

    // SERIAL
    sys.computeAccelerations();
    auto ref = sys.getBodies();

    // PARALLEL SIMPLE
    sys.computeAccelerationsParallelSimple();
    auto test = sys.getBodies();

    ASSERT_EQ(ref.size(), test.size());

    for (size_t i = 0; i < ref.size(); ++i) {

        EXPECT_NEAR(ref[i].getAx(), test[i].getAx(), 1e-6);
        EXPECT_NEAR(ref[i].getAy(), test[i].getAy(), 1e-6);
    }
}


// ------------------------------------------------------------
// TEST 2: OpenMP Schedules + Collapse
// ------------------------------------------------------------
TEST(NBodySystem_Parallel, OpenMP_Schedules_And_Collapse) {

    NBodySystem sys(1.0, 0.01);
    sys.initBinary(10, 42);

    // referencia serial
    sys.computeAccelerations();
    auto ref = sys.getBodies();

    for (int mode = 2; mode <= 5; ++mode) {

        sys.computeAccelerationsMode(mode);
        auto test = sys.getBodies();

        ASSERT_EQ(ref.size(), test.size());

        for (size_t i = 0; i < ref.size(); ++i) {

            EXPECT_NEAR(ref[i].getAx(), test[i].getAx(), 1e-6);
            EXPECT_NEAR(ref[i].getAy(), test[i].getAy(), 1e-6);
        }
    }
}