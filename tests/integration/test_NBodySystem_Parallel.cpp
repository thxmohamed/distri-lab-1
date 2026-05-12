#include <gtest/gtest.h>

#include "NBodySystem.h"
#include "Particle.h"

#include <cmath>
#include <stdexcept>
#include <vector>

// ============================================================
// TEST DE PARALELISMO N-BODY SYSTEM
// ============================================================
// NBodySystem expone varias variantes paralelas del cómputo de
// aceleraciones O(N²): paralelo simple, schedules static/dynamic/guided
// con chunk configurable y collapse(2). Todas deben producir el mismo
// resultado físico que la versión serial de referencia.
//
// La tolerancia de comparación es 1e-6: las variantes con atomic
// (como collapse) pueden acumular en orden distinto, introduciendo
// diferencias de redondeo en coma flotante que son correctas.
// La versión serial vs serial se verifica a 1e-12.
//
// Se valida:
// - Parallel simple vs serial
// - OpenMP schedules static, dynamic, guided sin chunk explícito
// - OpenMP schedules con chunk explícito {1, 2, 4}
// - collapse(2) con atomic vs serial
// - computeAccelerationsMode() para modos 0 a 5
// - Rechazo de schedule_type inválido y chunk_size ≤ 0
// - Rechazo de modo inválido en computeAccelerationsMode()
// ============================================================


// ------------------------------------------------------------
// Helper local para comparar aceleraciones entre dos estados
// ------------------------------------------------------------
static void expectAccelerationsNear(
    const std::vector<Particle>& ref,
    const std::vector<Particle>& test,
    double tol = 1e-6)
{
    ASSERT_EQ(ref.size(), test.size());

    for (size_t i = 0; i < ref.size(); ++i) {
        EXPECT_NEAR(ref[i].getAx(), test[i].getAx(), tol)
            << "Diferencia en ax para cuerpo i=" << i;

        EXPECT_NEAR(ref[i].getAy(), test[i].getAy(), tol)
            << "Diferencia en ay para cuerpo i=" << i;
    }
}


// ------------------------------------------------------------
// TEST 1: Parallel Simple vs Serial
// ------------------------------------------------------------
TEST(NBodySystem_Parallel, ParallelSimple_vs_Serial) {
    NBodySystem sys(1.0, 0.01);
    sys.initBinary(10, 42);

    // Referencia serial
    sys.computeAccelerationsSerial();
    auto ref = sys.getBodies();

    // Paralelo simple
    sys.computeAccelerationsParallelSimple();
    auto test = sys.getBodies();

    expectAccelerationsNear(ref, test, 1e-6);
}


// ------------------------------------------------------------
// TEST 2: computeAccelerations() es equivalente a Serial
// ------------------------------------------------------------
// Este test asegura que, en la versión compatible del proyecto,
// computeAccelerations() sigue funcionando como referencia serial.
TEST(NBodySystem_Parallel, DefaultCompute_Equals_Serial) {
    NBodySystem sys(1.0, 0.01);
    sys.initBinary(12, 42);

    sys.computeAccelerationsSerial();
    auto ref = sys.getBodies();

    sys.computeAccelerations();
    auto test = sys.getBodies();

    expectAccelerationsNear(ref, test, 1e-12);
}


// ------------------------------------------------------------
// TEST 3: OpenMP schedules sin chunk explícito
// ------------------------------------------------------------
TEST(NBodySystem_Parallel, OpenMP_Schedules_vs_Serial) {
    NBodySystem sys(1.0, 0.01);
    sys.initBinary(10, 42);

    // Referencia serial
    sys.computeAccelerationsSerial();
    auto ref = sys.getBodies();

    // 0 = static, 1 = dynamic, 2 = guided
    for (int schedule = 0; schedule <= 2; ++schedule) {
        sys.computeAccelerations(schedule);
        auto test = sys.getBodies();

        expectAccelerationsNear(ref, test, 1e-6);
    }
}


// ------------------------------------------------------------
// TEST 4: OpenMP schedules con chunk explícito
// ------------------------------------------------------------
TEST(NBodySystem_Parallel, OpenMP_Schedules_WithExplicitChunk_vs_Serial) {
    NBodySystem sys(1.0, 0.01);
    sys.initBinary(16, 42);

    // Referencia serial
    sys.computeAccelerationsSerial();
    auto ref = sys.getBodies();

    const int chunks[] = {1, 2, 4};

    // 0 = static, 1 = dynamic, 2 = guided
    for (int schedule = 0; schedule <= 2; ++schedule) {
        for (int chunk : chunks) {
            sys.computeAccelerations(schedule, chunk);
            auto test = sys.getBodies();

            expectAccelerationsNear(ref, test, 1e-6);
        }
    }
}


// ------------------------------------------------------------
// TEST 5: collapse(2) vs serial
// ------------------------------------------------------------
TEST(NBodySystem_Parallel, Collapse_vs_Serial) {
    NBodySystem sys(1.0, 0.01);
    sys.initBinary(10, 42);

    // Referencia serial
    sys.computeAccelerationsSerial();
    auto ref = sys.getBodies();

    // Variante collapse(2)
    sys.computeAccelerationsCollapse();
    auto test = sys.getBodies();

    // collapse usa atomic y puede acumular en orden distinto.
    expectAccelerationsNear(ref, test, 1e-6);
}


// ------------------------------------------------------------
// TEST 6: computeAccelerationsMode cubre modos 0 a 5
// ------------------------------------------------------------
TEST(NBodySystem_Parallel, ComputeAccelerationsMode_AllValidModes_vs_Serial) {
    NBodySystem sys(1.0, 0.01);
    sys.initBinary(10, 42);

    // Referencia serial
    sys.computeAccelerationsSerial();
    auto ref = sys.getBodies();

    // 0 = serial
    // 1 = paralelo simple
    // 2 = static
    // 3 = dynamic
    // 4 = guided
    // 5 = collapse
    for (int mode = 0; mode <= 5; ++mode) {
        sys.computeAccelerationsMode(mode);
        auto test = sys.getBodies();

        expectAccelerationsNear(ref, test, 1e-6);
    }
}


// ------------------------------------------------------------
// TEST 7: parámetros inválidos para schedules
// ------------------------------------------------------------
// schedule_type fuera de {0,1,2} y chunk_size ≤ 0 no tienen sentido
// físico ni en OpenMP. Deben rechazarse con std::invalid_argument
// antes de entrar a la región paralela.
// ------------------------------------------------------------
TEST(NBodySystem_Parallel, RejectsInvalidScheduleArguments) {
    NBodySystem sys(1.0, 0.01);
    sys.initBinary(10, 42);

    EXPECT_THROW(sys.computeAccelerations(-1, 1), std::invalid_argument);
    EXPECT_THROW(sys.computeAccelerations(3, 1), std::invalid_argument);

    EXPECT_THROW(sys.computeAccelerations(0, 0), std::invalid_argument);
    EXPECT_THROW(sys.computeAccelerations(0, -2), std::invalid_argument);
}


// ------------------------------------------------------------
// TEST 8: modo inválido en computeAccelerationsMode
// ------------------------------------------------------------
// Un modo fuera del rango 0–5 no corresponde a ninguna variante
// definida; debe lanzar std::invalid_argument para evitar comportamiento
// indefinido por caer en un branch no inicializado del dispatcher.
// ------------------------------------------------------------
TEST(NBodySystem_Parallel, ComputeAccelerationsMode_RejectsInvalidMode) {
    NBodySystem sys(1.0, 0.01);
    sys.initBinary(10, 42);

    EXPECT_THROW(sys.computeAccelerationsMode(99), std::invalid_argument);
}