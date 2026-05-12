#include <gtest/gtest.h>

#include "NBodySystem.h"
#include "Particle.h"

#include <stdexcept>

// ============================================================
// TEST BÁSICO NBodySystem
// ============================================================
// NBodySystem es el contenedor central del simulador: almacena las
// partículas y encapsula G y epsilon. Antes de cualquier benchmark o
// análisis físico, estos tests verifican que el sistema puede construirse,
// poblarse y limpiarse correctamente, y que las condiciones iniciales
// predefinidas (initBinary, initDisk, initPlummer) son reproducibles
// con la misma semilla y rechazan parámetros inválidos.
//
// Se valida:
// - Constructor: almacena G y epsilon, rechaza epsilon ≤ 0
// - Particle rechaza masa no positiva (integración con NBodySystem)
// - addParticle() y clear(): conteo y vaciado correctos
// - zeroAccelerations(): limpia ax y ay de todas las partículas
// - initBinary, initDisk, initPlummer: reproducibilidad y rechazo de parámetros inválidos
// ============================================================


// ------------------------------------------------------------
// TEST 1: Constructor válido
// ------------------------------------------------------------
TEST(NBodySystem_Basic, Constructor_ValidParameters) {
    NBodySystem sys(1.0, 0.01);

    EXPECT_DOUBLE_EQ(sys.getG(), 1.0);
    EXPECT_DOUBLE_EQ(sys.getEpsilon(), 0.01);
    EXPECT_EQ(sys.getCount(), 0);
}


// ------------------------------------------------------------
// TEST 2: Constructor rechaza epsilon no positivo
// ------------------------------------------------------------
// epsilon=0 elimina el suavizado de Plummer y permite singularidades
// numéricas cuando dos cuerpos se superponen. Debe rechazarse en
// construcción para garantizar que el denominador nunca sea cero.
// ------------------------------------------------------------
TEST(NBodySystem_Basic, Constructor_RejectsNonPositiveEpsilon) {
    EXPECT_THROW(NBodySystem(1.0, 0.0), std::invalid_argument);
    EXPECT_THROW(NBodySystem(1.0, -0.01), std::invalid_argument);
}


// ------------------------------------------------------------
// TEST 3: Particle rechaza masa no positiva
// ------------------------------------------------------------
TEST(NBodySystem_Basic, Particle_RejectsNonPositiveMass) {
    EXPECT_THROW(Particle(0.0, 0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(Particle(-1.0, 0.0, 0.0), std::invalid_argument);
}


// ------------------------------------------------------------
// TEST 4: addParticle y clear
// ------------------------------------------------------------
TEST(NBodySystem_Basic, AddParticle_And_Clear) {
    NBodySystem sys(1.0, 0.01);

    EXPECT_EQ(sys.getCount(), 0);

    sys.addParticle(Particle(1.0, 0.0, 0.0));
    sys.addParticle(Particle(2.0, 1.0, 0.0));

    EXPECT_EQ(sys.getCount(), 2);
    ASSERT_EQ(sys.getBodies().size(), 2u);

    EXPECT_DOUBLE_EQ(sys.getBodies()[0].getMass(), 1.0);
    EXPECT_DOUBLE_EQ(sys.getBodies()[1].getMass(), 2.0);

    sys.clear();

    EXPECT_EQ(sys.getCount(), 0);
    EXPECT_TRUE(sys.getBodies().empty());
}


// ------------------------------------------------------------
// TEST 5: zeroAccelerations
// ------------------------------------------------------------
TEST(NBodySystem_Basic, ZeroAccelerations) {
    NBodySystem sys(1.0, 0.01);

    sys.addParticle(Particle(1.0, 0.0, 0.0));
    sys.addParticle(Particle(1.0, 1.0, 0.0));

    // Primero calculamos aceleraciones para asegurar que no están en cero.
    sys.computeAccelerations();

    auto before = sys.getBodies();
    ASSERT_EQ(before.size(), 2u);
    EXPECT_NE(before[0].getAx(), 0.0);

    // Luego verificamos que zeroAccelerations limpia ax y ay.
    sys.zeroAccelerations();

    auto after = sys.getBodies();

    for (const auto& b : after) {
        EXPECT_NEAR(b.getAx(), 0.0, 1e-12);
        EXPECT_NEAR(b.getAy(), 0.0, 1e-12);
    }
}


// ------------------------------------------------------------
// TEST 6: initBinary genera N partículas reproducibles
// ------------------------------------------------------------
// La reproducibilidad con la misma semilla es esencial para que los
// benchmarks sean comparables entre ejecuciones. Dos sistemas con la
// misma semilla deben tener estado inicial bit a bit idéntico.
// ------------------------------------------------------------
TEST(NBodySystem_Basic, InitBinary_Reproducible) {
    NBodySystem sys1(1.0, 0.01);
    NBodySystem sys2(1.0, 0.01);

    sys1.initBinary(10, 42);
    sys2.initBinary(10, 42);

    ASSERT_EQ(sys1.getCount(), 10);
    ASSERT_EQ(sys2.getCount(), 10);

    const auto& b1 = sys1.getBodies();
    const auto& b2 = sys2.getBodies();

    for (int i = 0; i < sys1.getCount(); ++i) {
        EXPECT_DOUBLE_EQ(b1[i].getMass(), b2[i].getMass());
        EXPECT_DOUBLE_EQ(b1[i].getX(),    b2[i].getX());
        EXPECT_DOUBLE_EQ(b1[i].getY(),    b2[i].getY());
        EXPECT_DOUBLE_EQ(b1[i].getVx(),   b2[i].getVx());
        EXPECT_DOUBLE_EQ(b1[i].getVy(),   b2[i].getVy());
    }
}


// ------------------------------------------------------------
// TEST 7: initBinary rechaza N inválido
// ------------------------------------------------------------
// initBinary requiere al menos 2 cuerpos (las dos masas dominantes).
// Con N<2 el sistema binario no tiene sentido físico y debe rechazarse.
// ------------------------------------------------------------
TEST(NBodySystem_Basic, InitBinary_RejectsInvalidN) {
    NBodySystem sys(1.0, 0.01);

    EXPECT_THROW(sys.initBinary(0, 42), std::invalid_argument);
    EXPECT_THROW(sys.initBinary(1, 42), std::invalid_argument);
}


// ------------------------------------------------------------
// TEST 8: initDisk rechaza parámetros inválidos
// ------------------------------------------------------------
TEST(NBodySystem_Basic, InitDisk_RejectsInvalidParameters) {
    NBodySystem sys(1.0, 0.01);

    EXPECT_THROW(sys.initDisk(0, 1.0, 42), std::invalid_argument);
    EXPECT_THROW(sys.initDisk(10, 0.0, 42), std::invalid_argument);
    EXPECT_THROW(sys.initDisk(10, -1.0, 42), std::invalid_argument);
}


// ------------------------------------------------------------
// TEST 9: initPlummer rechaza parámetros inválidos
// ------------------------------------------------------------
TEST(NBodySystem_Basic, InitPlummer_RejectsInvalidParameters) {
    NBodySystem sys(1.0, 0.01);

    EXPECT_THROW(sys.initPlummer(0, 0.5, 42), std::invalid_argument);
    EXPECT_THROW(sys.initPlummer(10, 0.0, 42), std::invalid_argument);
    EXPECT_THROW(sys.initPlummer(10, -0.5, 42), std::invalid_argument);
}