#include <gtest/gtest.h>

#include "Particle.h"

#include <stdexcept>
#include <sstream>

// ============================================================
// TEST UNITARIOS: Particle
// ============================================================
// Valida:
// - Construcción válida e inválida
// - Estado inicial correcto (ax=ay=0)
// - kick()  : v += a*dt
// - drift() : r += v*dt
// - zeroAcceleration(), setAcceleration(), addAcceleration()
// - setPosition(), setVelocity()
// ============================================================


// ------------------------------------------------------------
// TEST 1: Constructor almacena todos los parámetros correctamente
// ------------------------------------------------------------
TEST(Particle_Constructor, StoresInitialState) {
    Particle p(2.5, 1.0, -3.0, 0.5, -1.5);

    EXPECT_DOUBLE_EQ(p.getMass(), 2.5);
    EXPECT_DOUBLE_EQ(p.getX(),    1.0);
    EXPECT_DOUBLE_EQ(p.getY(),   -3.0);
    EXPECT_DOUBLE_EQ(p.getVx(),   0.5);
    EXPECT_DOUBLE_EQ(p.getVy(),  -1.5);
    EXPECT_DOUBLE_EQ(p.getAx(),   0.0);
    EXPECT_DOUBLE_EQ(p.getAy(),   0.0);
}


// ------------------------------------------------------------
// TEST 2: Constructor con velocidades por defecto = 0
// ------------------------------------------------------------
TEST(Particle_Constructor, DefaultVelocitiesAreZero) {
    Particle p(1.0, 2.0, 3.0);

    EXPECT_DOUBLE_EQ(p.getVx(), 0.0);
    EXPECT_DOUBLE_EQ(p.getVy(), 0.0);
}


// ------------------------------------------------------------
// TEST 3: Constructor rechaza masa cero o negativa
// ------------------------------------------------------------
TEST(Particle_Constructor, RejectsNonPositiveMass) {
    EXPECT_THROW(Particle(0.0,  0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(Particle(-1.0, 0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(Particle(-1e-9, 0.0, 0.0), std::invalid_argument);
}


// ------------------------------------------------------------
// TEST 4: kick() actualiza velocidad con v += a*dt
// ------------------------------------------------------------
TEST(Particle_Kick, UpdatesVelocity) {
    Particle p(1.0, 0.0, 0.0, 1.0, -2.0);
    p.setAcceleration(3.0, -4.0);

    p.kick(0.5);

    EXPECT_DOUBLE_EQ(p.getVx(),  1.0 + 3.0 * 0.5);
    EXPECT_DOUBLE_EQ(p.getVy(), -2.0 + (-4.0) * 0.5);
}


// ------------------------------------------------------------
// TEST 5: kick() con aceleración cero no modifica la velocidad
// ------------------------------------------------------------
TEST(Particle_Kick, ZeroAccelerationLeavesVelocityUnchanged) {
    Particle p(1.0, 0.0, 0.0, 2.0, -3.0);
    // ax=ay=0 por defecto

    p.kick(1.0);

    EXPECT_DOUBLE_EQ(p.getVx(),  2.0);
    EXPECT_DOUBLE_EQ(p.getVy(), -3.0);
}


// ------------------------------------------------------------
// TEST 6: kick() no modifica la posición
// ------------------------------------------------------------
TEST(Particle_Kick, DoesNotChangePosition) {
    Particle p(1.0, 5.0, -7.0, 1.0, 1.0);
    p.setAcceleration(10.0, 10.0);

    p.kick(2.0);

    EXPECT_DOUBLE_EQ(p.getX(),  5.0);
    EXPECT_DOUBLE_EQ(p.getY(), -7.0);
}


// ------------------------------------------------------------
// TEST 7: drift() actualiza posición con r += v*dt
// ------------------------------------------------------------
TEST(Particle_Drift, UpdatesPosition) {
    Particle p(1.0, 1.0, -2.0, 3.0, -1.0);

    p.drift(0.25);

    EXPECT_DOUBLE_EQ(p.getX(),  1.0 + 3.0 * 0.25);
    EXPECT_DOUBLE_EQ(p.getY(), -2.0 + (-1.0) * 0.25);
}


// ------------------------------------------------------------
// TEST 8: drift() con velocidad cero no modifica la posición
// ------------------------------------------------------------
TEST(Particle_Drift, ZeroVelocityLeavesPositionUnchanged) {
    Particle p(1.0, 4.0, -5.0);
    // vx=vy=0 por defecto

    p.drift(10.0);

    EXPECT_DOUBLE_EQ(p.getX(),  4.0);
    EXPECT_DOUBLE_EQ(p.getY(), -5.0);
}


// ------------------------------------------------------------
// TEST 9: drift() no modifica la velocidad
// ------------------------------------------------------------
TEST(Particle_Drift, DoesNotChangeVelocity) {
    Particle p(1.0, 0.0, 0.0, 2.0, -3.0);

    p.drift(5.0);

    EXPECT_DOUBLE_EQ(p.getVx(),  2.0);
    EXPECT_DOUBLE_EQ(p.getVy(), -3.0);
}


// ------------------------------------------------------------
// TEST 10: zeroAcceleration() pone ax=ay=0
// ------------------------------------------------------------
TEST(Particle_Acceleration, ZeroAccelerationClearsValues) {
    Particle p(1.0, 0.0, 0.0);
    p.setAcceleration(7.0, -3.0);

    p.zeroAcceleration();

    EXPECT_DOUBLE_EQ(p.getAx(), 0.0);
    EXPECT_DOUBLE_EQ(p.getAy(), 0.0);
}


// ------------------------------------------------------------
// TEST 11: setAcceleration() sobreescribe ax y ay
// ------------------------------------------------------------
TEST(Particle_Acceleration, SetAccelerationOverwrites) {
    Particle p(1.0, 0.0, 0.0);
    p.setAcceleration(5.0, -2.0);
    p.setAcceleration(-1.0, 3.5);

    EXPECT_DOUBLE_EQ(p.getAx(), -1.0);
    EXPECT_DOUBLE_EQ(p.getAy(),  3.5);
}


// ------------------------------------------------------------
// TEST 12: addAcceleration() acumula incrementos
// ------------------------------------------------------------
TEST(Particle_Acceleration, AddAccelerationAccumulates) {
    Particle p(1.0, 0.0, 0.0);
    p.setAcceleration(1.0, 2.0);

    p.addAcceleration(0.5, -1.0);
    p.addAcceleration(0.5, -1.0);

    EXPECT_DOUBLE_EQ(p.getAx(), 2.0);
    EXPECT_DOUBLE_EQ(p.getAy(), 0.0);
}


// ------------------------------------------------------------
// TEST 13: setPosition() y setVelocity() actualizan los valores
// ------------------------------------------------------------
TEST(Particle_Setters, SetPositionAndVelocity) {
    Particle p(1.0, 0.0, 0.0);

    p.setPosition(3.0, -4.0);
    p.setVelocity(-1.0, 2.5);

    EXPECT_DOUBLE_EQ(p.getX(),   3.0);
    EXPECT_DOUBLE_EQ(p.getY(),  -4.0);
    EXPECT_DOUBLE_EQ(p.getVx(), -1.0);
    EXPECT_DOUBLE_EQ(p.getVy(),  2.5);
}
