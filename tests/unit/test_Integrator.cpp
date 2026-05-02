#include <gtest/gtest.h>

#include "Integrator.h"
#include "Particle.h"

#include <stdexcept>
#include <vector>

// ============================================================
// TEST UNITARIOS: Integrator
// ============================================================
// Valida:
// - applyKick()  : v += a*dt para todas las partículas
// - applyDrift() : r += v*dt para todas las partículas
// - stepEuler()  : kick seguido de drift (orden correcto)
// - Rechazo de dt no positivo en applyKick y applyDrift
// - Comportamiento con vector vacío (sin crash)
// ============================================================


// ------------------------------------------------------------
// TEST 1: applyKick actualiza velocidades de todas las partículas
// ------------------------------------------------------------
TEST(Integrator_ApplyKick, UpdatesAllVelocities) {
    std::vector<Particle> bodies;
    bodies.emplace_back(1.0, 0.0, 0.0, 1.0,  2.0);
    bodies.emplace_back(1.0, 0.0, 0.0, -3.0, 0.5);

    bodies[0].setAcceleration(2.0, -1.0);
    bodies[1].setAcceleration(-1.0, 4.0);

    Integrator::applyKick(bodies, 0.5);

    EXPECT_DOUBLE_EQ(bodies[0].getVx(),  1.0 + 2.0  * 0.5);
    EXPECT_DOUBLE_EQ(bodies[0].getVy(),  2.0 + (-1.0) * 0.5);
    EXPECT_DOUBLE_EQ(bodies[1].getVx(), -3.0 + (-1.0) * 0.5);
    EXPECT_DOUBLE_EQ(bodies[1].getVy(),  0.5 + 4.0  * 0.5);
}


// ------------------------------------------------------------
// TEST 2: applyKick no modifica las posiciones
// ------------------------------------------------------------
TEST(Integrator_ApplyKick, DoesNotChangePositions) {
    std::vector<Particle> bodies;
    bodies.emplace_back(1.0, 3.0, -2.0, 1.0, 1.0);
    bodies[0].setAcceleration(10.0, 10.0);

    Integrator::applyKick(bodies, 1.0);

    EXPECT_DOUBLE_EQ(bodies[0].getX(),  3.0);
    EXPECT_DOUBLE_EQ(bodies[0].getY(), -2.0);
}


// ------------------------------------------------------------
// TEST 3: applyKick rechaza dt no positivo
// ------------------------------------------------------------
TEST(Integrator_ApplyKick, RejectsNonPositiveDt) {
    std::vector<Particle> bodies;
    bodies.emplace_back(1.0, 0.0, 0.0);

    EXPECT_THROW(Integrator::applyKick(bodies, 0.0),  std::invalid_argument);
    EXPECT_THROW(Integrator::applyKick(bodies, -0.1), std::invalid_argument);
}


// ------------------------------------------------------------
// TEST 4: applyDrift actualiza posiciones de todas las partículas
// ------------------------------------------------------------
TEST(Integrator_ApplyDrift, UpdatesAllPositions) {
    std::vector<Particle> bodies;
    bodies.emplace_back(1.0, 1.0,  2.0, 3.0, -1.0);
    bodies.emplace_back(1.0, -4.0, 0.0, 0.5,  2.5);

    Integrator::applyDrift(bodies, 0.2);

    EXPECT_DOUBLE_EQ(bodies[0].getX(),  1.0 + 3.0 * 0.2);
    EXPECT_DOUBLE_EQ(bodies[0].getY(),  2.0 + (-1.0) * 0.2);
    EXPECT_DOUBLE_EQ(bodies[1].getX(), -4.0 + 0.5 * 0.2);
    EXPECT_DOUBLE_EQ(bodies[1].getY(),  0.0 + 2.5 * 0.2);
}


// ------------------------------------------------------------
// TEST 5: applyDrift no modifica las velocidades
// ------------------------------------------------------------
TEST(Integrator_ApplyDrift, DoesNotChangeVelocities) {
    std::vector<Particle> bodies;
    bodies.emplace_back(1.0, 0.0, 0.0, 5.0, -3.0);

    Integrator::applyDrift(bodies, 2.0);

    EXPECT_DOUBLE_EQ(bodies[0].getVx(),  5.0);
    EXPECT_DOUBLE_EQ(bodies[0].getVy(), -3.0);
}


// ------------------------------------------------------------
// TEST 6: applyDrift rechaza dt no positivo
// ------------------------------------------------------------
TEST(Integrator_ApplyDrift, RejectsNonPositiveDt) {
    std::vector<Particle> bodies;
    bodies.emplace_back(1.0, 0.0, 0.0);

    EXPECT_THROW(Integrator::applyDrift(bodies, 0.0),  std::invalid_argument);
    EXPECT_THROW(Integrator::applyDrift(bodies, -1.0), std::invalid_argument);
}


// ------------------------------------------------------------
// TEST 7: stepEuler aplica kick antes que drift
// ------------------------------------------------------------
TEST(Integrator_StepEuler, AppliesKickThenDrift) {
    std::vector<Particle> bodies;
    // v0=(0,0), a=(2,0), pos=(0,0)
    bodies.emplace_back(1.0, 0.0, 0.0, 0.0, 0.0);
    bodies[0].setAcceleration(2.0, 0.0);

    const double dt = 1.0;
    Integrator::stepEuler(bodies, dt);

    // Tras kick: vx = 0 + 2*1 = 2
    // Tras drift: x = 0 + 2*1 = 2
    EXPECT_DOUBLE_EQ(bodies[0].getVx(), 2.0);
    EXPECT_DOUBLE_EQ(bodies[0].getX(),  2.0);
}


// ------------------------------------------------------------
// TEST 8: stepEuler — la posición usa la velocidad post-kick
// ------------------------------------------------------------
TEST(Integrator_StepEuler, PositionUsesUpdatedVelocity) {
    std::vector<Particle> bodies;
    // v0=(0,0), a=(1,0), pos=(5,0)
    bodies.emplace_back(1.0, 5.0, 0.0, 0.0, 0.0);
    bodies[0].setAcceleration(1.0, 0.0);

    Integrator::stepEuler(bodies, 2.0);

    // kick: vx = 0 + 1*2 = 2
    // drift: x = 5 + 2*2 = 9
    EXPECT_DOUBLE_EQ(bodies[0].getVx(), 2.0);
    EXPECT_DOUBLE_EQ(bodies[0].getX(),  9.0);
}


// ------------------------------------------------------------
// TEST 9: vector vacío no produce error
// ------------------------------------------------------------
TEST(Integrator_EdgeCases, EmptyVectorIsNoOp) {
    std::vector<Particle> empty;

    EXPECT_NO_THROW(Integrator::applyKick(empty,  0.1));
    EXPECT_NO_THROW(Integrator::applyDrift(empty, 0.1));
    EXPECT_NO_THROW(Integrator::stepEuler(empty,  0.1));
}


// ------------------------------------------------------------
// TEST 10: stepEuler sobre una sola partícula con movimiento 2D
// ------------------------------------------------------------
TEST(Integrator_StepEuler, SingleParticle2D) {
    std::vector<Particle> bodies;
    bodies.emplace_back(1.0, 1.0, 2.0, -1.0, 0.5);
    bodies[0].setAcceleration(0.0, -2.0);

    const double dt = 0.1;
    Integrator::stepEuler(bodies, dt);

    // kick: vx=-1+0*0.1=-1, vy=0.5+(-2)*0.1=0.3
    // drift: x=1+(-1)*0.1=0.9, y=2+0.3*0.1=2.03
    EXPECT_DOUBLE_EQ(bodies[0].getVx(), -1.0);
    EXPECT_NEAR(bodies[0].getVy(),  0.3, 1e-12);
    EXPECT_NEAR(bodies[0].getX(),   0.9, 1e-12);
    EXPECT_NEAR(bodies[0].getY(),   2.03, 1e-12);
}
