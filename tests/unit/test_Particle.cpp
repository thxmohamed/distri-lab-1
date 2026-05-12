#include <gtest/gtest.h>

#include "Particle.h"

#include <stdexcept>
#include <sstream>

// ============================================================
// TEST UNITARIOS: Particle
// ============================================================
// Particle es la unidad mínima del simulador: encapsula masa,
// posición (x, y), velocidad (vx, vy) y aceleración (ax, ay).
// Sus métodos kick() y drift() implementan los dos sub-pasos del
// integrador de Euler: kick actualiza velocidad con v += a*dt y
// drift actualiza posición con r += v*dt. La correcta separación
// de estas responsabilidades es fundamental para que el integrador
// funcione en el orden correcto sin efectos cruzados.
//
// Se valida:
// - Constructor: almacena estado inicial y rechaza masa no positiva
// - Estado inicial correcto (ax=ay=0 siempre)
// - kick()  : v += a*dt, sin efecto sobre posición
// - drift() : r += v*dt, sin efecto sobre velocidad
// - zeroAcceleration(), setAcceleration(), addAcceleration()
// - setPosition(), setVelocity()
// ============================================================


// ------------------------------------------------------------
// TEST 1: Constructor almacena todos los parámetros correctamente
// ------------------------------------------------------------
// Verifica que los cinco parámetros (masa, x, y, vx, vy) quedan
// almacenados con exactitud doble y que ax=ay=0 al construir,
// ya que las aceleraciones se calculan externamente por NBodySystem.
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
// El constructor de tres argumentos omite vx y vy; deben inicializarse
// en cero para que partículas estáticas no tengan movimiento espurio.
// ------------------------------------------------------------
TEST(Particle_Constructor, DefaultVelocitiesAreZero) {
    Particle p(1.0, 2.0, 3.0);

    EXPECT_DOUBLE_EQ(p.getVx(), 0.0);
    EXPECT_DOUBLE_EQ(p.getVy(), 0.0);
}


// ------------------------------------------------------------
// TEST 3: Constructor rechaza masa cero o negativa
// ------------------------------------------------------------
// Una masa no positiva produciría divisiones por cero o fuerzas
// invertidas en la fórmula gravitacional. El constructor debe lanzar
// std::invalid_argument antes de que el objeto quede en estado inválido.
// ------------------------------------------------------------
TEST(Particle_Constructor, RejectsNonPositiveMass) {
    EXPECT_THROW(Particle(0.0,  0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(Particle(-1.0, 0.0, 0.0), std::invalid_argument);
    EXPECT_THROW(Particle(-1e-9, 0.0, 0.0), std::invalid_argument);
}


// ------------------------------------------------------------
// TEST 4: kick() actualiza velocidad con v += a*dt
// ------------------------------------------------------------
// Primer sub-paso del integrador de Euler. Usa valores no triviales
// de v y a en ambas componentes para detectar errores de signo o
// mezcla de componentes (ej. ax aplicado sobre vy).
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
// Caso borde: si ax=ay=0 (sin fuerza), la velocidad debe ser invariante.
// Detecta implementaciones que suman un término no nulo por defecto.
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
// kick() solo actualiza velocidad; la posición se actualiza en drift().
// Este test detecta implementaciones que mezclan los dos sub-pasos,
// lo que produciría un integrador incorrecto.
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
// Segundo sub-paso del integrador de Euler. La posición debe avanzar
// con la velocidad actual (post-kick), no con la velocidad anterior.
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
// Caso borde: partícula estática (vx=vy=0) no debe desplazarse aunque
// se llame drift(). Detecta términos espurios en la implementación.
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
// drift() solo actualiza posición; la velocidad es responsabilidad de
// kick(). Detecta implementaciones que actualizan ambos en un único paso.
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
// Antes de cada paso de cómputo, NBodySystem llama zeroAccelerations()
// para limpiar las acumulaciones anteriores. Si no funciona, las
// aceleraciones crecerían indefinidamente entre pasos temporales.
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
// Verifica que una segunda llamada descarta el valor anterior en lugar
// de acumular, lo que distingue setAcceleration de addAcceleration.
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
// La variante collapse(2) de computeAccelerations usa addAcceleration
// con atomic para sumar contribuciones de distintos hilos. Verifica
// que la acumulación es correcta tras múltiples llamadas sucesivas.
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
// Estos setters los usa el Integrator para aplicar kick y drift.
// Verifica que ambas componentes se almacenan correctamente y de
// forma independiente entre sí.
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
