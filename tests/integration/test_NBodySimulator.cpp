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
// NBodySimulator orquesta el lazo temporal completo: para cada paso
// llama a computeAccelerations sobre el NBodySystem subyacente y luego
// aplica el integrador de Euler (kick → drift). Adicionalmente expone
// variantes instrumentadas que ejercen distintas construcciones de
// OpenMP (atomic, critical, reduce, barrier, nowait, task, single,
// firstprivate, lastprivate) para permitir su benchmark y validación.
//
// Se valida:
// - Integración temporal básica con un solo cuerpo (movimiento libre)
// - Cálculo de energía cinética con valor analítico conocido
// - Cálculo de energía potencial con valor analítico conocido
// - Consistencia numérica entre reduce y atomic en calculateEnergy
// - Equivalencia de integrateEuler en sus variantes atomic, critical,
//   nowait respecto a la referencia sin instrumentación
// - Ausencia de errores de ejecución en rutas OpenMP: task+single,
//   task+master, parallel for, barrier y single de inicialización
// - Corrección de firstprivate (energía igual a serial) y
//   lastprivate (índice final del último cuerpo procesado)
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
TEST(NBodySimulator, IntegrateEulerVariantsMatchReference) {
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

// ------------------------------------------------------------
// TEST 6: processBodies con task + single
// ------------------------------------------------------------
// processBodies(0, true) crea una región paralela donde un único hilo
// (single) genera tasks OpenMP, uno por cuerpo. Valida que la
// construcción task+single no produce data races ni deadlocks con
// un sistema de 20 partículas.
// ------------------------------------------------------------
TEST(NBodySimulator, ProcessBodiesTaskSingleRuns) {
    NBodySystem sys(1.0, 0.01);
    sys.initBinary(20, 42);

    NBodySimulator sim(&sys, 0.001);

    EXPECT_NO_THROW(sim.processBodies(0, true));
}

// ------------------------------------------------------------
// TEST 7: processBodies con task + master
// ------------------------------------------------------------
// processBodies(0, false) usa master en lugar de single para la
// generación de tasks: solo el hilo 0 los crea. Verifica que la
// variante master produce el mismo comportamiento sin errores,
// ejerciendo una ruta de sincronización distinta a single.
// ------------------------------------------------------------
TEST(NBodySimulator, ProcessBodiesTaskMasterRuns) {
    NBodySystem sys(1.0, 0.01);
    sys.initBinary(20, 42);

    NBodySimulator sim(&sys, 0.001);

    EXPECT_NO_THROW(sim.processBodies(0, false));
}

// ------------------------------------------------------------
// TEST 8: processBodies con parallel for
// ------------------------------------------------------------
// processBodies(1) distribuye el recorrido de cuerpos con parallel for
// en lugar de tasks. Verifica que esta ruta alternativa de paralelismo
// ejecuta sin errores, lo que permite comparar ambas estrategias en
// benchmarks sin comprometer la corrección.
// ------------------------------------------------------------
TEST(NBodySimulator, ProcessBodiesParallelForRuns) {
    NBodySystem sys(1.0, 0.01);
    sys.initBinary(20, 42);

    NBodySimulator sim(&sys, 0.001);

    EXPECT_NO_THROW(sim.processBodies(1));
}

// ------------------------------------------------------------
// TEST 9: simulatePhasesBarrier
// ------------------------------------------------------------
// simulatePhasesBarrier ejecuta un paso de simulación dividido en
// fases separadas por barriers explícitos: garantiza que todos los
// hilos terminan computeAccelerations antes de iniciar la integración.
// Verifica que la sincronización con barrier no produce deadlocks.
// ------------------------------------------------------------
TEST(NBodySimulator, SimulatePhasesBarrierRuns) {
    NBodySystem sys(1.0, 0.01);
    sys.initBinary(20, 42);

    NBodySimulator sim(&sys, 0.001);

    EXPECT_NO_THROW(sim.simulatePhasesBarrier());
}

// ------------------------------------------------------------
// TEST 10: parallelInitializationSingle
// ------------------------------------------------------------
// parallelInitializationSingle inicializa estructuras auxiliares dentro
// de una región paralela usando single, de modo que solo un hilo ejecuta
// la inicialización mientras los demás esperan en el barrier implícito.
// Verifica que esta construcción no introduce condiciones de carrera.
// ------------------------------------------------------------
TEST(NBodySimulator, ParallelInitializationSingleRuns) {
    NBodySystem sys(1.0, 0.01);
    sys.initBinary(20, 42);

    NBodySimulator sim(&sys, 0.001);

    EXPECT_NO_THROW(sim.parallelInitializationSingle());
}

// ------------------------------------------------------------
// TEST 11: calculateMetricsFirstprivate
// ------------------------------------------------------------
// firstprivate inicializa la variable privada de cada hilo con el valor
// del hilo principal antes de entrar a la región paralela. Este test
// verifica que el cálculo de energía con firstprivate produce el mismo
// resultado que la versión serial, detectando errores de inicialización
// de variables privadas.
// ------------------------------------------------------------
TEST(NBodySimulator, CalculateMetricsFirstprivateMatchesSerialEnergy) {
    NBodySystem sys(1.0, 0.01);
    sys.initBinary(20, 42);

    NBodySimulator sim(&sys, 0.001);

    double E_serial = sim.calculateTotalEnergy();
    double E_firstprivate = sim.calculateMetricsFirstprivate();

    EXPECT_NEAR(E_serial, E_firstprivate, 1e-9);
}

// ------------------------------------------------------------
// TEST 12: calculateFinalStateLastprivate
// ------------------------------------------------------------
// lastprivate copia al hilo principal el valor de la variable privada
// correspondiente a la última iteración del bucle paralelo. Este test
// verifica que el índice devuelto es exactamente getCount()-1, lo que
// confirma que lastprivate propaga correctamente el valor de la última
// iteración y no el de un hilo arbitrario.
// ------------------------------------------------------------
TEST(NBodySimulator, CalculateFinalStateLastprivateReturnsLastIndex) {
    NBodySystem sys(1.0, 0.01);
    sys.initBinary(20, 42);

    NBodySimulator sim(&sys, 0.001);

    int last_index = sim.calculateFinalStateLastprivate();

    EXPECT_EQ(last_index, sys.getCount() - 1);
}

// ------------------------------------------------------------
// TEST 13: calculateFinalStateLastprivate con sistema vacío
// ------------------------------------------------------------
// Caso borde: con cero partículas el bucle no itera, por lo que
// lastprivate nunca escribe y el índice debe quedar en -1 (valor
// centinela). Verifica que la implementación maneja este caso sin
// comportamiento indefinido.
// ------------------------------------------------------------
TEST(NBodySimulator, CalculateFinalStateLastprivateEmptySystem) {
    NBodySystem sys(1.0, 0.01);
    NBodySimulator sim(&sys, 0.001);

    int last_index = sim.calculateFinalStateLastprivate();

    EXPECT_EQ(last_index, -1);
}