#include "model/NBodySystem.h"
#include "model/Particle.h"

#include <iostream>
#include <cmath>
#include <iomanip>

// ============================================================
// Helpers
// ============================================================

// Comparación con tolerancia (por errores de punto flotante)
static bool nearlyEqual(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}

// Imprime aceleraciones de una partícula
void printParticleState(const Particle& p, int i) {
    std::cout << "  Body[" << i << "] -> "
              << "ax=" << std::setw(12) << p.getAx()
              << ", ay=" << std::setw(12) << p.getAy() << "\n";
}

// ============================================================
// TEST 1: 2 cuerpos (caso base simple)
// ============================================================

bool test_two_body() {
    std::cout << "\n=== TEST: Two Body ===\n";

    NBodySystem sys(1.0, 0.01);

    // Dos partículas simétricas
    sys.addParticle(Particle(1.0, -1.0, 0.0, 0.0, 0.0));
    sys.addParticle(Particle(1.0,  1.0, 0.0, 0.0, 0.0));

    std::cout << "Sistema inicializado con 2 partículas\n";

    // ---------------- SERIAL ----------------
    std::cout << "\n[Serial]\n";
    sys.computeAccelerations();
    auto serial = sys.getBodies();

    for (int i = 0; i < sys.getCount(); ++i)
        printParticleState(serial[i], i);

    // ---------------- PARALELO ----------------
    std::cout << "\n[Parallel Simple]\n";
    sys.computeAccelerationsParallelSimple();
    auto parallel = sys.getBodies();

    for (int i = 0; i < sys.getCount(); ++i)
        printParticleState(parallel[i], i);

    // ---------------- COMPARACIÓN ----------------
    bool ok = true;

    for (int i = 0; i < sys.getCount(); ++i) {
        if (!nearlyEqual(serial[i].getAx(), parallel[i].getAx()) ||
            !nearlyEqual(serial[i].getAy(), parallel[i].getAy())) {

            std::cout << "Mismatch en cuerpo " << i << "\n";
            ok = false;
        }
    }

    std::cout << (ok ? "[OK]" : "[FAIL]") << " test_two_body\n";
    return ok;
}

// ============================================================
// TEST 2: 3 cuerpos (verificable a mano)
// ============================================================

bool test_three_body() {
    std::cout << "\n=== TEST: Three Body ===\n";

    NBodySystem sys(1.0, 0.01);

    // Configuración simple:
    //  (-1,0)   (0,0)   (1,0)
    sys.addParticle(Particle(1.0, -1.0, 0.0, 0.0, 0.0));
    sys.addParticle(Particle(1.0,  0.0, 0.0, 0.0, 0.0));
    sys.addParticle(Particle(1.0,  1.0, 0.0, 0.0, 0.0));

    std::cout << "Sistema inicializado con 3 partículas\n";

    // ---------------- SERIAL ----------------
    std::cout << "\n[Serial]\n";
    sys.computeAccelerations();
    auto serial = sys.getBodies();

    for (int i = 0; i < sys.getCount(); ++i)
        printParticleState(serial[i], i);

    // ---------------- PARALELO ----------------
    std::cout << "\n[Parallel Simple]\n";
    sys.computeAccelerationsParallelSimple();
    auto parallel = sys.getBodies();

    for (int i = 0; i < sys.getCount(); ++i)
        printParticleState(parallel[i], i);

    // ---------------- COMPARACIÓN ----------------
    bool ok = true;

    for (int i = 0; i < sys.getCount(); ++i) {
        if (!nearlyEqual(serial[i].getAx(), parallel[i].getAx(), 1e-9) ||
            !nearlyEqual(serial[i].getAy(), parallel[i].getAy(), 1e-9)) {

            std::cout << "Mismatch en cuerpo " << i << "\n";

            std::cout << "  serial:   ax=" << serial[i].getAx()
                      << " ay=" << serial[i].getAy() << "\n";

            std::cout << "  parallel: ax=" << parallel[i].getAx()
                      << " ay=" << parallel[i].getAy() << "\n";

            ok = false;
        }
    }

    std::cout << (ok ? "[OK]" : "[FAIL]") << " test_three_body\n";
    return ok;
}

// ============================================================
// TEST 3: Schedules OpenMP
// ============================================================

bool test_schedules() {
    std::cout << "\n=== TEST: Schedules ===\n";

    NBodySystem sys(1.0, 0.01);

    // Sistema más grande
    sys.initBinary(10, 42);

    std::cout << "Sistema inicializado (initBinary, N=10)\n";

    // Referencia serial
    sys.computeAccelerations();
    auto ref = sys.getBodies();

    std::cout << "\nReferencia (Serial) calculada\n";

    bool ok = true;

    // Probamos modos 2–5:
    // static, dynamic, guided, collapse
    for (int mode = 2; mode <= 5; ++mode) {
        std::cout << "\n[Modo " << mode << "]\n";

        sys.computeAccelerationsMode(mode);
        auto test = sys.getBodies();

        for (int i = 0; i < sys.getCount(); ++i) {
            if (!nearlyEqual(ref[i].getAx(), test[i].getAx(), 1e-6) ||
                !nearlyEqual(ref[i].getAy(), test[i].getAy(), 1e-6)) {

                std::cout << "Mismatch en body " << i << "\n";

                std::cout << "  ref:  ax=" << ref[i].getAx()
                          << " ay=" << ref[i].getAy() << "\n";

                std::cout << "  test: ax=" << test[i].getAx()
                          << " ay=" << test[i].getAy() << "\n";

                ok = false;
                break;
            }
        }

        if (ok)
            std::cout << "Modo " << mode << " OK\n";
        else
            break;
    }

    std::cout << (ok ? "[OK]" : "[FAIL]") << " test_schedules\n";
    return ok;
}

// ============================================================
// MAIN
// ============================================================

int main() {
    std::cout << "========== NBodySystem TEST ==========\n";

    bool ok = true;

    ok &= test_two_body();     // caso base
    ok &= test_three_body();   // verificable a mano
    ok &= test_schedules();    // OpenMP

    std::cout << "\n=====================================\n";

    if (ok) {
        std::cout << "ALL TESTS PASSED\n";
        return 0;
    } else {
        std::cout << "TESTS FAILED\n";
        return 1;
    }
}