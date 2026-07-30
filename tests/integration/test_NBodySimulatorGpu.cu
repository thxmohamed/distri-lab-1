#include "NBodySimulator.h"
#include "NBodySystem.h"
#include "Particle.h"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr double kGravitationalConstant = 1.0;
constexpr double kEpsilon = 1e-2;
constexpr double kTimeStep = 1e-3;
constexpr int kBlockSize = 256;
constexpr int kNumberOfSteps = 5;

constexpr double kRelativeTolerance = 1e-4;
constexpr double kAbsoluteTolerance = 1e-8;

// ----------------------------------------------------------------
// Funciones auxiliares
// ----------------------------------------------------------------

bool approximatelyEqual(
    double expected,
    double obtained
) {
    if (std::isnan(expected) || std::isnan(obtained)) {
        return false;
    }

    if (std::isinf(expected) || std::isinf(obtained)) {
        return expected == obtained;
    }

    return std::abs(expected - obtained) <=
        kAbsoluteTolerance +
        kRelativeTolerance * std::abs(expected);
}

bool compareValue(
    double expected,
    double obtained,
    const std::string& testName,
    const std::string& fieldName
) {
    if (!approximatelyEqual(expected, obtained)) {
        std::cerr
            << "[FAIL] " << testName
            << " | " << fieldName
            << " | esperado=" << expected
            << " | obtenido=" << obtained
            << " | diff=" << std::abs(expected - obtained)
            << '\n';

        return false;
    }

    return true;
}

NBodySystem createSmallSystem() {
    NBodySystem system(
        kGravitationalConstant,
        kEpsilon
    );

    system.addParticle(Particle(
        1.0,
        0.0, 0.0,
        0.10, 0.00
    ));

    system.addParticle(Particle(
        2.0,
        1.0, 0.0,
        0.00, 0.20
    ));

    system.addParticle(Particle(
        0.5,
        -0.5, 0.75,
        -0.10, 0.05
    ));

    return system;
}

bool compareSystems(
    const NBodySystem& expectedSystem,
    const NBodySystem& obtainedSystem,
    const std::string& testName
) {
    const auto& expectedBodies =
        expectedSystem.getBodies();

    const auto& obtainedBodies =
        obtainedSystem.getBodies();

    if (expectedBodies.size() != obtainedBodies.size()) {
        std::cerr
            << "[FAIL] " << testName
            << " | cantidad de cuerpos distinta"
            << " | esperado=" << expectedBodies.size()
            << " | obtenido=" << obtainedBodies.size()
            << '\n';

        return false;
    }

    bool passed = true;

    for (std::size_t i = 0;
         i < expectedBodies.size();
         ++i) {
        const std::string prefix =
            "body[" + std::to_string(i) + "]";

        passed &= compareValue(
            expectedBodies[i].getX(),
            obtainedBodies[i].getX(),
            testName,
            prefix + ".x"
        );

        passed &= compareValue(
            expectedBodies[i].getY(),
            obtainedBodies[i].getY(),
            testName,
            prefix + ".y"
        );

        passed &= compareValue(
            expectedBodies[i].getVx(),
            obtainedBodies[i].getVx(),
            testName,
            prefix + ".vx"
        );

        passed &= compareValue(
            expectedBodies[i].getVy(),
            obtainedBodies[i].getVy(),
            testName,
            prefix + ".vy"
        );

        passed &= compareValue(
            expectedBodies[i].getAx(),
            obtainedBodies[i].getAx(),
            testName,
            prefix + ".ax"
        );

        passed &= compareValue(
            expectedBodies[i].getAy(),
            obtainedBodies[i].getAy(),
            testName,
            prefix + ".ay"
        );
    }

    return passed;
}

bool compareEnergyComponents(
    double expectedKinetic,
    double expectedPotential,
    double expectedTotal,
    NBodySimulator& obtainedSimulator,
    double obtainedTotal,
    const std::string& testName
) {
    bool passed = true;

    passed &= compareValue(
        expectedKinetic,
        obtainedSimulator.getKineticEnergy(),
        testName,
        "K"
    );

    passed &= compareValue(
        expectedPotential,
        obtainedSimulator.getPotentialEnergy(),
        testName,
        "U"
    );

    passed &= compareValue(
        expectedTotal,
        obtainedTotal,
        testName,
        "E"
    );

    return passed;
}

template <typename Function>
bool expectInvalidArgument(
    Function function,
    const std::string& caseName
) {
    try {
        function();

        std::cerr
            << "[FAIL] " << caseName
            << " | no lanzo std::invalid_argument\n";

        return false;
    } catch (const std::invalid_argument&) {
        return true;
    } catch (const std::exception& exception) {
        std::cerr
            << "[FAIL] " << caseName
            << " | excepcion incorrecta: "
            << exception.what()
            << '\n';

        return false;
    } catch (...) {
        std::cerr
            << "[FAIL] " << caseName
            << " | excepcion desconocida\n";

        return false;
    }
}

using TestFunction = bool (*)();

bool runTest(
    const std::string& testName,
    TestFunction testFunction
) {
    try {
        const bool passed = testFunction();

        std::cout
            << (passed ? "[PASS] " : "[FAIL] ")
            << testName
            << '\n';

        return passed;
    } catch (const std::exception& exception) {
        std::cerr
            << "[FAIL] " << testName
            << " | excepcion inesperada: "
            << exception.what()
            << '\n';

        return false;
    } catch (...) {
        std::cerr
            << "[FAIL] " << testName
            << " | excepcion desconocida\n";

        return false;
    }
}

// ----------------------------------------------------------------
// Tests básicos de integración Euler
// ----------------------------------------------------------------

bool testEulerBasicMatchesCpu() {
    NBodySystem cpuSystem = createSmallSystem();
    NBodySystem gpuSystem = createSmallSystem();

    NBodySimulator cpuSimulator(
        &cpuSystem,
        kTimeStep
    );

    NBodySimulator gpuSimulator(
        &gpuSystem,
        kTimeStep
    );

    cpuSimulator.integrateEuler();

    gpuSimulator.stepEulerGpu(
        0,
        kBlockSize
    );

    bool passed = compareSystems(
        cpuSystem,
        gpuSystem,
        "Euler GPU basico vs CPU"
    );

    passed &= compareValue(
        cpuSimulator.getCurrentTime(),
        gpuSimulator.getCurrentTime(),
        "Euler GPU basico vs CPU",
        "current_time"
    );

    return passed;
}

bool testEulerSharedMatchesCpu() {
    NBodySystem cpuSystem = createSmallSystem();
    NBodySystem gpuSystem = createSmallSystem();

    NBodySimulator cpuSimulator(
        &cpuSystem,
        kTimeStep
    );

    NBodySimulator gpuSimulator(
        &gpuSystem,
        kTimeStep
    );

    cpuSimulator.integrateEuler();

    gpuSimulator.stepEulerGpu(
        1,
        kBlockSize
    );

    bool passed = compareSystems(
        cpuSystem,
        gpuSystem,
        "Euler GPU shared vs CPU"
    );

    passed &= compareValue(
        cpuSimulator.getCurrentTime(),
        gpuSimulator.getCurrentTime(),
        "Euler GPU shared vs CPU",
        "current_time"
    );

    return passed;
}

// ----------------------------------------------------------------
// Reutilización del estado durante múltiples pasos
// ----------------------------------------------------------------

bool testMultipleEulerStepsMatchCpu() {
    bool passed = true;

    for (int variant = 0;
         variant <= 1;
         ++variant) {
        NBodySystem cpuSystem = createSmallSystem();
        NBodySystem gpuSystem = createSmallSystem();

        NBodySimulator cpuSimulator(
            &cpuSystem,
            kTimeStep
        );

        NBodySimulator gpuSimulator(
            &gpuSystem,
            kTimeStep
        );

        for (int step = 0;
             step < kNumberOfSteps;
             ++step) {
            cpuSimulator.integrateEuler();

            gpuSimulator.stepEulerGpu(
                variant,
                kBlockSize
            );
        }

        const std::string variantName =
            variant == 0
                ? "kernel basico"
                : "kernel shared";

        const std::string testName =
            "Cinco pasos Euler " +
            variantName +
            " vs CPU";

        passed &= compareSystems(
            cpuSystem,
            gpuSystem,
            testName
        );

        passed &= compareValue(
            cpuSimulator.getCurrentTime(),
            gpuSimulator.getCurrentTime(),
            testName,
            "current_time"
        );
    }

    return passed;
}

// ----------------------------------------------------------------
// Validación de energía GPU
// ----------------------------------------------------------------

bool testEnergyGpuMatchesCpu() {
    NBodySystem cpuSystem = createSmallSystem();

    NBodySimulator cpuSimulator(
        &cpuSystem,
        kTimeStep
    );

    const double expectedTotal =
        cpuSimulator.calculateTotalEnergy();

    const double expectedKinetic =
        cpuSimulator.getKineticEnergy();

    const double expectedPotential =
        cpuSimulator.getPotentialEnergy();

    bool passed = true;

    // Método 0: reducción en shared memory.
    {
        NBodySystem gpuSystem = createSmallSystem();

        NBodySimulator gpuSimulator(
            &gpuSystem,
            kTimeStep
        );

        const double gpuTotal =
            gpuSimulator.calculateEnergyGpu(0);

        passed &= compareEnergyComponents(
            expectedKinetic,
            expectedPotential,
            expectedTotal,
            gpuSimulator,
            gpuTotal,
            "Energia GPU reduction"
        );
    }

    // Método 1: acumulación con atomicAdd.
    {
        NBodySystem gpuSystem = createSmallSystem();

        NBodySimulator gpuSimulator(
            &gpuSystem,
            kTimeStep
        );

        const double gpuTotal =
            gpuSimulator.calculateEnergyGpu(1);

        passed &= compareEnergyComponents(
            expectedKinetic,
            expectedPotential,
            expectedTotal,
            gpuSimulator,
            gpuTotal,
            "Energia GPU atomic"
        );
    }

    return passed;
}

// ----------------------------------------------------------------
// Estado persistente entre aceleraciones, Euler y energía
// ----------------------------------------------------------------

bool testPersistentStateAcrossGpuOperations() {
    NBodySystem cpuSystem = createSmallSystem();
    NBodySystem gpuSystem = createSmallSystem();

    NBodySimulator cpuSimulator(
        &cpuSystem,
        kTimeStep
    );

    NBodySimulator gpuSimulator(
        &gpuSystem,
        kTimeStep
    );

    bool passed = true;

    // 1. Primera operación GPU:
    // crea y carga el estado persistente.
    cpuSystem.computeAccelerationsSerial();

    gpuSystem.computeAccelerationsGpu(
        0,
        kBlockSize
    );

    passed &= compareSystems(
        cpuSystem,
        gpuSystem,
        "Estado inicial despues de aceleraciones"
    );

    // 2. Euler debe reutilizar el estado existente,
    // actualizar posiciones y mantener resultados equivalentes.
    cpuSimulator.integrateEuler();

    gpuSimulator.stepEulerGpu(
        1,
        kBlockSize
    );

    passed &= compareSystems(
        cpuSystem,
        gpuSystem,
        "Estado persistente despues de Euler"
    );

    passed &= compareValue(
        cpuSimulator.getCurrentTime(),
        gpuSimulator.getCurrentTime(),
        "Estado persistente despues de Euler",
        "current_time"
    );

    // 3. La energía debe reutilizar el mismo estado
    // y transferir las velocidades actuales.
    const double expectedTotal =
        cpuSimulator.calculateTotalEnergy();

    const double expectedKinetic =
        cpuSimulator.getKineticEnergy();

    const double expectedPotential =
        cpuSimulator.getPotentialEnergy();

    const double reductionTotal =
        gpuSimulator.calculateEnergyGpu(0);

    passed &= compareEnergyComponents(
        expectedKinetic,
        expectedPotential,
        expectedTotal,
        gpuSimulator,
        reductionTotal,
        "Energia reduction despues de Euler"
    );

    // 4. Cambiar de reduction a atomic debe reutilizar
    // nuevamente el mismo estado GPU.
    const double atomicTotal =
        gpuSimulator.calculateEnergyGpu(1);

    passed &= compareEnergyComponents(
        expectedKinetic,
        expectedPotential,
        expectedTotal,
        gpuSimulator,
        atomicTotal,
        "Energia atomic despues de Euler"
    );

    // 5. Calcular energía no debe dejar posiciones,
    // velocidades o buffers en un estado inconsistente.
    cpuSystem.computeAccelerationsSerial();

    gpuSystem.computeAccelerationsGpu(
        0,
        kBlockSize
    );

    passed &= compareSystems(
        cpuSystem,
        gpuSystem,
        "Aceleraciones despues de calcular energia"
    );

    return passed;
}

// ----------------------------------------------------------------
// Invalidación y redimensionamiento al agregar partículas
// ----------------------------------------------------------------

bool testPersistentStateResizesAfterAddingParticle() {
    NBodySystem cpuSystem = createSmallSystem();
    NBodySystem gpuSystem = createSmallSystem();

    bool passed = true;

    // Inicializa el estado GPU con tres cuerpos.
    cpuSystem.computeAccelerationsSerial();

    gpuSystem.computeAccelerationsGpu(
        0,
        kBlockSize
    );

    passed &= compareSystems(
        cpuSystem,
        gpuSystem,
        "Estado antes de agregar particula"
    );

    // Agregar una partícula debe invalidar la carga completa
    // y redimensionar los buffers en la siguiente operación GPU.
    const Particle additionalParticle(
        1.25,
        2.0, -1.0,
        0.15, -0.05
    );

    cpuSystem.addParticle(additionalParticle);
    gpuSystem.addParticle(additionalParticle);

    cpuSystem.computeAccelerationsSerial();

    gpuSystem.computeAccelerationsGpu(
        1,
        kBlockSize
    );

    passed &= compareSystems(
        cpuSystem,
        gpuSystem,
        "Estado despues de agregar particula"
    );

    // También se verifica que el estado redimensionado
    // pueda reutilizarse para calcular energía.
    NBodySimulator cpuSimulator(
        &cpuSystem,
        kTimeStep
    );

    NBodySimulator gpuSimulator(
        &gpuSystem,
        kTimeStep
    );

    const double expectedTotal =
        cpuSimulator.calculateTotalEnergy();

    const double expectedKinetic =
        cpuSimulator.getKineticEnergy();

    const double expectedPotential =
        cpuSimulator.getPotentialEnergy();

    const double reductionTotal =
        gpuSimulator.calculateEnergyGpu(0);

    passed &= compareEnergyComponents(
        expectedKinetic,
        expectedPotential,
        expectedTotal,
        gpuSimulator,
        reductionTotal,
        "Energia despues de redimensionar estado GPU"
    );

    return passed;
}

// ----------------------------------------------------------------
// Sobrecargas obligatorias sin parámetros
// ----------------------------------------------------------------

bool testDefaultGpuOverloads() {
    bool passed = true;

    // computeAccelerationsGpu()
    {
        NBodySystem cpuSystem = createSmallSystem();
        NBodySystem gpuSystem = createSmallSystem();

        cpuSystem.computeAccelerationsSerial();
        gpuSystem.computeAccelerationsGpu();

        passed &= compareSystems(
            cpuSystem,
            gpuSystem,
            "Sobrecarga computeAccelerationsGpu()"
        );
    }

    // stepEulerGpu()
    {
        NBodySystem cpuSystem = createSmallSystem();
        NBodySystem gpuSystem = createSmallSystem();

        NBodySimulator cpuSimulator(
            &cpuSystem,
            kTimeStep
        );

        NBodySimulator gpuSimulator(
            &gpuSystem,
            kTimeStep
        );

        cpuSimulator.integrateEuler();
        gpuSimulator.stepEulerGpu();

        passed &= compareSystems(
            cpuSystem,
            gpuSystem,
            "Sobrecarga stepEulerGpu()"
        );

        passed &= compareValue(
            cpuSimulator.getCurrentTime(),
            gpuSimulator.getCurrentTime(),
            "Sobrecarga stepEulerGpu()",
            "current_time"
        );
    }

    // calculateEnergyGpu()
    {
        NBodySystem cpuSystem = createSmallSystem();
        NBodySystem gpuSystem = createSmallSystem();

        NBodySimulator cpuSimulator(
            &cpuSystem,
            kTimeStep
        );

        NBodySimulator gpuSimulator(
            &gpuSystem,
            kTimeStep
        );

        const double expectedTotal =
            cpuSimulator.calculateTotalEnergy();

        const double expectedKinetic =
            cpuSimulator.getKineticEnergy();

        const double expectedPotential =
            cpuSimulator.getPotentialEnergy();

        const double gpuTotal =
            gpuSimulator.calculateEnergyGpu();

        passed &= compareEnergyComponents(
            expectedKinetic,
            expectedPotential,
            expectedTotal,
            gpuSimulator,
            gpuTotal,
            "Sobrecarga calculateEnergyGpu()"
        );
    }

    return passed;
}

// ----------------------------------------------------------------
// Parámetros inválidos
// ----------------------------------------------------------------

bool testInvalidGpuEulerParameters() {
    NBodySystem system = createSmallSystem();

    NBodySimulator simulator(
        &system,
        kTimeStep
    );

    bool passed = true;

    passed &= expectInvalidArgument(
        [&simulator]() {
            simulator.stepEulerGpu(
                -1,
                kBlockSize
            );
        },
        "stepEulerGpu rechaza variant=-1"
    );

    passed &= expectInvalidArgument(
        [&simulator]() {
            simulator.stepEulerGpu(
                2,
                kBlockSize
            );
        },
        "stepEulerGpu rechaza variant=2"
    );

    passed &= expectInvalidArgument(
        [&simulator]() {
            simulator.stepEulerGpu(
                0,
                0
            );
        },
        "stepEulerGpu rechaza block_size=0"
    );

    passed &= expectInvalidArgument(
        [&simulator]() {
            simulator.stepEulerGpu(
                0,
                -1
            );
        },
        "stepEulerGpu rechaza block_size=-1"
    );

    return passed;
}

bool testInvalidGpuEnergyParameters() {
    NBodySystem system = createSmallSystem();

    NBodySimulator simulator(
        &system,
        kTimeStep
    );

    bool passed = true;

    passed &= expectInvalidArgument(
        [&simulator]() {
            simulator.calculateEnergyGpu(-1);
        },
        "calculateEnergyGpu rechaza method=-1"
    );

    passed &= expectInvalidArgument(
        [&simulator]() {
            simulator.calculateEnergyGpu(2);
        },
        "calculateEnergyGpu rechaza method=2"
    );

    return passed;
}

} // namespace

int main() {
    std::cout
        << "=== Tests NBodySimulator GPU ===\n"
        << "rtol=" << kRelativeTolerance
        << ", atol=" << kAbsoluteTolerance
        << ", dt=" << kTimeStep
        << ", epsilon=" << kEpsilon
        << ", block_size=" << kBlockSize
        << '\n';

    bool passed = true;

    passed &= runTest(
        "Euler GPU basico vs CPU",
        testEulerBasicMatchesCpu
    );

    passed &= runTest(
        "Euler GPU shared vs CPU",
        testEulerSharedMatchesCpu
    );

    passed &= runTest(
        "Multiples pasos Euler con ambos kernels",
        testMultipleEulerStepsMatchCpu
    );

    passed &= runTest(
        "Energia GPU reduction y atomic vs CPU",
        testEnergyGpuMatchesCpu
    );

    passed &= runTest(
        "Estado persistente entre operaciones GPU",
        testPersistentStateAcrossGpuOperations
    );

    passed &= runTest(
        "Redimensionamiento tras agregar particula",
        testPersistentStateResizesAfterAddingParticle
    );

    passed &= runTest(
        "Sobrecargas GPU por defecto",
        testDefaultGpuOverloads
    );

    passed &= runTest(
        "Parametros invalidos de Euler GPU",
        testInvalidGpuEulerParameters
    );

    passed &= runTest(
        "Parametros invalidos de energia GPU",
        testInvalidGpuEnergyParameters
    );

    std::cout
        << "\nResultado NBodySimulator GPU: "
        << (passed ? "PASS" : "FAIL")
        << '\n';

    return passed ? 0 : 1;
}
