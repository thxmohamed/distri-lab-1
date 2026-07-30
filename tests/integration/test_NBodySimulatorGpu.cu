#include "NBodySimulator.h"
#include "NBodySystem.h"
#include "Particle.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr double kGravitationalConstant = 1.0;
constexpr double kEpsilon = 1e-2;
constexpr double kTimeStep = 1e-3;
constexpr int kBlockSize = 256;
constexpr double kRelativeTolerance = 1e-4;
constexpr double kAbsoluteTolerance = 1e-8;

// ----------------------------------------------------------------
// Funciones auxiliares
// ----------------------------------------------------------------

bool approximatelyEqual(double expected, double obtained) {
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
    NBodySystem system(kGravitationalConstant, kEpsilon);

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
    const auto& expectedBodies = expectedSystem.getBodies();
    const auto& obtainedBodies = obtainedSystem.getBodies();

    if (expectedBodies.size() != obtainedBodies.size()) {
        std::cerr
            << "[FAIL] " << testName
            << " | cantidad de cuerpos distinta\n";

        return false;
    }

    bool passed = true;

    for (std::size_t i = 0; i < expectedBodies.size(); ++i) {
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

    if (passed) {
        std::cout << "[PASS] " << testName << '\n';
    }

    return passed;
}

// ----------------------------------------------------------------
// Tests
// ----------------------------------------------------------------

bool testEulerBasicMatchesCpu() {
    NBodySystem cpuSystem = createSmallSystem();
    NBodySystem gpuSystem = createSmallSystem();

    NBodySimulator cpuSimulator(&cpuSystem, kTimeStep);
    NBodySimulator gpuSimulator(&gpuSystem, kTimeStep);

    cpuSimulator.integrateEuler();
    gpuSimulator.stepEulerGpu(0, kBlockSize);

    return compareSystems(
        cpuSystem,
        gpuSystem,
        "Euler GPU basico vs CPU"
    );
}

bool testEulerSharedMatchesCpu() {
    NBodySystem cpuSystem = createSmallSystem();
    NBodySystem gpuSystem = createSmallSystem();

    NBodySimulator cpuSimulator(&cpuSystem, kTimeStep);
    NBodySimulator gpuSimulator(&gpuSystem, kTimeStep);

    cpuSimulator.integrateEuler();
    gpuSimulator.stepEulerGpu(1, kBlockSize);

    return compareSystems(
        cpuSystem,
        gpuSystem,
        "Euler GPU shared vs CPU"
    );
}

bool testMultipleEulerStepsMatchCpu() {
    NBodySystem cpuSystem = createSmallSystem();
    NBodySystem gpuSystem = createSmallSystem();

    NBodySimulator cpuSimulator(&cpuSystem, kTimeStep);
    NBodySimulator gpuSimulator(&gpuSystem, kTimeStep);

    for (int step = 0; step < 5; ++step) {
        cpuSimulator.integrateEuler();
        gpuSimulator.stepEulerGpu(1, kBlockSize);
    }

    bool passed = compareSystems(
        cpuSystem,
        gpuSystem,
        "Cinco pasos Euler GPU shared vs CPU"
    );

    passed &= compareValue(
        cpuSimulator.getCurrentTime(),
        gpuSimulator.getCurrentTime(),
        "Tiempo acumulado GPU vs CPU",
        "current_time"
    );

    if (passed) {
        std::cout << "[PASS] Tiempo acumulado GPU vs CPU\n";
    }

    return passed;
}

bool testEnergyGpuMatchesCpu() {
    NBodySystem system = createSmallSystem();
    NBodySimulator simulator(&system, kTimeStep);

    const double cpuEnergy =
        simulator.calculateTotalEnergy();

    const double gpuReductionEnergy =
        simulator.calculateEnergyGpu(0);

    const double gpuAtomicEnergy =
        simulator.calculateEnergyGpu(1);

    bool passed = true;

    passed &= compareValue(
        cpuEnergy,
        gpuReductionEnergy,
        "Energia GPU reduction vs CPU",
        "E"
    );

    passed &= compareValue(
        cpuEnergy,
        gpuAtomicEnergy,
        "Energia GPU atomic vs CPU",
        "E"
    );

    if (passed) {
        std::cout
            << "[PASS] Energia GPU reduction/atomic vs CPU\n";
    }

    return passed;
}

bool testInvalidGpuEulerParameters() {
    NBodySystem system = createSmallSystem();
    NBodySimulator simulator(&system, kTimeStep);

    bool passed = true;

    try {
        simulator.stepEulerGpu(-1, kBlockSize);
        std::cerr
            << "[FAIL] stepEulerGpu no rechazo variant invalido\n";
        passed = false;
    } catch (const std::invalid_argument&) {
        std::cout
            << "[PASS] stepEulerGpu rechaza variant invalido\n";
    }

    try {
        simulator.stepEulerGpu(0, 0);
        std::cerr
            << "[FAIL] stepEulerGpu no rechazo block_size invalido\n";
        passed = false;
    } catch (const std::invalid_argument&) {
        std::cout
            << "[PASS] stepEulerGpu rechaza block_size invalido\n";
    }

    return passed;
}

bool testInvalidGpuEnergyParameters() {
    NBodySystem system = createSmallSystem();
    NBodySimulator simulator(&system, kTimeStep);

    bool passed = true;

    try {
        simulator.calculateEnergyGpu(2);
        std::cerr
            << "[FAIL] calculateEnergyGpu no rechazo method invalido\n";
        passed = false;
    } catch (const std::invalid_argument&) {
        std::cout
            << "[PASS] calculateEnergyGpu rechaza method invalido\n";
    }

    return passed;
}

}

int main() {
    std::cout
        << "=== Tests NBodySimulator GPU ===\n"
        << "rtol=1e-4" << kRelativeTolerance
        << ", atol=1e-8" << kAbsoluteTolerance
        << ", dt=" << kTimeStep
        << ", epsilon=" << kEpsilon
        << '\n';

    bool passed = true;

    passed &= testEulerBasicMatchesCpu();
    passed &= testEulerSharedMatchesCpu();
    passed &= testMultipleEulerStepsMatchCpu();
    passed &= testEnergyGpuMatchesCpu();
    passed &= testInvalidGpuEulerParameters();
    passed &= testInvalidGpuEnergyParameters();

    std::cout
        << "\nResultado NBodySimulator GPU: "
        << (passed ? "PASS" : "FAIL")
        << '\n';

    return passed ? 0 : 1;
}
