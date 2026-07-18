#include "accelerations.cuh"
#include "CudaCheck.cuh"
#include "NBodySystem.h"
#include "Particle.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kGravitationalConstant = 1.0;
constexpr double kEpsilon = 1e-2;
constexpr double kRelativeTolerance = 1e-4;
constexpr double kAbsoluteTolerance = 1e-8;

struct Accelerations {
    std::vector<double> ax;
    std::vector<double> ay;
};

struct TestCase {
    std::string name;
    std::vector<double> mass;
    std::vector<double> x;
    std::vector<double> y;
    int blockSize;
    bool verifyTwoBodyAnalytic;
};

void validateTestCase(const TestCase& testCase) {
    const std::size_t n = testCase.mass.size();

    if (n == 0U) {
        throw std::invalid_argument(
            "El caso de prueba debe contener al menos un cuerpo.");
    }

    if (testCase.x.size() != n || testCase.y.size() != n) {
        throw std::invalid_argument(
            "mass, x e y deben tener el mismo numero de elementos.");
    }

    if (testCase.blockSize <= 0) {
        throw std::invalid_argument(
            "El tamano de bloque del caso de prueba debe ser positivo.");
    }

    if (testCase.verifyTwoBodyAnalytic && n != 2U) {
        throw std::invalid_argument(
            "La validacion analitica de dos cuerpos requiere N = 2.");
    }
}

Accelerations computeCpuSerialReference(const TestCase& testCase) {
    NBodySystem system(kGravitationalConstant, kEpsilon);

    for (std::size_t i = 0; i < testCase.mass.size(); ++i) {
        system.addParticle(Particle(
            testCase.mass[i],
            testCase.x[i],
            testCase.y[i]
        ));
    }

    // La implementacion serial del Lab 1 es el baseline de correccion.
    system.computeAccelerationsSerial();

    Accelerations result;
    result.ax.reserve(testCase.mass.size());
    result.ay.reserve(testCase.mass.size());

    for (const Particle& body : system.getBodies()) {
        result.ax.push_back(body.getAx());
        result.ay.push_back(body.getAy());
    }

    return result;
}

Accelerations computeTwoBodyAnalytic(const TestCase& testCase) {
    const double dx01 = testCase.x[1] - testCase.x[0];
    const double dy01 = testCase.y[1] - testCase.y[0];
    const double epsilonSquared = kEpsilon * kEpsilon;
    const double distanceSquared =
        dx01 * dx01 + dy01 * dy01 + epsilonSquared;
    const double distanceCubed =
        distanceSquared * std::sqrt(distanceSquared);

    Accelerations result;
    result.ax.resize(2);
    result.ay.resize(2);

    // Aceleracion del cuerpo 0 causada por el cuerpo 1.
    result.ax[0] =
        kGravitationalConstant * testCase.mass[1] * dx01 /
        distanceCubed;
    result.ay[0] =
        kGravitationalConstant * testCase.mass[1] * dy01 /
        distanceCubed;

    // Aceleracion del cuerpo 1 causada por el cuerpo 0.
    result.ax[1] =
        kGravitationalConstant * testCase.mass[0] * (-dx01) /
        distanceCubed;
    result.ay[1] =
        kGravitationalConstant * testCase.mass[0] * (-dy01) /
        distanceCubed;

    return result;
}

Accelerations runGpuVariant(
    const TestCase& testCase,
    AccelerationKernelVariant variant
) {
    const int n = static_cast<int>(testCase.mass.size());
    const std::size_t bytes =
        static_cast<std::size_t>(n) * sizeof(double);

    double* dMass = nullptr;
    double* dX = nullptr;
    double* dY = nullptr;
    double* dAx = nullptr;
    double* dAy = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dMass), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dX), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dY), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dAx), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dAy), bytes));

    CUDA_CHECK(cudaMemcpy(
        dMass,
        testCase.mass.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        dX,
        testCase.x.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        dY,
        testCase.y.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    // Si un kernel no escribe algun resultado, el NaN hace fallar la prueba.
    const std::vector<double> initialOutput(
        testCase.mass.size(),
        std::numeric_limits<double>::quiet_NaN()
    );
    CUDA_CHECK(cudaMemcpy(
        dAx,
        initialOutput.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        dAy,
        initialOutput.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    launchComputeAccelerations(
        dMass,
        dX,
        dY,
        dAx,
        dAy,
        n,
        kGravitationalConstant,
        kEpsilon,
        variant,
        testCase.blockSize
    );

    // El launcher es asincrono. Esta sincronizacion detecta errores
    // de ejecucion y garantiza que los resultados ya esten disponibles.
    CUDA_CHECK(cudaDeviceSynchronize());

    Accelerations result{
        std::vector<double>(testCase.mass.size()),
        std::vector<double>(testCase.mass.size())
    };

    CUDA_CHECK(cudaMemcpy(
        result.ax.data(),
        dAx,
        bytes,
        cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaMemcpy(
        result.ay.data(),
        dAy,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaFree(dMass));
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dY));
    CUDA_CHECK(cudaFree(dAx));
    CUDA_CHECK(cudaFree(dAy));

    return result;
}

bool compareVectors(
    const std::string& label,
    const std::vector<double>& expected,
    const std::vector<double>& obtained
) {
    if (expected.size() != obtained.size()) {
        std::cout
            << "  [FAIL] " << label
            << ": tamanos distintos (esperado=" << expected.size()
            << ", obtenido=" << obtained.size() << ")\n";
        return false;
    }

    bool passed = true;
    std::size_t worstIndex = 0;
    double worstRatio = -1.0;
    double worstDifference = 0.0;
    double worstAllowed = 0.0;

    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double expectedValue = expected[i];
        const double obtainedValue = obtained[i];

        double ratio = std::numeric_limits<double>::infinity();
        double difference = std::numeric_limits<double>::infinity();
        double allowed = kAbsoluteTolerance;

        if (std::isfinite(expectedValue) && std::isfinite(obtainedValue)) {
            difference = std::abs(expectedValue - obtainedValue);
            allowed =
                kAbsoluteTolerance +
                kRelativeTolerance * std::abs(expectedValue);
            ratio = difference / allowed;
        }

        if (ratio > worstRatio) {
            worstRatio = ratio;
            worstIndex = i;
            worstDifference = difference;
            worstAllowed = allowed;
        }

        if (!std::isfinite(obtainedValue) || difference > allowed) {
            passed = false;
        }
    }

    std::cout
        << "  [" << (passed ? "PASS" : "FAIL") << "] "
        << label
        << " | error max=" << worstDifference
        << " | tolerancia=" << worstAllowed
        << " | indice=" << worstIndex;

    if (!passed) {
        std::cout
            << " | esperado=" << expected[worstIndex]
            << " | obtenido=" << obtained[worstIndex];
    }

    std::cout << '\n';
    return passed;
}

bool runTestCase(const TestCase& testCase) {
    validateTestCase(testCase);

    std::cout
        << "\n============================================================\n"
        << testCase.name
        << " | N=" << testCase.mass.size()
        << " | blockSize=" << testCase.blockSize
        << "\n============================================================\n";

    const Accelerations cpu = computeCpuSerialReference(testCase);
    const Accelerations basic = runGpuVariant(
        testCase,
        AccelerationKernelVariant::Basic
    );
    const Accelerations shared = runGpuVariant(
        testCase,
        AccelerationKernelVariant::Shared
    );

    bool passed = true;

    // Cada variante debe coincidir con el baseline CPU serial.
    passed &= compareVectors("Basic ax vs CPU", cpu.ax, basic.ax);
    passed &= compareVectors("Basic ay vs CPU", cpu.ay, basic.ay);
    passed &= compareVectors("Shared ax vs CPU", cpu.ax, shared.ax);
    passed &= compareVectors("Shared ay vs CPU", cpu.ay, shared.ay);

    // El criterio de aceptacion tambien exige equivalencia directa
    // entre la variante basica y la variante shared.
    passed &= compareVectors("Shared ax vs Basic", basic.ax, shared.ax);
    passed &= compareVectors("Shared ay vs Basic", basic.ay, shared.ay);

    if (testCase.verifyTwoBodyAnalytic) {
        const Accelerations analytic = computeTwoBodyAnalytic(testCase);

        passed &= compareVectors("CPU ax vs analitico", analytic.ax, cpu.ax);
        passed &= compareVectors("CPU ay vs analitico", analytic.ay, cpu.ay);
        passed &= compareVectors(
            "Basic ax vs analitico", analytic.ax, basic.ax);
        passed &= compareVectors(
            "Basic ay vs analitico", analytic.ay, basic.ay);
        passed &= compareVectors(
            "Shared ax vs analitico", analytic.ax, shared.ax);
        passed &= compareVectors(
            "Shared ay vs analitico", analytic.ay, shared.ay);
    }

    std::cout
        << "Resultado del caso: "
        << (passed ? "PASS" : "FAIL")
        << '\n';

    return passed;
}

} // namespace

int main() {
    std::cout << std::setprecision(12);
    std::cout
        << "Validacion de kernels CUDA de aceleraciones\n"
        << "rtol=" << kRelativeTolerance
        << ", atol=" << kAbsoluteTolerance
        << ", G=" << kGravitationalConstant
        << ", epsilon=" << kEpsilon
        << "\n";

    const std::vector<TestCase> testCases = {
        {
            "N=1: sin autointeraccion",
            {2.0},
            {1.5},
            {-0.5},
            4,
            false
        },
        {
            "N=2: validacion analitica",
            {1.0, 2.0},
            {0.0, 1.0},
            {0.0, 0.0},
            4,
            true
        },
        {
            "N=3: acumulacion de interacciones en 2D",
            {1.0, 2.0, 0.5},
            {0.0, 1.0, -0.5},
            {0.0, 0.25, 0.75},
            2,
            false
        },
        {
            "N=5: N no multiplo de blockSize",
            {1.0, 2.0, 3.0, 4.0, 5.0},
            {0.0, 1.0, -1.0, 0.5, -0.75},
            {0.0, 0.5, 1.0, -1.0, 0.25},
            4,
            false
        }
    };

    bool allPassed = true;

    try {
        for (const TestCase& testCase : testCases) {
            allPassed &= runTestCase(testCase);
        }
    } catch (const std::exception& error) {
        std::cerr << "Error al ejecutar las pruebas: "
                  << error.what() << '\n';
        return 1;
    }

    std::cout
        << "\n============================================================\n"
        << (allPassed
                ? "TODAS LAS PRUEBAS CUDA PASARON."
                : "UNA O MAS PRUEBAS CUDA FALLARON.")
        << "\n============================================================\n";

    return allPassed ? 0 : 1;
}
