#include "CudaCheck.cuh"
#include "NBodyDeviceState.h"
#include "Particle.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kTolerance = 1e-12;

// -----------------------------------------------------------------------------
// Funciones auxiliares
// -----------------------------------------------------------------------------

bool approximatelyEqual(double expected, double obtained) {
    return std::abs(expected - obtained) <= kTolerance;
}

bool compareVectors(
    const std::vector<double>& expected,
    const std::vector<double>& obtained,
    const std::string& testName
) {
    if (expected.size() != obtained.size()) {
        std::cerr
            << "[FAIL] " << testName
            << ": tamaños diferentes. Esperado=" << expected.size()
            << ", obtenido=" << obtained.size()
            << '\n';

        return false;
    }

    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (!approximatelyEqual(expected[i], obtained[i])) {
            std::cerr
                << "[FAIL] " << testName
                << ": diferencia en índice " << i
                << ". Esperado=" << expected[i]
                << ", obtenido=" << obtained[i]
                << '\n';

            return false;
        }
    }

    std::cout << "[PASS] " << testName << '\n';
    return true;
}

/**
 * Copia un arreglo device hacia un vector host.
 *
 * Esta función se usa solo para inspeccionar el estado interno durante el test.
 */
std::vector<double> copyDeviceToHost(
    const double* deviceData,
    std::size_t count
) {
    std::vector<double> hostData(count, 0.0);

    if (count > 0) {
        CUDA_CHECK(cudaMemcpy(
            hostData.data(),
            deviceData,
            count * sizeof(double),
            cudaMemcpyDeviceToHost
        ));
    }

    return hostData;
}

/**
 * Copia valores conocidos directamente hacia un arreglo device.
 *
 * En este test reemplaza temporalmente al kernel: permite comprobar
 * downloadAccelerations() sin depender del código del Rol 1.
 */
void copyHostToDevice(
    double* deviceData,
    const std::vector<double>& hostData
) {
    if (!hostData.empty()) {
        CUDA_CHECK(cudaMemcpy(
            deviceData,
            hostData.data(),
            hostData.size() * sizeof(double),
            cudaMemcpyHostToDevice
        ));
    }
}

std::vector<Particle> createTestBodies() {
    std::vector<Particle> bodies;

    bodies.emplace_back(
        1.5,        // masa
        -1.0, 2.0, // posición
        0.1, -0.2  // velocidad
    );

    bodies.emplace_back(
        2.0,
        3.5, -4.0,
        -0.3, 0.4
    );

    bodies.emplace_back(
        0.75,
        5.0, 6.5,
        0.8, -0.9
    );

    return bodies;
}

// -----------------------------------------------------------------------------
// Test 1: carga inicial
// -----------------------------------------------------------------------------

bool testInitialStateUpload() {
    std::vector<Particle> bodies = createTestBodies();

    NBodyDeviceState deviceState;
    deviceState.uploadInitialState(bodies);

    bool passed = true;

    if (!deviceState.isInitialized()) {
        std::cerr
            << "[FAIL] Estado marcado como no inicializado\n";

        passed = false;
    } else {
        std::cout
            << "[PASS] Estado marcado como inicializado\n";
    }

    if (deviceState.size() != bodies.size()) {
        std::cerr
            << "[FAIL] Tamaño del estado device incorrecto\n";

        passed = false;
    } else {
        std::cout
            << "[PASS] Tamaño del estado device correcto\n";
    }

    passed &= compareVectors(
        {1.5, 2.0, 0.75},
        copyDeviceToHost(
            deviceState.massData(),
            deviceState.size()
        ),
        "Carga inicial de masas"
    );

    passed &= compareVectors(
        {-1.0, 3.5, 5.0},
        copyDeviceToHost(
            deviceState.positionXData(),
            deviceState.size()
        ),
        "Carga inicial de posiciones X"
    );

    passed &= compareVectors(
        {2.0, -4.0, 6.5},
        copyDeviceToHost(
            deviceState.positionYData(),
            deviceState.size()
        ),
        "Carga inicial de posiciones Y"
    );

    passed &= compareVectors(
        {0.1, -0.3, 0.8},
        copyDeviceToHost(
            deviceState.velocityXData(),
            deviceState.size()
        ),
        "Carga inicial de velocidades X"
    );

    passed &= compareVectors(
        {-0.2, 0.4, -0.9},
        copyDeviceToHost(
            deviceState.velocityYData(),
            deviceState.size()
        ),
        "Carga inicial de velocidades Y"
    );

    return passed;
}

// -----------------------------------------------------------------------------
// Test 2: transferencias selectivas
// -----------------------------------------------------------------------------

bool testSelectiveUpdates() {
    std::vector<Particle> bodies = createTestBodies();

    NBodyDeviceState deviceState;
    deviceState.uploadInitialState(bodies);

    const std::vector<double> originalMass{
        1.5,
        2.0,
        0.75
    };

    const std::vector<double> originalVx{
        0.1,
        -0.3,
        0.8
    };

    const std::vector<double> originalVy{
        -0.2,
        0.4,
        -0.9
    };

    // Simula posiciones modificadas después de un paso Euler.
    bodies[0].setPosition(10.0, 20.0);
    bodies[1].setPosition(30.0, 40.0);
    bodies[2].setPosition(50.0, 60.0);

    deviceState.uploadPositions(bodies);

    bool passed = true;

    passed &= compareVectors(
        {10.0, 30.0, 50.0},
        copyDeviceToHost(
            deviceState.positionXData(),
            deviceState.size()
        ),
        "Actualización selectiva de posiciones X"
    );

    passed &= compareVectors(
        {20.0, 40.0, 60.0},
        copyDeviceToHost(
            deviceState.positionYData(),
            deviceState.size()
        ),
        "Actualización selectiva de posiciones Y"
    );

    // uploadPositions() no debe alterar las masas.
    passed &= compareVectors(
        originalMass,
        copyDeviceToHost(
            deviceState.massData(),
            deviceState.size()
        ),
        "Masas conservadas al actualizar posiciones"
    );

    // uploadPositions() tampoco debe alterar las velocidades.
    passed &= compareVectors(
        originalVx,
        copyDeviceToHost(
            deviceState.velocityXData(),
            deviceState.size()
        ),
        "Velocidades X conservadas al actualizar posiciones"
    );

    passed &= compareVectors(
        originalVy,
        copyDeviceToHost(
            deviceState.velocityYData(),
            deviceState.size()
        ),
        "Velocidades Y conservadas al actualizar posiciones"
    );

    // Ahora modifica únicamente las velocidades.
    bodies[0].setVelocity(1.0, -1.0);
    bodies[1].setVelocity(2.0, -2.0);
    bodies[2].setVelocity(3.0, -3.0);

    deviceState.uploadVelocities(bodies);

    passed &= compareVectors(
        {1.0, 2.0, 3.0},
        copyDeviceToHost(
            deviceState.velocityXData(),
            deviceState.size()
        ),
        "Actualización selectiva de velocidades X"
    );

    passed &= compareVectors(
        {-1.0, -2.0, -3.0},
        copyDeviceToHost(
            deviceState.velocityYData(),
            deviceState.size()
        ),
        "Actualización selectiva de velocidades Y"
    );

    return passed;
}

// -----------------------------------------------------------------------------
// Test 3: aceleraciones device -> host
// -----------------------------------------------------------------------------

bool testAccelerationTransfers() {
    std::vector<Particle> bodies = createTestBodies();

    NBodyDeviceState deviceState;
    deviceState.uploadInitialState(bodies);

    bool passed = true;

    constexpr double sentinel = -999.5;

    deviceState.initializeAccelerationOutputs(sentinel);

    passed &= compareVectors(
        {sentinel, sentinel, sentinel},
        copyDeviceToHost(
            deviceState.accelerationXData(),
            deviceState.size()
        ),
        "Inicialización de aceleraciones X"
    );

    passed &= compareVectors(
        {sentinel, sentinel, sentinel},
        copyDeviceToHost(
            deviceState.accelerationYData(),
            deviceState.size()
        ),
        "Inicialización de aceleraciones Y"
    );

    /*
     * Estos valores simulan el resultado que normalmente escribiría
     * el kernel de aceleraciones.
     */
    const std::vector<double> expectedAx{
        0.25,
        -0.50,
        1.75
    };

    const std::vector<double> expectedAy{
        -1.25,
        2.50,
        -3.75
    };

    copyHostToDevice(
        deviceState.accelerationXData(),
        expectedAx
    );

    copyHostToDevice(
        deviceState.accelerationYData(),
        expectedAy
    );

    /*
     * En la simulación real esta sincronización ocurre después
     * del lanzamiento del kernel y antes de leer los resultados.
     */
    CUDA_CHECK(cudaDeviceSynchronize());

    deviceState.downloadAccelerations(bodies);

    for (std::size_t i = 0; i < bodies.size(); ++i) {
        if (
            !approximatelyEqual(expectedAx[i], bodies[i].getAx()) ||
            !approximatelyEqual(expectedAy[i], bodies[i].getAy())
        ) {
            std::cerr
                << "[FAIL] Descarga de aceleraciones en partícula "
                << i
                << ". Esperado=("
                << expectedAx[i] << ", " << expectedAy[i]
                << "), obtenido=("
                << bodies[i].getAx() << ", "
                << bodies[i].getAy() << ")\n";

            passed = false;
        }
    }

    if (passed) {
        std::cout
            << "[PASS] Descarga de aceleraciones device -> host\n";
    }

    return passed;
}

// -----------------------------------------------------------------------------
// Test 4: validaciones
// -----------------------------------------------------------------------------

bool testValidationErrors() {
    bool passed = true;

    std::vector<Particle> bodies = createTestBodies();

    // Estado reservado, pero aún no inicializado.
    NBodyDeviceState uninitializedState(bodies.size());

    bool uninitializedErrorDetected = false;

    try {
        uninitializedState.uploadPositions(bodies);
    } catch (const std::logic_error&) {
        uninitializedErrorDetected = true;
    }

    if (uninitializedErrorDetected) {
        std::cout
            << "[PASS] Detección de estado no inicializado\n";
    } else {
        std::cerr
            << "[FAIL] No se detectó estado no inicializado\n";

        passed = false;
    }

    NBodyDeviceState initializedState;
    initializedState.uploadInitialState(bodies);

    std::vector<Particle> wrongSizeBodies;

    wrongSizeBodies.emplace_back(
        1.0,
        0.0,
        0.0
    );

    wrongSizeBodies.emplace_back(
        2.0,
        1.0,
        1.0
    );

    bool sizeErrorDetected = false;

    try {
        initializedState.uploadPositions(wrongSizeBodies);
    } catch (const std::invalid_argument&) {
        sizeErrorDetected = true;
    }

    if (sizeErrorDetected) {
        std::cout
            << "[PASS] Detección de cantidad incorrecta de cuerpos\n";
    } else {
        std::cerr
            << "[FAIL] No se detectó cantidad incorrecta de cuerpos\n";

        passed = false;
    }

    return passed;
}

// -----------------------------------------------------------------------------
// Test 5: cambio de cantidad de cuerpos
// -----------------------------------------------------------------------------

bool testAutomaticResize() {
    std::vector<Particle> bodies = createTestBodies();

    NBodyDeviceState deviceState;
    deviceState.uploadInitialState(bodies);

    std::vector<Particle> smallerSystem;

    smallerSystem.emplace_back(
        4.0,
        100.0,
        200.0,
        1.0,
        2.0
    );

    smallerSystem.emplace_back(
        5.0,
        300.0,
        400.0,
        3.0,
        4.0
    );

    /*
     * uploadInitialState() debe realocar automáticamente los buffers
     * cuando cambia la cantidad de cuerpos.
     */
    deviceState.uploadInitialState(smallerSystem);

    bool passed = true;

    if (deviceState.size() != smallerSystem.size()) {
        std::cerr
            << "[FAIL] Realocación automática: tamaño incorrecto\n";

        passed = false;
    } else {
        std::cout
            << "[PASS] Realocación automática de buffers\n";
    }

    passed &= compareVectors(
        {4.0, 5.0},
        copyDeviceToHost(
            deviceState.massData(),
            deviceState.size()
        ),
        "Masas después de realocación"
    );

    passed &= compareVectors(
        {100.0, 300.0},
        copyDeviceToHost(
            deviceState.positionXData(),
            deviceState.size()
        ),
        "Posiciones X después de realocación"
    );

    passed &= compareVectors(
        {200.0, 400.0},
        copyDeviceToHost(
            deviceState.positionYData(),
            deviceState.size()
        ),
        "Posiciones Y después de realocación"
    );

    return passed;
}

} // namespace

// -----------------------------------------------------------------------------
// Programa principal
// -----------------------------------------------------------------------------

int main() {
    try {
        bool allPassed = true;

        std::cout
            << "=== Tests NBodyDeviceState ===\n\n";

        allPassed &= testInitialStateUpload();

        std::cout << '\n';

        allPassed &= testSelectiveUpdates();

        std::cout << '\n';

        allPassed &= testAccelerationTransfers();

        std::cout << '\n';

        allPassed &= testValidationErrors();

        std::cout << '\n';

        allPassed &= testAutomaticResize();

        std::cout
            << "\nResultado final: "
            << (allPassed ? "PASS" : "FAIL")
            << '\n';

        return allPassed ? 0 : 1;

    } catch (const std::exception& error) {
        std::cerr
            << "[ERROR] Excepción inesperada: "
            << error.what()
            << '\n';

        return 1;
    }
}