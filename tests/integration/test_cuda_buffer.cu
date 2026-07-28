#include "CudaBuffer.h"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

bool vectorsAreEqual(
    const std::vector<double>& expected,
    const std::vector<double>& obtained
) {
    if (expected.size() != obtained.size()) {
        return false;
    }

    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != obtained[i]) {
            return false;
        }
    }

    return true;
}

bool testHostDeviceRoundTrip() {
    const std::vector<double> input{
        1.0,
        -2.5,
        3.75,
        100.0
    };

    std::vector<double> output(input.size(), 0.0);

    CudaBuffer<double> buffer(input.size());

    buffer.copyFromHost(input.data(), input.size());
    buffer.copyToHost(output.data(), output.size());

    const bool passed = vectorsAreEqual(input, output);

    std::cout
        << "[" << (passed ? "PASS" : "FAIL") << "] "
        << "Copia host -> device -> host\n";

    return passed;
}

bool testReallocation() {
    CudaBuffer<double> buffer(2);

    buffer.allocate(5);

    const std::vector<double> input{
        10.0,
        20.0,
        30.0,
        40.0,
        50.0
    };

    std::vector<double> output(input.size(), 0.0);

    buffer.copyFromHost(input.data(), input.size());
    buffer.copyToHost(output.data(), output.size());

    const bool passed =
        buffer.size() == input.size() &&
        vectorsAreEqual(input, output);

    std::cout
        << "[" << (passed ? "PASS" : "FAIL") << "] "
        << "Realocacion del buffer\n";

    return passed;
}

bool testOutOfRangeCopy() {
    CudaBuffer<double> buffer(2);

    const std::vector<double> input{
        1.0,
        2.0,
        3.0
    };

    bool exceptionDetected = false;

    try {
        buffer.copyFromHost(input.data(), input.size());
    } catch (const std::out_of_range&) {
        exceptionDetected = true;
    }

    std::cout
        << "[" << (exceptionDetected ? "PASS" : "FAIL") << "] "
        << "Deteccion de copia fuera de rango\n";

    return exceptionDetected;
}

bool testMoveOwnership() {
    const std::vector<double> input{
        7.0,
        8.0,
        9.0
    };

    std::vector<double> output(input.size(), 0.0);

    CudaBuffer<double> original(input.size());
    original.copyFromHost(input.data(), input.size());

    CudaBuffer<double> moved(std::move(original));

    moved.copyToHost(output.data(), output.size());

    const bool passed =
        original.data() == nullptr &&
        original.size() == 0 &&
        moved.size() == input.size() &&
        vectorsAreEqual(input, output);

    std::cout
        << "[" << (passed ? "PASS" : "FAIL") << "] "
        << "Transferencia de propiedad mediante movimiento\n";

    return passed;
}

} // namespace

int main() {
    try {
        bool allPassed = true;

        allPassed &= testHostDeviceRoundTrip();
        allPassed &= testReallocation();
        allPassed &= testOutOfRangeCopy();
        allPassed &= testMoveOwnership();

        std::cout
            << "\nResultado CudaBuffer: "
            << (allPassed ? "PASS" : "FAIL")
            << '\n';

        return allPassed ? 0 : 1;

    } catch (const std::exception& error) {
        std::cerr
            << "Error inesperado en test_cuda_buffer: "
            << error.what()
            << '\n';

        return 1;
    }
}