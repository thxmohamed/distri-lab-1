#include "NBodyDeviceState.h"

#include <algorithm>
#include <stdexcept>
#include <string>

NBodyDeviceState::NBodyDeviceState(std::size_t count) {
    resize(count);
}

void NBodyDeviceState::resize(std::size_t count) {
    if (count == count_) {
        return;
    }

    dMass_.allocate(count);
    dX_.allocate(count);
    dY_.allocate(count);
    dVx_.allocate(count);
    dVy_.allocate(count);
    dAx_.allocate(count);
    dAy_.allocate(count);

    hMass_.resize(count);
    hX_.resize(count);
    hY_.resize(count);
    hVx_.resize(count);
    hVy_.resize(count);
    hAx_.resize(count);
    hAy_.resize(count);

    count_ = count;
    initialized_ = false;
}

void NBodyDeviceState::validateReady(
    const std::vector<Particle>& bodies,
    const char* operation
) const {
    if (!initialized_) {
        throw std::logic_error(
            std::string(operation) +
            ": el estado device todavía no fue inicializado."
        );
    }

    if (bodies.size() != count_) {
        throw std::invalid_argument(
            std::string(operation) +
            ": la cantidad de partículas no coincide con los buffers device."
        );
    }
}

void NBodyDeviceState::uploadInitialState(
    const std::vector<Particle>& bodies
) {
    if (bodies.size() != count_) {
        resize(bodies.size());
    }

    for (std::size_t i = 0; i < count_; ++i) {
        hMass_[i] = bodies[i].getMass();
        hX_[i] = bodies[i].getX();
        hY_[i] = bodies[i].getY();
        hVx_[i] = bodies[i].getVx();
        hVy_[i] = bodies[i].getVy();
    }

    dMass_.copyFromHost(hMass_.data(), count_);
    dX_.copyFromHost(hX_.data(), count_);
    dY_.copyFromHost(hY_.data(), count_);
    dVx_.copyFromHost(hVx_.data(), count_);
    dVy_.copyFromHost(hVy_.data(), count_);

    initialized_ = true;
}

void NBodyDeviceState::uploadPositions(
    const std::vector<Particle>& bodies
) {
    validateReady(
        bodies,
        "NBodyDeviceState::uploadPositions"
    );

    for (std::size_t i = 0; i < count_; ++i) {
        hX_[i] = bodies[i].getX();
        hY_[i] = bodies[i].getY();
    }

    dX_.copyFromHost(hX_.data(), count_);
    dY_.copyFromHost(hY_.data(), count_);
}

void NBodyDeviceState::uploadVelocities(
    const std::vector<Particle>& bodies
) {
    validateReady(
        bodies,
        "NBodyDeviceState::uploadVelocities"
    );

    for (std::size_t i = 0; i < count_; ++i) {
        hVx_[i] = bodies[i].getVx();
        hVy_[i] = bodies[i].getVy();
    }

    dVx_.copyFromHost(hVx_.data(), count_);
    dVy_.copyFromHost(hVy_.data(), count_);
}

void NBodyDeviceState::initializeAccelerationOutputs(double value) {
    if (!initialized_) {
        throw std::logic_error(
            "NBodyDeviceState::initializeAccelerationOutputs: "
            "el estado device todavía no fue inicializado."
        );
    }

    std::fill(hAx_.begin(), hAx_.end(), value);
    std::fill(hAy_.begin(), hAy_.end(), value);

    dAx_.copyFromHost(hAx_.data(), count_);
    dAy_.copyFromHost(hAy_.data(), count_);
}

void NBodyDeviceState::downloadAccelerations(
    std::vector<Particle>& bodies
) {
    validateReady(
        bodies,
        "NBodyDeviceState::downloadAccelerations"
    );

    dAx_.copyToHost(hAx_.data(), count_);
    dAy_.copyToHost(hAy_.data(), count_);

    for (std::size_t i = 0; i < count_; ++i) {
        bodies[i].setAcceleration(hAx_[i], hAy_[i]);
    }
}