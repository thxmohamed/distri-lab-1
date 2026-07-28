CXX ?= g++ # Compilador a usar

# Habilita paralelismo, optimizaciones y advertencias
# -Isrc para encontrar los archivos de encabezado
CXXFLAGS ?= -Wall -Wextra -O3 -fopenmp -std=c++17 -Iinclude
LDFLAGS  ?= -fopenmp # Bandera para que la librería OpenMP se enlace correctamente
NVCC ?= nvcc

NVCCFLAGS ?= -O3 -std=c++17 -Iinclude -Ikernels
NVCC_HOST_FLAGS ?= -Wall,-Wextra,-fopenmp

CUDA_BUFFER_TEST_BIN := run_cuda_buffer
CUDA_DEVICE_STATE_TEST_BIN := run_nbody_device_state
CUDA_ACCEL_TEST_BIN := run_cuda_accelerations
CUDA_SIMULATOR_TEST_BIN := run_nbody_simulator_gpu
TARGET := lab1_distri

# Asigna la extensión correcta al ejecutable dependiendo de si es Windows o no
ifeq ($(OS),Windows_NT)
  EXEEXT := .exe
endif
TARGET_BIN := $(TARGET)$(EXEEXT)

SOURCES := \
	src/main.cpp \
	src/model/Particle.cpp \
	src/model/NBodySystem.cpp \
	src/simulation/Integrator.cpp \
	src/simulation/NBodySimulator.cpp \
	src/Visualizer.cpp \
	benchmarks/Benchmark.cpp \
	benchmarks/MetricsCalculator.cpp
	
UNIT_SOURCES      := $(wildcard tests/unit/*.cpp)
INTEGRATION_SOURCES := $(wildcard tests/integration/*.cpp)
TEST_LIB          := src/model/Particle.cpp src/model/NBodySystem.cpp src/simulation/Integrator.cpp src/simulation/NBodySimulator.cpp

# Indica la Lista de archivos .o que son necesarios para construir el ejecutable
OBJECTS := $(SOURCES:.cpp=.o)

# Indica nombres que no corresponden a archivos reales, sino a tareas o comandos que Make debe ejecutar
.PHONY: all clean benchmark analysis test cuda-test plots

all: $(TARGET_BIN)

# Enlaza todos los archivos .o indicados por OBJECTS para generar el ejecutable final
$(TARGET_BIN): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compila los archivos .cpp en archivos .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Borra los archivos generados por la compilación
clean:
	rm -f $(TARGET_BIN) $(OBJECTS) run_unit run_integration \
		$(CUDA_BUFFER_TEST_BIN) \
		$(CUDA_DEVICE_STATE_TEST_BIN) \
		$(CUDA_ACCEL_TEST_BIN) \
		$(CUDA_SIMULATOR_TEST_BIN) \
		*.o *.dat
	rm -rf output/

benchmark: $(TARGET_BIN)
	./$(TARGET_BIN) --benchmark

analysis: $(TARGET_BIN)
	./$(TARGET_BIN) --analysis

test:
	$(CXX) $(CXXFLAGS) -o run_unit $(UNIT_SOURCES) $(TEST_LIB) $(LDFLAGS) -lgtest -lpthread
	$(CXX) $(CXXFLAGS) -o run_integration $(INTEGRATION_SOURCES) $(TEST_LIB) $(LDFLAGS) -lgtest -lpthread
	./run_unit
	./run_integration

cuda-test:
	$(NVCC) $(NVCCFLAGS) \
		-Xcompiler $(NVCC_HOST_FLAGS) \
		-o $(CUDA_BUFFER_TEST_BIN) \
		tests/integration/test_cuda_buffer.cu

	$(NVCC) $(NVCCFLAGS) \
		-Xcompiler $(NVCC_HOST_FLAGS) \
		-o $(CUDA_DEVICE_STATE_TEST_BIN) \
		tests/integration/test_NBodyDeviceState.cu \
		src/cuda/NBodyDeviceState.cu \
		src/model/Particle.cpp

	$(NVCC) $(NVCCFLAGS) \
		-Xcompiler $(NVCC_HOST_FLAGS) \
		-o $(CUDA_ACCEL_TEST_BIN) \
		tests/integration/test_accelerations.cu \
		kernels/accelerations.cu \
		src/model/Particle.cpp \
		src/model/NBodySystem.cpp

	./$(CUDA_BUFFER_TEST_BIN)
	./$(CUDA_DEVICE_STATE_TEST_BIN)
	./$(CUDA_ACCEL_TEST_BIN)

plots:
	mkdir -p output
	python3 scripts/plot_performance.py
	python3 scripts/plot_schedule.py
	python3 scripts/plot_amdahl.py
	python3 scripts/plot_trajectories.py
	python3 scripts/plot_energy.py
	python3 scripts/plot_physics.py
	python3 scripts/plot_energy_drift.py
	python3 scripts/plot_clauses.py
