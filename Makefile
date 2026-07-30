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
GPU_BENCHMARK_BIN := run_benchmark_gpu
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
.PHONY: all clean benchmark analysis test cuda-test benchmark-gpu plots plots-gpu

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
		$(GPU_BENCHMARK_BIN) \
		*.o *.dat *.log
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

	$(NVCC) $(NVCCFLAGS) \
		-Xcompiler $(NVCC_HOST_FLAGS) \
		-o $(CUDA_SIMULATOR_TEST_BIN) \
		tests/integration/test_NBodySimulatorGpu.cu \
		src/cuda/NBodySystemGpu.cu \
		src/cuda/NBodySimulatorGpu.cu \
		src/cuda/NBodyDeviceState.cu \
		kernels/accelerations.cu \
		kernels/metrics.cu \
		src/model/Particle.cpp \
		src/model/NBodySystem.cpp \
		src/simulation/Integrator.cpp \
		src/simulation/NBodySimulator.cpp

	./$(CUDA_BUFFER_TEST_BIN)
	./$(CUDA_DEVICE_STATE_TEST_BIN)
	./$(CUDA_ACCEL_TEST_BIN)
	./$(CUDA_SIMULATOR_TEST_BIN)

# Compila el driver de la matriz de benchmarks GPU (N x variante x blockDim.x).
# Pensado para correr en el clúster DIINF, no en CI: las mediciones finales de
# rendimiento solo valen si salen de ahi, no de una corrida en CI.
benchmark-gpu:
	$(NVCC) $(NVCCFLAGS) \
		-Xcompiler $(NVCC_HOST_FLAGS) \
		-o $(GPU_BENCHMARK_BIN) \
		benchmarks/benchmark_gpu_main.cu \
		benchmarks/BenchmarkGpu.cu \
		benchmarks/Benchmark.cpp \
		kernels/accelerations.cu \
		src/cuda/NBodyDeviceState.cu \
		src/cuda/NBodySystemGpu.cu \
		src/model/Particle.cpp \
		src/model/NBodySystem.cpp \
		src/simulation/Integrator.cpp \
		src/simulation/NBodySimulator.cpp

	./$(GPU_BENCHMARK_BIN)

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

# Graficos GPU del Lab 2. Requieren los .dat generados por `make benchmark-gpu`
# en el clúster DIINF; separado de `plots` para no depender de ellos al
# graficar solo el Lab 1.
plots-gpu:
	mkdir -p output
	python3 scripts/plot_gpu_speedup_vs_n.py
	python3 scripts/plot_gpu_transfer_impact.py
	python3 scripts/plot_gpu_blockdim.py
	python3 scripts/plot_gpu_amdahl.py
	python3 scripts/plot_gpu_variant_comparison.py
	python3 scripts/plot_trajectories.py
	python3 scripts/plot_energy.py
