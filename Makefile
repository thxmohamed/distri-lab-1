CXX ?= g++ # Compilador a usar

# Habilita paralelismo, optimizaciones y advertencias
# -Isrc para encontrar los archivos de encabezado
CXXFLAGS ?= -Wall -Wextra -O3 -fopenmp -std=c++17 -Iinclude
LDFLAGS  ?= -fopenmp # Bandera para que la librería OpenMP se enlace correctamente

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
.PHONY: all clean benchmark analysis test plots

all: $(TARGET_BIN)

# Enlaza todos los archivos .o indicados por OBJECTS para generar el ejecutable final
$(TARGET_BIN): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compila los archivos .cpp en archivos .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Borra los archivos generados por la compilación
clean:
	rm -f $(TARGET_BIN) $(OBJECTS) run_unit run_integration *.o *.dat
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

plots:
	mkdir -p output
	python3 scripts/plot_performance.py
	python3 scripts/plot_schedule.py
	python3 scripts/plot_amdahl.py
	python3 scripts/plot_trajectories.py
	python3 scripts/plot_energy.py
	python3 scripts/plot_physics.py
	python3 scripts/plot_energy_drift.py
