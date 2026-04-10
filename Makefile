CXX ?= g++ # Compilador a usar

# Habilita paralelismo, optimizaciones y advertencias
# -Isrc para encontrar los archivos de encabezado
CXXFLAGS ?= -Wall -Wextra -O3 -fopenmp -std=c++17 -Isrc -Isrc/model -Isrc/simulation
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
	src/model/NBodySystem.cpp

# Indica la Lista de archivos .o que son necesarios para construir el ejecutable
OBJECTS := $(SOURCES:.cpp=.o)

# Indica nombres que no corresponden a archivos reales, sino a tareas o comandos que Make debe ejecutar
.PHONY: all clean benchmark analysis test

all: $(TARGET_BIN)

# Enlaza todos los archivos .o indicados por OBJECTS para generar el ejecutable final
$(TARGET_BIN): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Compila los archivos .cpp en archivos .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Borra los archivos generados por la compilación
clean:
	rm -f $(TARGET_BIN) $(OBJECTS) *.o *.dat *.png

benchmark: $(TARGET_BIN)
	./$(TARGET_BIN) --benchmark

analysis: $(TARGET_BIN)
	./$(TARGET_BIN) --analysis

test: $(TARGET_BIN)
	./$(TARGET_BIN) --self-test
