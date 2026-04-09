CXX ?= g++

# Flags requeridas por el enunciado 
CXXFLAGS ?= -Wall -Wextra -O3 -fopenmp -std=c++17 -Isrc
LDFLAGS  ?= -fopenmp

TARGET := nbody2d

ifeq ($(OS),Windows_NT)
  EXEEXT := .exe
endif

TARGET_BIN := $(TARGET)$(EXEEXT)

# Nota: actualmente solo src/model tiene implementación real. El resto
# de carpetas están como placeholders, por eso compilamos un demo mínimo.
SOURCES := \
	src/main.cpp \
	src/model/Particle.cpp \
	src/model/NBodySystem.cpp

OBJECTS := $(SOURCES:.cpp=.o)

.PHONY: all clean benchmark analysis test

all: $(TARGET_BIN)

$(TARGET_BIN): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(TARGET_BIN) $(OBJECTS) *.o *.dat *.png

benchmark: $(TARGET_BIN)
	./$(TARGET_BIN) --benchmark

analysis: $(TARGET_BIN)
	./$(TARGET_BIN) --analysis

test: $(TARGET_BIN)
	./$(TARGET_BIN) --self-test
