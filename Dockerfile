# OS que usará el contenedor (imagen con toolkit CUDA para compilar kernels .cu)
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

# Evita que apt-get quede esperando input interactivo (ej. selección de zona horaria vía tzdata)
ENV DEBIAN_FRONTEND=noninteractive

# g++ Compilador de C++
# make Herramienta para automatizar la compilación
# libomp-dev Biblioteca de OpenMP para paralelismo
# libgtest-dev Fuentes de GoogleTest
# cmake Para compilar GoogleTest
RUN apt-get update && apt-get install -y \
    g++ \
    make \
    libomp-dev \
    libgtest-dev \
    cmake \
    python3 \
    python3-numpy \
    python3-matplotlib \
    && cmake -S /usr/src/googletest -B /tmp/gtest-build \
    && cmake --build /tmp/gtest-build \
    && cmake --install /tmp/gtest-build \
    && rm -rf /var/lib/apt/lists/*

# Crea el directorio /app en el contenedor y se situa allí
WORKDIR /app
# Copia el contenido del directorio actual al directorio /app del contenedor
COPY . .

# Por defecto el contenedor al ejecutarse abrirá una terminal bash
CMD ["bash"]