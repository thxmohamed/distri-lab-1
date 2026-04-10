# OS que usará el contenedor
FROM ubuntu:24.04 

# g++ Compilador de C++
# make Herramienta para automatizar la compilación
# libomp-dev Biblioteca de OpenMP para paralelismo
RUN apt-get update && apt-get install -y \
    g++ \
    make \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Crea el directorio /app en el contenedor y se situa allí
WORKDIR /app
# Copia el contenido del directorio actual al directorio /app del contenedor
COPY . .

# Por defecto el contenedor al ejecutarse abrirá una terminal bash
CMD ["bash"]