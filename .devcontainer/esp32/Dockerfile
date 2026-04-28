FROM espressif/idf:release-v5.5

# Install netstat tools and QEMU emulator
RUN apt-get update && \
    apt-get install -y net-tools qemu-system-xtensa && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
