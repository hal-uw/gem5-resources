HIP_PATH?= /opt/rocm/hip
HIPCC=$(HIP_PATH)/bin/hipcc --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13

BIN_DIR?= ./bin

square: $(BIN_DIR)/square

$(BIN_DIR)/square: square.cpp $(BIN_DIR)
	$(HIPCC) --amdgpu-target=gfx900,gfx902,gfx90a $(CXXFLAGS) square.cpp -o $(BIN_DIR)/square

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

.PHONY: square clean
