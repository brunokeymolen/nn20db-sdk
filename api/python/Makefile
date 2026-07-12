REPO_ROOT := $(abspath ../..)
SDK_ROOT  ?= $(REPO_ROOT)/sdk/linux/current
LIB_DIR   := $(SDK_ROOT)/linux/x86_64-linux-gnu/lib
INC_DIR   := $(SDK_ROOT)/include
SHARED_LIB := libnn20db_py.so

.PHONY: all clean

all: $(SHARED_LIB)

$(SHARED_LIB): $(LIB_DIR)/libnn20db.a abi_check.c
	gcc -shared -fPIC -o $@ \
	    -I$(INC_DIR) abi_check.c \
	    -Wl,--whole-archive $(LIB_DIR)/libnn20db.a -Wl,--no-whole-archive \
	    -lpthread -lm

clean:
	rm -f $(SHARED_LIB)
