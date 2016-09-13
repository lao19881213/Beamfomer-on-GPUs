#!/bin/bash

# This script returns whether the provided library or executable
# links against libgcrypt.

ldd $@ | grep -Eq '^[[:space:]]+libgcrypt.so'
