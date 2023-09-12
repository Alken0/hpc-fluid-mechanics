#!/usr/bin/env sh

mpiexec --oversubscribe -np 16 python ./src/experiments/sliding_lit/server.py "$@"
