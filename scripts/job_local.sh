#!/usr/bin/env sh

# mpiexec python -m src.experiments.sliding_lit.main_parallel "$@"

mpirun --oversubscribe -n 4 python server.py "$@"
