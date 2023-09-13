#!/usr/bin/env sh

# mpiexec python -m src.experiments.sliding_lit.main_parallel "$@"

mpiexec --oversubscribe -np 4 python -m src.experiments.sliding_lit.main_parallel "$@"
