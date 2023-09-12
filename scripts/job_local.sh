#!/usr/bin/env sh

mpiexec --oversubscribe -np 16 python -m src.experiments.sliding_lit.main_parallel "$@"
