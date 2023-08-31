#!/usr/bin/env sh

mpiexec --oversubscribe -np 4 main_parallel.py

mpiexec -np 16 C:\Users\jonas\workspace\hpc-fluid-mechanics\venv\Scripts\python.exe C:\Users\jonas\workspace\hpc-fluid-mechanics\src\experiments\sliding_lit\main_parallel.py

mpiexec -np 16 C:\Users\jonas\workspace\hpc-fluid-mechanics\venv\Scripts\python.exe -m src.experiments.sliding_lit.main_parallel