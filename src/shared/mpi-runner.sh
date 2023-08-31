#!/usr/bin/env sh

mpiexec -np 16 mpitest.py

mpiexec -np 16 C:\Users\jonas\workspace\hpc-fluid-mechanics\venv\Scripts\python.exe C:\Users\jonas\workspace\hpc-fluid-mechanics\src\experiments\sliding_lit\main_parallel.py
