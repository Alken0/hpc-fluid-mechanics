import argparse
import time
from typing import Tuple

import numpy as np
from mpi4py import MPI
from mpi4py.MPI import Cartcomm

from src.experiments.sliding_lit.main import init
from src.shared import boltzmann, plot
from src.shared.util import Parameters


def get_corners(coords, size_x, size_y) -> (bool, bool, bool, bool):
    """
    determine if the current posiiton is at any side; important for applying bounce back or sliding
    :returns: top, bot, left, right
    """
    x = coords[1]
    y = coords[0]

    top, bot, left, right = False, False, False, False
    if x == 0:
        left = True
    if y == 0:
        top = True
    if x == size_x - 1:
        right = True
    if y == size_y - 1:
        bot = True
    return top, bot, left, right


def communicate(F: np.ndarray, cartcomm: Cartcomm, s_and_d: Tuple):
    """sends the data of the current process other processes and receives data as well"""
    sU, dU, sD, dD, sL, dL, sR, dR = s_and_d

    # shift to top (send to the top and receive from the bottom)
    receive_buffer = np.copy(F[:, :, 0])
    send_buffer = np.copy(F[:, :, -2])
    cartcomm.Sendrecv(send_buffer, dU, recvbuf=receive_buffer, source=sU)
    F[:, :, 0] = receive_buffer

    # shift to bot (send to the bottom and receive from the top)
    receive_buffer = np.copy(F[:, :, -1])
    send_buffer = np.copy(F[:, :, 1])
    cartcomm.Sendrecv(send_buffer, dD, recvbuf=receive_buffer, source=sD)
    F[:, :, -1] = receive_buffer

    # shift to left (send to the left and receive from the right
    receive_buffer = np.copy(F[:, -1, :])
    send_buffer = np.copy(F[:, 1, :])
    cartcomm.Sendrecv(send_buffer, dL, recvbuf=receive_buffer, source=sL)
    F[:, -1, :] = receive_buffer

    # shift to right (send to the right and receive from the left)
    receive_buffer = np.copy(F[:, 0, :])
    send_buffer = np.copy(F[:, -2, :])
    cartcomm.Sendrecv(send_buffer, dR, recvbuf=receive_buffer, source=sR)
    F[:, 0, :] = receive_buffer

    return F


def collect_and_plot(data: np.array, rank: int, params: Parameters, cartcomm: Cartcomm, domain_x: int,
                     domain_y: int, size: int, step: int, size_x: int):
    """collects all necessary data on process 0 and creates necessary plots"""
    buf = data[:, 1:-1, 1:-1].copy()
    if rank == 0:
        print(f"currently @ {step}")
        all_data = np.zeros(shape=(9, params.x_dim, params.y_dim))
        for r in range(size):
            coords = cartcomm.Get_coords(r)
            x = coords[1]
            y = coords[0]
            if r != 0:
                cartcomm.Recv(buf, r, 0)
            x_start = x * (domain_x - 2)
            x_end = (x + 1) * (domain_x - 2)
            y_start = (size_x - y - 1) * (domain_y - 2)
            y_end = (size_x - y) * (domain_y - 2)
            # print(f"{x=} {y=} {x_start=} {x_end=} {y_start=} {y_end=} {domain_x=} {domain_y=}")
            # print(f"buf={buf.shape}  all_data={all_data.shape} slice={all_data[:, x_start:x_end, y_start:y_end].shape}")
            all_data[:, x_start:x_end, y_start:y_end] = buf
        print("finished gathering")
        u = boltzmann.velocity(all_data)
        rho = np.average(boltzmann.density(all_data))
        speed = np.average(np.sqrt(u[0] ** 2 + u[1] ** 2))
        v = boltzmann.viscosity_for_omega(params.omega)
        L = params.x_dim
        print(f"{step=} {speed=}, {v=}, Re: {(L * rho * speed) / v}")
        plot.stream_field_sliding_lit(all_data, step=step, path=params.path)
        print("finished plotting")
    else:
        cartcomm.Send(buf, 0, 0)


def get_sources_and_destinations(cartcomm: Cartcomm):
    """returns source and destination ranks for the current process, important for communcation"""
    sR, dR = cartcomm.Shift(1, 1)
    sL, dL = cartcomm.Shift(1, -1)
    sU, dU = cartcomm.Shift(0, -1)
    sD, dD = cartcomm.Shift(0, 1)
    return sU, dU, sD, dD, sL, dL, sR, dR


def main(params: Parameters):
    # start the communicator
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print(f"running: processes={size}; x={params.x_dim}; y={params.y_dim}; iterations={params.iterations}")

    # get coordinates of the current process
    size_x = int(np.floor(np.sqrt(size)))
    size_y = int(size // size_x)
    cartcomm = comm.Create_cart(
        dims=[size_x, size_y],
        periods=[False, False],  # periods: True: continuous space, False not
        reorder=False
    )

    # gather more information about the current computational domain
    coords = cartcomm.Get_coords(rank)
    s_and_d = get_sources_and_destinations(cartcomm)
    top, bot, left, right = get_corners(coords, size_x, size_y)

    # start timing
    start_time = time.time()

    # create data
    domain_x = params.x_dim // size_x + 2
    domain_y = params.y_dim // size_y + 2
    F, sliding_u = init(domain_x, domain_y, params.sliding_u)

    # run simulation
    for i in range(params.iterations):
        communicate(F=F, s_and_d=s_and_d, cartcomm=cartcomm)
        boltzmann.collision(F, omega=params.omega)
        if top:
            boltzmann.slide_top(F, 1, sliding_u)
        F_star = np.copy(F)
        boltzmann.stream(F)
        boltzmann.bounce_back(F, F_star, bot=bot, left=left, right=right)
        if i == 1000 or i % 5000 == 0:
            collect_and_plot(
                data=F,
                cartcomm=cartcomm,
                domain_x=domain_x,
                domain_y=domain_y,
                rank=rank,
                size=size,
                step=i,
                size_x=size_x,
                params=params
            )
    if rank == 0:
        print(f"total time: {time.time() - start_time}ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--x_dim', type=int, default=324)
    parser.add_argument('-y', '--y_dim', type=int, default=324)
    parser.add_argument('-i', '--iterations', type=int, required=False, default=100000)
    args = parser.parse_args()

    parameters = Parameters(
        path="data/sliding_lit_parallel_reynold",
        x_dim=args.x_dim,
        y_dim=args.y_dim,
        iterations=args.iterations,
        omega=1.7,
        sliding_u=0.2,
        sliding_rho=1,
    )
    main(params=parameters)
