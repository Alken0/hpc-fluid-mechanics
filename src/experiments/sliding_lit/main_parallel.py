import numpy as np
from mpi4py import MPI
from mpi4py.MPI import Cartcomm

from src.experiments.sliding_lit.main import init
from src.shared import boltzmann, plot
from src.shared.util import Parameters


def get_corners(coords, size_x, size_y) -> (bool, bool, bool, bool):
    """
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


def Communicate2(grid, cartcomm, s_and_d):
    sU, dU, sD, dD, sL, dL, sR, dR = s_and_d

    # Send to right which is destination right and receive from left which is source right
    rb = np.copy(grid[:, :, 0])
    sb = np.copy(grid[:, :, -2])
    cartcomm.Sendrecv(sb, dU, recvbuf=rb, source=sU)
    grid[:, :, 0] = rb

    # Send to the bottom and receive from the top
    rb = np.copy(grid[:, :, -1])
    sb = np.copy(grid[:, :, 1])
    cartcomm.Sendrecv(sb, dD, recvbuf=rb, source=sD)
    grid[:, :, -1] = rb

    # Send to the left and receive from the right
    rb = np.copy(grid[:, -1, :])
    sb = np.copy(grid[:, 1, :])
    cartcomm.Sendrecv(sb, dL, recvbuf=rb, source=sL)
    grid[:, -1, :] = rb

    # Send to the right and receive from the left
    rb = np.copy(grid[:, 0, :])
    sb = np.copy(grid[:, -2, :])
    cartcomm.Sendrecv(sb, dR, recvbuf=rb, source=sR)
    grid[:, 0, :] = rb

    return grid


def communicate(data: np.array, rank: int, size_x: int, cartcomm: Cartcomm, top, bot, left, right):
    if not top:
        recvbuf = data[[4, 7, 8], 1, :].copy()
        sendbuf = data[[2, 5, 6], 1, :].copy()
        # print(f"{rank} <-> {rank - size_x}")
        cartcomm.Sendrecv(
            dest=rank - size_x,
            source=rank - size_x,
            sendbuf=sendbuf,
            recvbuf=recvbuf,
        )
        data[[4, 7, 8], 1, :] = recvbuf
    if not bot:
        recvbuf = data[[2, 5, 6], -2, :].copy()
        sendbuf = data[[4, 7, 8], -2, :].copy()
        cartcomm.Sendrecv(
            dest=rank + size_x,
            source=rank + size_x,
            sendbuf=sendbuf,
            recvbuf=recvbuf,
        )
        data[[2, 5, 6], -2, :] = recvbuf
    if not left:
        recvbuf = data[[1, 5, 8], :, 1].copy()
        sendbuf = data[[3, 6, 7], :, 1].copy()
        cartcomm.Sendrecv(
            dest=rank - 1,
            source=rank - 1,
            sendbuf=sendbuf,
            recvbuf=recvbuf,
        )
        data[[1, 5, 8], :, 1] = recvbuf
    if not right:
        recvbuf = data[[3, 6, 7], :, -2].copy()
        sendbuf = data[[1, 5, 8], :, -2].copy()
        cartcomm.Sendrecv(
            dest=rank + 1,
            source=rank + 1,
            sendbuf=sendbuf,
            recvbuf=recvbuf,
        )
        data[[3, 6, 7], :, -2] = recvbuf


def collect(data: np.array, rank, x_dim, y_dim, cartcomm, domain_x, domain_y, size, step):
    if rank == 0:
        print(f"currently @ {step}")
        all_data = np.zeros(shape=(9, x_dim, y_dim))
        buf = data[:, 1:-1, 1:-1].copy()
        for r in range(size):
            coords = cartcomm.Get_coords(r)
            x = coords[1]
            y = coords[0]
            if r != 0:
                cartcomm.Recv(buf, r, 0)
            x_start = x * (domain_x - 2)
            x_end = (x + 1) * (domain_x - 2)
            y_start = (4 - y - 1) * (domain_y - 2)
            y_end = (4 - y) * (domain_y - 2)
            all_data[:, x_start:x_end, y_start:y_end] = buf
        print("finished gathering")
        plot.velocity_field_couette_flow(all_data, step=step, path=params.path)
        print("finished plotting")
    else:
        cartcomm.Send(data[:, 1:-1, 1:-1].copy(), 0, 0)


def get_sources_and_destinations(cartcomm):
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

    # get coordinates of current process
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

    # create data
    domain_x = params.x_dim // size_x + 2
    domain_y = params.y_dim // size_y + 2
    F, sliding_u = init(domain_x, domain_y, params.sliding_u)

    # test
    for i in range(50000):
        boltzmann.collision(F, omega=params.omega)
        if top:
            boltzmann.slide_top(F, params.sliding_rho, sliding_u)
        F_star = np.copy(F)
        boltzmann.stream(F)
        boltzmann.bounce_back(F, F_star, bot=True)
        Communicate2(
            grid=F,
            cartcomm=cartcomm,
            s_and_d=s_and_d
        )
    collect(
        data=F,
        cartcomm=cartcomm,
        domain_x=domain_x,
        domain_y=domain_y,
        x_dim=params.x_dim,
        y_dim=params.y_dim,
        rank=rank,
        size=size,
        step=49999
    )
    exit(0)

    # print(f"{rank=}  {sU=} {dU=}   {sD=} {dD=}   {sL=} {dL=}   {sR=} {dR=}   {top=} {bot=} {left=} {right=}")

    for i in range(params.iterations):
        boltzmann.collision(F, omega=params.omega)
        if top:
            boltzmann.slide_top(F, 1, sliding_u)
        F_star = np.copy(F)
        boltzmann.stream(F)
        boltzmann.bounce_back(F, F_star, bot=bot, left=left, right=right)
        Communicate2(grid=F, s_and_d=s_and_d, cartcomm=cartcomm)
        if i % 1000 == 0:
            collect(
                data=F,
                cartcomm=cartcomm,
                domain_x=domain_x,
                domain_y=domain_y,
                x_dim=params.x_dim,
                y_dim=params.y_dim,
                rank=rank,
                size=size,
                step=i
            )


if __name__ == '__main__':
    params = Parameters(
        path="data/sliding_lit_parallel",
        x_dim=100,
        y_dim=100,
        iterations=100000,
    )
    main(params)

    # copy paste
    # print(f"{rank=} {coordinates=}  {so urce=}  {destination=}")
