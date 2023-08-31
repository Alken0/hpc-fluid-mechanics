import numpy as np
from mpi4py import MPI
from mpi4py.MPI import Cartcomm

from src.shared.util import Parameters


def get_corners(size_x, size_y, x, y) -> (bool, bool, bool, bool):
    """
    :returns: top, bot, left, right
    """
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


def communicate(data: np.array, rank: int, size_x: int, cartcomm: Cartcomm, top, bot, left, right):
    if not left:
        recvbuf = np.zeros(shape=(3, data.shape[2]), dtype="float64")
        sendbuf = data[[3, 6, 7], :, 0].copy()
        cartcomm.Sendrecv(
            dest=rank - 1,
            source=rank - 1,
            sendbuf=sendbuf,
            recvbuf=recvbuf,
        )
        data[[1, 5, 8], :, 0] = recvbuf
    if not right:
        recvbuf = np.zeros(shape=(3, data.shape[2]), dtype="float64")
        sendbuf = data[[1, 5, 8], :, -1].copy()
        cartcomm.Sendrecv(
            dest=rank + 1,
            source=rank + 1,
            sendbuf=sendbuf,
            recvbuf=recvbuf,
        )
        data[[3, 6, 7], :, -1] = recvbuf
    if not top:
        recvbuf = np.zeros(shape=(3, data.shape[1]), dtype="float64")
        sendbuf = data[[2, 5, 6], 0, :].copy()
        cartcomm.Sendrecv(
            dest=rank - size_x,
            source=rank - size_x,
            sendbuf=sendbuf,
            recvbuf=recvbuf,
        )
        data[[4, 7, 8], 0, :] = recvbuf
    if not bot:
        recvbuf = np.zeros(shape=(3, data.shape[1]), dtype="float64")
        sendbuf = data[[4, 7, 8], -1, :].copy()
        cartcomm.Sendrecv(
            dest=rank + size_x,
            source=rank + size_x,
            sendbuf=sendbuf,
            recvbuf=recvbuf,
        )
        data[[2, 5, 6], -1, :] = recvbuf


def collect(data: np.array, rank, x_dim, y_dim, cartcomm, domain_x, domain_y, size):
    if rank == 0:
        all_data = np.zeros(shape=(9, x_dim, y_dim))
        buf = data
        for r in range(size):
            coords = cartcomm.Get_coords(r)
            if r != 0:
                cartcomm.Recv(buf, r, 0)
            x_start = coords[0] * domain_x
            x_end = (coords[0] + 1) * domain_x
            y_start = coords[1] * domain_y
            y_end = (coords[1] + 1) * domain_y
            all_data[:, x_start:x_end, y_start:y_end] = buf
        print(all_data[5])
    else:
        cartcomm.Send(data, 0, 0)


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
    coords = cartcomm.Get_coords(rank)

    # create data
    domain_x = params.x_dim // size_x
    domain_y = params.y_dim // size_y
    data = np.zeros(shape=(9, domain_x, domain_y), dtype="float64") + rank

    # test
    top, bot, left, right = get_corners(size_x, size_y, coords[1], coords[0])
    # print(f"{rank} {coords} {top, bot, left, right}")
    communicate(
        data=data,
        rank=rank,
        size_x=size_x,
        cartcomm=cartcomm,
        right=right,
        top=top,
        bot=bot,
        left=left,
    )
    collect(
        data=data,
        cartcomm=cartcomm,
        domain_x=domain_x,
        domain_y=domain_y,
        x_dim=params.x_dim,
        y_dim=params.y_dim,
        rank=rank,
        size=size
    )


if __name__ == '__main__':
    params = Parameters(
        path="",
        x_dim=16,
        y_dim=16
    )
    main(params)

    # copy paste
    # print(f"{rank=} {coordinates=}  {so urce=}  {destination=}")
