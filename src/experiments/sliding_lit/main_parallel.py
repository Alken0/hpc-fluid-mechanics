#!/home/jonas/workspace/hpc-fluid-mechanics/venv/bin/python

import numpy as np
from mpi4py import MPI


def main(x_dim, y_dim):
    # start the communicator
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # get coordinates of current parallel computing field (section)
    x_dim_sub = int(np.floor(np.sqrt(size)))
    y_dim_sub = int(size // x_dim_sub)

    # get coordinates of current process
    cartcomm = comm.Create_cart(dims=[x_dim_sub, y_dim_sub], periods=[False, False], reorder=False)  # periods: continuous space
    coordinates = cartcomm.Get_coords(rank)

    # create data
    domain_x = x_dim // x_dim_sub + 2
    domain_y = y_dim // y_dim_sub + 2
    data = np.zeros((9, domain_x, domain_y)) + rank

    # apply communication
    output = f"{coordinates} -> "
    for name, direction, disp in [('right', 1, 1), ('left', 1, -1), ('up', 0, -1), ('down', 0, 1)]:
        _, destination_rank = cartcomm.Shift(direction, disp)
        if destination_rank > 0:
            destination_coordinates = cartcomm.Get_coords(destination_rank)
            output += f"{destination_coordinates}, "
        else:
            output += "______, "

    print(output)

    # where to receive from and where send to
    sR, dR = cartcomm.Shift(1, 1)
    sL, dL = cartcomm.Shift(1, -1)
    # sU,dU = cartcomm.Shift(0,1)
    # sD,dD = cartcomm.Shift(0,-1)
    sU, dU = cartcomm.Shift(0, -1)
    sD, dD = cartcomm.Shift(0, 1)
    #
    sd = np.array([sR, dR, sL, dL, sU, dU, sD, dD], dtype=int)
    # ## Analysis of the domain
    allrcoords = comm.gather(coordinates, root=0)
    allDestSourBuf = np.zeros(size * 8, dtype=int)
    comm.Gather(sd, allDestSourBuf, root=0)
    #
    # print(sd)
    if rank == 0:
        # print(' ')
        cartarray = np.ones((y_dim_sub, x_dim_sub), dtype=int)
        allDestSour = np.array(allDestSourBuf).reshape((size, 8))
    # copy paste
    # print(f"{rank=} {coordinates=}  {source=}  {destination=}")


if __name__ == '__main__':
    main(16, 16)
