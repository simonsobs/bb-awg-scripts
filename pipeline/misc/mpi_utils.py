#!/usr/bin/env python
import sys
import numpy as np
import math

# set the default value
_initialized = False
_switch = False
rank = 0
size = 1
comm = None


def print_rnk0(text, rank, if_rank=0, logger=None, verbosity=2):
    if rank == if_rank:
        if logger:
            if verbosity == 1:
                logger.warning(text)
            elif verbosity == 2:
                logger.info(text)
            elif verbosity == 3:
                logger.debug(text)
            else:
                raise ValueError("Verbosity must be between 1 and 3.")
        else:
            print(text)


def init(switch=False, logger=None):
    ''' initialize MPI set-up '''
    global _initialized, _switch
    global rank, size, comm

    exit_code = 0

    if not _initialized:
        _initialized = True
    else:
        print_rnk0("MPI is already intialized", rank, logger=logger)
        return exit_code

    if not switch:
        print_rnk0("WARNING: MPI is turned off by default. "
                   "Use mpi.init(switch=True) to initialize MPI."
                   "MPI is turned off", rank, logger=logger)
        return exit_code
    else:
        _switch = True

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    except ImportError as exc:
        sys.stderr.write("IMPORT ERROR: " + __file__ + " (" + str(exc) + "). "
                         "Could not load mpi4py. MPI will not be used.\n")
    return rank, size, comm


def is_initialized():
    global _initialized
    return _initialized


def is_mpion():
    global _switch
    return _switch


def taskrange(imax, imin=0, shift=0, logger=None):
    """
    """
    global rank, size

    if (not isinstance(imin, int) or not isinstance(imax, int)
            or not isinstance(shift, int)):
        raise TypeError("imin, imax and shift must be integers")
    elif not is_initialized():
        print_rnk0("MPI is not yet properly initialized. "
                   "Are you sure this is what you want to do?",
                   rank, logger=logger, verbosity=1)

    if not is_mpion():
        return np.arange(imin, imax + 1)

    ntask = math.ceil((imax - imin + 1)/size)*size

    subrange = None
    if ntask <= 0:
        print_rnk0("number of task can't be zero", rank, logger=logger,
                   verbosity=1)
        subrange = np.arange(0, 0)  # return zero range
    else:
        if ntask != imax - imin + 1:
            print_rnk0(f"WARNING: setting ntask={ntask}", rank, logger=logger,
                       verbosity=1)
        perrank = ntask // size
        print_rnk0(f"Running {ntask} simulations on {size} nodes",
                   rank, logger=logger)
        subrange = np.arange(rank*perrank, (rank + 1)*perrank)

    return subrange


class DynamicTaskCounter:
    """
    Dynamic (work-stealing) task distribution through a shared counter.

    A single int64 counter lives in an MPI one-sided window on rank 0.
    Each call to next() atomically fetches the current value and increments
    it (MPI_Fetch_and_op), so every task index in [0, ntasks) is handed out
    exactly once, and a rank asks for new work as soon as it finishes its
    current task. No rank is dedicated to bookkeeping.

    Creation and free() are collective: every rank in comm must call them.

    Usage
    -----
        counter = DynamicTaskCounter(comm, ntasks)
        for itask in counter:
            ...
        counter.free()
    """

    def __init__(self, comm, ntasks):
        self.ntasks = int(ntasks)
        self.comm = comm
        self.win = None
        self._serial_next = 0
        if comm is None:
            return
        from mpi4py import MPI
        self._MPI = MPI
        if comm.Get_rank() == 0:
            self._buf = np.zeros(1, dtype=np.int64)
        else:
            self._buf = np.zeros(0, dtype=np.int64)
        self.win = MPI.Win.Create(self._buf, disp_unit=self._buf.itemsize,
                                  comm=comm)
        self._one = np.ones(1, dtype=np.int64)
        self._res = np.zeros(1, dtype=np.int64)

    def next(self):
        """ Returns the next unclaimed task index, or None if none is left. """
        if self.win is None:
            itask = self._serial_next
            self._serial_next += 1
        else:
            MPI = self._MPI
            self.win.Lock(0, MPI.LOCK_SHARED)
            self.win.Fetch_and_op([self._one, MPI.INT64_T],
                                  [self._res, MPI.INT64_T],
                                  0, 0, MPI.SUM)
            self.win.Unlock(0)
            itask = int(self._res[0])
        return itask if itask < self.ntasks else None

    def __iter__(self):
        while True:
            itask = self.next()
            if itask is None:
                return
            yield itask

    def free(self):
        """ Collective: frees the MPI window. """
        if self.win is not None:
            self.win.Free()
            self.win = None


def distribute_tasks(size, rank, ntasks, id_start=0, logger=None):
    """
    Distributes [ntasks] tasks among [size] workers, and outputs
    the list of tasks assigned to a given rank.

    Parameters
    ----------
        size: int
            The number of workers available.
        rank: int
            The number (ID) of the current worker.
        ntasks: int
            The number of tasks.
        logger: logging.Logger
            Logging instance to write output. If None, ignore.
    Returns
    -------
        local_task_ids: list
            List with indices corresponding to the tasks assigned
            to worker [rank]
    """
    if size > ntasks:
        local_start = rank
        local_stop = rank + 1
        if rank == 0 and logger is not None:
            logger.warning(f"You assign {ntasks} tasks to {size} workers. "
                           "Consider using less workers.")
        elif logger is None:
            print(f"Warning: You assign {ntasks} tasks to {size} workers. "
                  "Consider using less workers.")
    else:
        local_start = rank * (ntasks // size)
        local_stop = local_start + (ntasks // size)

    local_task_ids = list(range(id_start, id_start+ntasks))[local_start:local_stop]  # noqa

    if rank >= ntasks:
        local_task_ids = []

    # If there are more tasks than size and ntasks is not divisible by size,
    # there will be a set of ntasks_left < size leftover tasks. We distribute
    # one of them each to the first ntasks_left workers.
    if ntasks % size != 0 and size < ntasks:
        leftover = np.arange(ntasks)[-(ntasks % size):]
        if rank < len(leftover):
            local_task_ids.append(leftover[rank])

    print_rnk0(f"Rank {rank} has {len(local_task_ids)} tasks: "
               f"{np.array(local_task_ids, dtype=int)}",
               rank, if_rank=rank, logger=logger, verbosity=3)
    print_rnk0(f"Total number of tasks is {ntasks}", rank, logger=logger)

    return local_task_ids
