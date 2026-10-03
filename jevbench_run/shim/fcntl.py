# Windows stand-in for POSIX fcntl, only for JevBench's budget ledger lock. Our run is one process at $0 cost,
# so there is nothing to lock against.
LOCK_EX, LOCK_SH, LOCK_UN, LOCK_NB = 2, 1, 8, 4
def flock(fd, op): pass
def lockf(fd, op, *a): pass
