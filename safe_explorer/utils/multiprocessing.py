import multiprocessing


def _paramap_fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def paramap(f, X, nprocs=multiprocessing.cpu_count()):
    """
    A parallel map function that works
    because multiprocessing.pool.map sucks
    """
    nprocs = multiprocessing.cpu_count() if nprocs < 0 else nprocs
    
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=_paramap_fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]