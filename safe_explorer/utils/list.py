

def select_with_predicate(X, predicates):
    assert len(X) == len(predicates)
    return [x for (x, predicate) in zip(X, predicates) if predicate]

def flatten(l):
    return [item for sublist in l for item in sublist]

def for_each(f, l):
    for x in l:
        f(x)
