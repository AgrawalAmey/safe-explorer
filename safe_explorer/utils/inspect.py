from functools import reduce
import inspect


def get_source_of_caller_class(depth=1):
    try:
        current_frame = inspect.currentframe()
        caller_frame = reduce(lambda x, _: x.f_back, [current_frame] + [None] * depth)
        return (inspect.getsource(caller_frame
                                    .f_locals["self"]
                                    .__class__))
    except:
        raise TypeError("get_source_of_caller_class should only be used when the caller is an instance method")

def get_class_that_defined_method(meth):
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__):
            if cls.__dict__.get(meth.__name__) is meth:
                return cls
        meth = meth.__func__  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(inspect.getmodule(meth),
                      meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])
        if isinstance(cls, type):
            return cls
    return None  # not required since None would have been implicitly returned anyway