

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    return False