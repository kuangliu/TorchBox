def valid_return(func):
    """Loop until func returns a valid result. Used in dataset get_item."""
    def wrapper(*args, **kwargs):
        while True:
            ret = func(*args, **kwargs)
            if ret is not None:
                return ret
    return wrapper
