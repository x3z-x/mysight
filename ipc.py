from threading import Event

event = Event()

def set_flag() -> None:
    event.set()


def clear_flag() -> None:
    event.clear()


def is_flag_set() -> bool:
    return event.is_set()
