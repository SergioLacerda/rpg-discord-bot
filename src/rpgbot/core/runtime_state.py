event_version = 0


def bump_event_version():
    global event_version
    event_version += 1


def get_event_version():
    return event_version