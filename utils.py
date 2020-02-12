import os
import sys


def mkdir(d):
    # exception handling mkdir -p
    try:
        os.makedirs(d)
    except os.error as e:
        if 17 == e.errno:
            # the directory already exists
            pass
        else:
            print('Failed to create "%s" directory!' % d)
            sys.exit(e.errno)
