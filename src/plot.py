import csv
import os
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt

plt.axes().set_aspect("equal", adjustable="box")
fig, ax = plt.subplots()
basedir: str = "../fig/"

def to_imshow(path: str, name: str, **kwargs):
    try:
        os.makedirs(path)
    except:
        pass

    rows = []
    with open(path+name) as f:
        d = csv.reader(f)
        for row in d:
            cols = []
            for col in row:
                if (col != ""):
                    cols.append(float(col))
            rows.append(cols)

    img = ax.imshow(rows)
    plt.tight_layout()
    plt.show()

    return

def to_animation():
    pass


if __name__ == "__main__":
    to_imshow(basedir, "test")

