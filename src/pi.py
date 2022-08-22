from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# matplotlib setting
plt.axes().set_aspect("equal", adjustable="box")
fig = plt.figure()

# constant
N: int = 5_000
accepted: list[list[float]] = [[], []]
rejected: list[list[float]] = [[], []]

def calc_pi():

    for i in tqdm(range(N)):
        x, y = np.random.rand(2)
        if x ** 2 + y ** 2 <= 1.0:
            accepted[0].append(x)
            accepted[1].append(y)
            # plt.plot(x, y, marker="o", c="g", alpha=0.3, mec="g")
        else:
            rejected[0].append(x)
            rejected[1].append(y)
            # plt.plot(x, y, marker="o", c="r", alpha=0.3, mec="r")

    pi = 4.0 * len(accepted[0]) / N
    return pi

if __name__ == "__main__":
    num_pi = np.pi

    print("Calculate pi by probabilistic algorithm.")
    print(f"Trials: {N}")
    mc_pi = calc_pi()
    err = abs(num_pi - mc_pi) / num_pi

    print("Done.")
    result = f"Pi: {mc_pi}, Error: {err:.2%}"
    print(result)

    print("Save fig.")
    plt.scatter(accepted[0], accepted[1], c="g", alpha=0.3, edgecolors="g")
    plt.scatter(rejected[0], rejected[1], c="r", alpha=0.3, edgecolors="r")

    x = np.linspace(0, 1, 300)
    y = np.sqrt(1 - x ** 2)
    plt.plot(x, y, c="k", label="Boundary")
    plt.title(f"Trials: {N}, " + result)
    plt.legend()
    plt.grid()
    plt.savefig("../fig/monte_carlo_pi.png")
