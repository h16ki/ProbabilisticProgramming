from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# matplotlib setting
plt.axes().set_aspect("equal", adjustable="box")
fig = plt.figure()

# constant
N: int = 5_000

np.random.seed(1)
dt = [[], [], [], []]

def calc_pi(n, save=5000):

    accepted: list[list[float]] = [[], []]
    rejected: list[list[float]] = [[], []]
    for i in tqdm(range(n)):
        x, y = np.random.rand(2)
        if x ** 2 + y ** 2 <= 1.0:
            accepted[0].append(x)
            accepted[1].append(y)
            # plt.plot(x, y, marker="o", c="g", alpha=0.3, mec="g")
        else:
            rejected[0].append(x)
            rejected[1].append(y)
            # plt.plot(x, y, marker="o", c="r", alpha=0.3, mec="r")

        if n == save:
            dt[0] = accepted[0]
            dt[1] = accepted[1]
            dt[2] = rejected[0]
            dt[3] = rejected[1]

    pi = 4.0 * len(accepted[0]) / n

    return pi


def trapezoid_pi(n):
    integrand = lambda x: 1.0 / (1.0 + x ** 2.0)
    delta = 1 / n
    x = 0
    area = 0
    for k in range(1, n):
        upper = x + k*delta
        lower = x + (k-1) * delta
        area += 0.5 * delta * (integrand(lower) + integrand(upper))

    return 4.0 * area

if __name__ == "__main__":
    num_pi = np.pi
    n_list = [5, 50, 500, 5000, 50000]
    print("Calculate pi by trapezoid formula.")
    print(f"Division: {N}")
    # trapezoid_pi = trapezoid_pi()
    trapezoid, err = [], []
    for n in n_list:
        tpi = trapezoid_pi(n)
        trapezoid.append(tpi)
        e = abs(num_pi - tpi) / num_pi
        err.append(f"{e:.3%}")

    # err = abs(num_pi - trapezoid_pi) / num_pi

    for i,n in enumerate(n_list):
        print(f"| {n} | {trapezoid[i]:.10} | {err[i]}")
    print(f"| - | 3.141592653 | {0:.3%}")

    # result = f"Pi: {trapezoid}, Error: {err}"
    # print(result)

    print("Calculate pi by probabilistic algorithm.")
    print(f"Trials: {N}")
    mpi = []
    err = []
    for n in n_list:
        mc_pi = calc_pi(n)
        mpi.append(mc_pi)
        e = abs(num_pi - mc_pi) / num_pi
        err.append(e)

    print("Done.")
    for i,n in enumerate(n_list):
        print(f"| {n} | {mpi[i]:.10} | {err[i]:.3%}")
    print(f"| - | 3.14159265 | {0:.3%}")

    print("Save fig.")
    plt.scatter(dt[0], dt[1], c="g", alpha=0.3, edgecolors="g")
    plt.scatter(dt[2], dt[3], c="r", alpha=0.3, edgecolors="r")

    x = np.linspace(0, 1, 300)
    y = np.sqrt(1 - x ** 2)
    plt.plot(x, y, c="k", label="Boundary")
    plot_pi = 4.0 * len(dt[0]) / 5000
    plot_err = abs(num_pi - plot_pi) / num_pi
    plt.title(f"N: {5000}, pi: {plot_pi}, err: {plot_err:.3%}")
    plt.legend()
    plt.grid()
    plt.savefig("../../fig/monte_carlo_pi.png")
