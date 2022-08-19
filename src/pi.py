import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# matplotlib setting
plt.axes().set_aspect("equal", adjustable="box")
fig = plt.figure()

# constant
N: int = 5_000

def calc_pi():
    accepted: int = 0
    for i in tqdm(range(N)):
        x, y = np.random.rand(2)
        if x ** 2 + y ** 2 <= 1.0:
            accepted += 1
            plt.plot(x, y, marker="o", c="g", alpha=0.3, mec="g")
        else:
            plt.plot(x, y, marker="o", c="r", alpha=0.3, mec="r")

    pi = 4.0 * accepted / N
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
    plt.title(f"Trials: {N}, " + result)
    plt.grid()
    plt.savefig("../fig/monte_carlo_pi.png")
