from cProfile import label
from timeit import repeat
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from dataclasses import dataclass
import os, glob, shutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation

MTH = ["Metropolis-Hastings", "Heat-Bath", "Self-Learning", "Swendsen-Wang", "Wolff"]

# Simulatioin property
np.random.seed(1)
fig_path = "./../../fig/"

MAX_ITER: int = 4096 # Total iteration in simulation
BURN_IN: int = 1024
CHAIN: int = 2
THINNING_INTERVAL: int = 8
DELTA_T: float = 0.1 # Stride of temperature
DELTA_TC: float = 0.01 # Stride of critical temperature

# System
X_LENGTH: int = 36
Y_LENGTH: int = 64
VOLUME: float = 1.0 * X_LENGTH * Y_LENGTH

# Physical parameters
J: float = 1.0 # Coupling constant of neighborhood interaction
K: float = 0.0 # Strength of external magnetic field
Tc: float = 2.26918531421 * J # Critical temperature, exact value is 2.0 * J / np.log(1.0 + np.sqrt(2))
T_MIN: float = 0.1 # Lower limit of temperature
T_MAX: float = 4.0 # Upper limit of temperature
Tc_MIN: float = 0.1 # Lower limit around critical temperature
Tc_MAX: float = 0.1 # Upper limit around critical temperature

# Alias of type annotation
Spin = npt.NDArray[np.float_]

@dataclass
class IsingModel:
    size: list[float]
    temperature: float
    save_dir: str

    def total_energy(self, sigma: Spin) -> float:
        energy: float = 0.0
        for i in range(X_LENGTH):
            for j in range(Y_LENGTH):
                up = (i - 1) % X_LENGTH
                down = (i + 1) % X_LENGTH
                left = (j - 1) % Y_LENGTH
                right = (j + 1) % Y_LENGTH

                neighbor = sigma[up, j] + sigma[i, right] + sigma[down, j] + sigma[i, left]
                energy += -0.5 * J * sigma[i, j] * neighbor

        energy += -K * np.sum(sigma)
        return energy

    def magnetization(self, sigma: Spin) -> float:
        m: float = np.sum(sigma)
        return m / VOLUME

    def susceptibility(self, sigma: Spin) -> float:
        return 1.0

    def heat_capacity(self, sigma: Spin) -> float:
        return 1.0

    @staticmethod
    def array_to_img(sigma: Spin):
        pass

    @staticmethod
    def export(sigma: Spin, path: str):
        pass

def to_image(path: str):
    for n, d in enumerate(glob.glob(path)):
        array = np.load(d)
        name = os.path.basename(d)
        t, _ = os.path.splitext(name)
        print(t)

        artists = []
        fig, ax = plt.subplots(1,1)
        for mcmc in range(array.shape[0]):
            if mcmc % THINNING_INTERVAL == 0:
                sigma = array[mcmc]
                # title = plt.text(0.5,1.01,mcmc, ha="center",va="bottom",color=np.random.rand(3),
                #          transform=ax.transAxes, fontsize="large")
                # text = ax.text(mcmc, mcmc, mcmc)
                img = plt.imshow(sigma, label=f"T: {t}, MCMC: {mcmc}")
                artists.append([img])
                # plt.title(f"T: {t}, MCMC: {mcmc}")

        ani = animation.ArtistAnimation(fig, artists=artists, interval=1, repeat=False)
        save = fig_path + f"T:{t}" + ".gif"
        ani.save(save)
        # break
    return

def metropolis_hastings_criterion(sigma: Spin, i: int, j: int, temperature: float) -> bool:
    up = (i - 1) % X_LENGTH
    down = (i + 1) % X_LENGTH
    left = (j - 1) % Y_LENGTH
    right = (j + 1) % Y_LENGTH

    neighbor = sigma[up, j] + sigma[i, right] + sigma[down, j] + sigma[i, left]
    de = 2.0 * J * sigma[i, j] * neighbor
    delta_boltzmann_weight = np.exp(-de / temperature)

    if (de < 0 or np.random.rand() < delta_boltzmann_weight):
        return True
    else:
        return False

def mcstep(sigma: Spin, temperature: float, num: int = MAX_ITER):
    for n in range(num):
        i = np.random.randint(0, X_LENGTH)
        j = np.random.randint(0, Y_LENGTH)
        is_accepted: bool = metropolis_hastings_criterion(sigma, i, j, temperature)

        if is_accepted:
            sigma[i][j] = -sigma[i][j]

        yield n, sigma


if __name__ == "__main__":
    print(f"\033[1;34mMonte Carlo Simulation: 2D Ising Model\033[m")
    print(f"Select a MCMC algorithm (0: {MTH[0]})")
    print("> ", end="")
    alg = int(input().strip())
    dir = MTH[alg].replace("-", "")

    skip = False
    print("Initializing...")
    if (os.path.exists(f"./{dir}")):
        print("The spin state has been generated.")
        print("Could you regenerate?")
        print(f"(yes/No) > ", end="")
        select = input().lower()
        skip = {"no": True, "yes": False}[select]
    else:
        pass

    if not skip:
        try:
            shutil.rmtree(dir)
        except FileNotFoundError:
            pass
        sigma = 2.0 * np.random.randint(2, size=(X_LENGTH, Y_LENGTH)) - 1.0 # initial state
        # sigma = np.ones([X_LENGTH, Y_LENGTH])
        # print(sigma)

        print(f"Generate {MAX_ITER} spin states")
        print(f"where the first {BURN_IN} is the burn-in step.")

        # temperature = 0.1 * Tc
        # for n, state in mcstep(sigma, temperature, num=MAX_ITER):
        #     print(state)
            # if (n % THINNING_INTERVAL == 0):
            #     pass

        t_range = np.linspace(T_MIN, T_MAX)

        for t in t_range:
            conf = np.zeros([MAX_ITER, X_LENGTH, Y_LENGTH])
            for n, state in mcstep(sigma, t, MAX_ITER):
                conf[n] = state

            os.makedirs(f"{dir}", exist_ok=True)
            np.save(dir + f"/{t}", conf)

    to_image(path=f"./{dir}/*")

    print("Complete.")
