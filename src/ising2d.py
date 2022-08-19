import numpy as np
import numpy.typing as npt
from tqdm import tqdm

# matplotlib setting
DATA = {
    "total_energy": 0.0,
    "magnetization": 0.0,
    "susceptibility": 0.0,
}

# Simulatioin property
MAX_ITER: int = 4 # Total iteration in simulation
BURN_IN: int = 1024
CHAIN: int = 2
THINNING_INTERVAL: int = 4
DELTA_T: float = 0.1 # Stride of temperature
DELTA_TC: float = 0.01 # Stride of critical temperature

# System
X_LENGTH: int = 4
Y_LENGTH: int = 4
VOLUME: float = 1.0 * X_LENGTH * Y_LENGTH

# Physical parameters
J: float = 1.0 # Coupling constant of neighborhood interaction
K: float = 0.0 # Strength of external magnetic field
Tc: float = 2.26918531421 * J # Critical temperature, exact value is 2.0 * J / np.log(1.0 + np.sqrt(2))
T_MIN: float = 0.1 # Lower limit of temperature
T_MAX: float = 0.1 # Upper limit of temperature
Tc_MIN: float = 0.1 # Lower limit around critical temperature
Tc_MAX: float = 0.1 # Upper limit around critical temperature

# Alias of type annotation
Spin = npt.NDArray[np.float_]


def total_energy(sigma: Spin) -> float:
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

def magnetization(sigma: Spin) -> float:
    m: float = np.sum(sigma)
    return m / VOLUME

def susceptibility(sigma: Spin) -> float:
    pass

def specific_heat_capacity(sigma: Spin) -> float:
    pass

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

def mcstep(num: int, sigma: Spin, temperature: float):
    for n in range(num):
        i = np.random.randint(0, X_LENGTH)
        j = np.random.randint(0, Y_LENGTH)
        is_accepted: bool = metropolis_hastings_criterion(sigma, i, j, temperature)

        if is_accepted:
            sigma[i][j] = -sigma[i][j]

        yield n, sigma

if __name__ == "__main__":
    print(f"\033[1;34mMonte Carlo Simulation: 2D Ising Model\033[m")
    sigma = 2.0 * np.random.randint(2, size=(X_LENGTH, Y_LENGTH)) - 1.0 # initial state
    # sigma = np.ones([X_LENGTH, Y_LENGTH])
    print(sigma)

    temperature = 10 * Tc
    for n, state in mcstep(MAX_ITER, sigma, temperature):
        print(n, state)
        # if (n % THINNING_INTERVAL == 0):
        #     pass
    print(DATA)
