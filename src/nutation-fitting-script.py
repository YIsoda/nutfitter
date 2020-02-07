import numpy as np
from numpy import cos, exp, pi
import scipy.optimize
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import click
import pathlib
import io


def func(x, a, s, c, T, phi):
    return a * exp(-s * x) * cos(x * 2 * pi / T + phi) + c


@click.command()
@click.argument("array_data_path", type=click.Path(exists=True))
@click.option("-a", "--amplitude", type=float, default=100)
@click.option("-s", "--transverse-relaxation-time", type=float, default=0.1)  # T2
@click.option('-p', '--initial-phase', type=float, default=0)
@click.option('-t', '--nutation-period', type=float, default=20)
@click.option('-c', '--constant', type=float)
def fit_array_data(array_data_path, amplitude=100, transverse_relaxation_time=0.1, initial_phase=0, nutation_period=20,
                   constant=0):  # initial_params: [int]):

    path = pathlib.Path(array_data_path)
    print(path)

    data = np.loadtxt(path)

    plt.plot(data[:, 0], data[:, 1])
    a, s, p, t, c = amplitude, transverse_relaxation_time, initial_phase, nutation_period, constant

    c = 0
    p_opt, p_cov = curve_fit(func, data[:, 0], data[:, 1],
                             # p0=[63000, 0.04, 0, 19, 3 * pi / 2]
                             # p0=[6000, 1/30, 0, 19, 3 * pi / 2]
                             # p0=[63000, 0.04, 0, 20, 3 * pi / 2]
                             p0=[a, s, c, t, p]  # [100000, 0.1, 0, 1, 0]
                             )

    a, s, c, T, phi = p_opt
    # 63000, 0.04, 0, 19, 3 * pi / 2  # p_opt

    x = np.linspace(start=0, stop=40, num=160)
    np.set_printoptions(linewidth=200)
    print(p_opt)
    if p_cov[0, 0] != np.inf:
        print(p_cov)
        print(f'period: {p_opt[3]}')
        with open(f'{path.parent}\\{path.stem}-fit.txt', mode='w') as f:
            f.write(f'p_cov\nperiod: {p_opt[3]}\nfunc: {a} exp(-{s}t) cos(2πt / {T} + {phi}) + {c}')

    plt.plot(x, a * exp(-s * x) * cos(x * 2 * pi / T + phi) + c)
    plt.show()
    plt.savefig(f'{path.parent}\\{path.stem}.png')


# @click.command()
# @click.option('--initial-params', type=(float, float, float, float))  # ??
# def fit_array_data():
#     pass


if __name__ == '__main__':
    fit_array_data()
