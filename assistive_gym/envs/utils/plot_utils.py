import matplotlib.pyplot as plt
import numpy as np


def plot(vals, title, xlabel, ylabel):
    plt.figure()
    plt.plot(vals)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_cmaes_metrics(mean_cost, mean_dist, mean_m, mean_energy, mean_torque):
    # Plot the fitness values
    plot(mean_cost, "Cost Function", "Iteration", "Cost")

    # Plot the distance values
    plot(mean_dist, "Distance Values", "Iteration", "Distance")

    # Plot the manipubility values
    plot(mean_m, "Manipubility Values", "Iteration", "Manipubility")

    # Plot energy
    plot (mean_energy, "Energy Values", "Iteration", "Energy")

    # Plot torque
    plot (mean_torque, "Torque Values", "Iteration", "Torque")

def plot_mean_evolution(mean_evolution):
    # Plot the mean vector evolution
    mean_evolution = np.array(mean_evolution)
    plt.figure()
    for i in range(mean_evolution.shape[1]):
        plt.plot(mean_evolution[:, i], label=f"Dimension {i + 1}")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Value")
    plt.title("Mean Vector Evolution")
    plt.legend()
    plt.show()
