import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize

c = 299792.458
# Define the speed of light as a global variable (in km/s)


class Cosmology:
    def __init__(self, H0, Omega_m, Omega_lambda):
        """
        __init__ method is used to create classes, it defines the attributes of the class
        i.e. the distinct features of the objects contained in the class and it is up to us later
        to assign each object is own values or words to each attribute
        """

        self.H0 = H0
        # H0 is given in km/s/Mpc
        self.Omega_m = Omega_m
        self.Omega_lambda = Omega_lambda

        self.Omega_k = 1 - Omega_m - Omega_lambda

        """
        "self" acts as a refrence to the attributes of a singular object in the class
        "self" allows the attributes of one specific object to be accesed instead of
        the program accesing all attributes of all objects in the class
        "self" can also be used to ammend one specific attribute of an object in the class
        """

    def integrand(self, z):
        """
        Define a method which calculates the integrand using parameters stored in self, and returns it
        """

        return (
            self.Omega_m * (1 + z) ** 3
            + self.Omega_k * (1 + z) ** 2
            + self.Omega_lambda
        ) ** (-0.5)

    def rectangle_rule(self, z, n):
        """
        Method will use rectangle rule to approximate the integral
        """

        dz = z / n

        z_vals = np.arange(n) * dz

        return np.sum(self.integrand(z_vals)) * dz * (c / self.H0)

    def trapezoid_rule(self, z, n):
        """This method will approximate the integral using the trapezoid rules"""

        dz = z / (n - 1)

        z_vals = np.linspace(0, z, n)
        f_vals = self.integrand(z_vals)
        return (
            (f_vals[0] + 2 * np.sum(f_vals[1:-1]) + f_vals[-1])
            * (dz / 2)
            * (c / self.H0)
        )

    def simpsons_rule(self, z, n):
        """
        Two diffrent arrays, each element is the integrand evaluated at a step in z.
        One array for the odd and even values of z step.
        """
        dz = z / n

        z_step = np.linspace(0, z, n + 1)

        sum_1 = np.sum(self.integrand(z_step[1:n:2]))

        sum_2 = np.sum(self.integrand(z_step[2 : n - 1 : 2]))

        return (
            (
                self.integrand(z_step[0])
                + (4 * sum_1)
                + (2 * sum_2)
                + self.integrand(z_step[-1])
            )
            * (dz / 3)
            * (c / self.H0)
        )

    def vary_n(self, z, n, accurate_value, width):

        n_steps = np.arange(2, n + 1, 2)

        rectangle_evaluated = []
        trapezoid_evaluated = []
        simpsons_evaluated = []

        rect_evals = []
        trap_evals = []
        simpsons_evals = []

        # For testing n values
        """
        rectangle_met = False
        trapezoid_met = False
        simpson_met = False
        """

        for i in n_steps:

            rect_val = self.rectangle_rule(z, i)
            trap_val = self.trapezoid_rule(z, i)
            simp_val = self.simpsons_rule(z, i)

            rectangle_evaluated.append(rect_val)
            trapezoid_evaluated.append(trap_val)
            simpsons_evaluated.append(simp_val)

            rect_evals.append(i)
            trap_evals.append(i)
            simpsons_evals.append(i)

            # Compute fractional errors
            rect_err = abs((rect_val - accurate_value) / accurate_value)
            trap_err = abs((trap_val - accurate_value) / accurate_value)
            simp_err = abs((simp_val - accurate_value) / accurate_value)

            """
            # Check each method where fractional error = 0.0005 = 5%

            if (not simpson_met) and simp_err < 0.0005:
                print("Simpson achieves required accuracy at n =", i)
                simpson_met = True

            if (not trapezoid_met) and trap_err < 0.0005:
                print("Trapezoid rule achieves required accuracy at n =", i)
                trapezoid_met = True

            if (not rectangle_met) and rect_err < 0.0005:
                print("Rectangle rule achieves required accuracy at n =", i)
                rectangle_met = True

            # If ALL THREE have met accuracy, stop the loop
            if rectangle_met and trapezoid_met and simpson_met:
                print("All distances are within the target accuracy at n =", i)
                break
                """

        plt.figure()
        plt.plot(rect_evals, rectangle_evaluated, label="Rectangle Rule")
        plt.plot(trap_evals, trapezoid_evaluated, label="Trapezoid Rule")
        plt.plot(simpsons_evals, simpsons_evaluated, label="Simpson's Rule")
        plt.axhline(
            accurate_value + width,
            color="black",
            linestyle="--",
            label="Target distances",
        )
        plt.axhline(
            accurate_value - width,
            color="black",
            linestyle="--",
        )
        plt.ylim(3200, 3260)
        plt.xlim(0, 50)
        plt.xlabel("Number of function evaluations")
        plt.ylabel("Distance, Mpc")
        plt.legend()
        plt.title("Distance vs number of function evaluations")
        plt.show()

        rectangle_evaluated_arr = np.array(rectangle_evaluated)
        trapezoid_evaluated_arr = np.array(trapezoid_evaluated)
        simpsons_evaluated_arr = np.array(simpsons_evaluated)

        percent_accuracy_rectanlge = (
            abs(accurate_value - rectangle_evaluated_arr)
        ) / accurate_value
        percent_accuracy_trapezoid = (
            abs(accurate_value - trapezoid_evaluated_arr)
        ) / accurate_value
        percent_accuracy_simpsons = (
            abs(accurate_value - simpsons_evaluated_arr)
        ) / accurate_value

        plt.figure()
        plt.plot(rect_evals, percent_accuracy_rectanlge, label="Rectanlge Rule")
        plt.plot(trap_evals, percent_accuracy_trapezoid, label="Trapezoid Rule")
        plt.plot(simpsons_evals, percent_accuracy_simpsons, label="Simpsons Rule")
        plt.axhline(0.0005, color="black", linestyle="--", label="Target accuracy")
        plt.title(
            "Absolute uncertainty of diffrent models as a function of number of functions evaluated"
        )
        plt.ylim(-1e-6, 0.004)
        plt.xlim(0, 600)
        plt.xlabel("number of functions evaluated")
        plt.ylabel("Percentage accuracy compared to Function Evaluations")
        plt.legend()
        plt.show()

    def cumulitive(self, z_max, n):
        """
        This method will calulate the ditsance for different z values (redshits).
        The integral will be calculated using the cumulitive trapezoid rule
        """

        dz = z_max / n

        z_values = np.linspace(0, z_max, n + 1)

        evalutated_integral = self.integrand(z_values)

        step_area = (dz / 2) * (evalutated_integral[1:] + evalutated_integral[:-1])
        distances = np.zeros(n + 1)

        for i in range(1, n + 1):
            distances[i] = distances[i - 1] + (
                (dz / 2) * (evalutated_integral[i] + evalutated_integral[i - 1])
            )

        distances *= c / self.H0

        return z_values, distances

    def interpolate_distance(self, z_array, n=2000):
        """
        Takes as input the z_array and aproximates the distances for each element in the array using interpolation.
        The higher n is the more accurate our interpolation, i.e. a finer run of cumulitive method
        """

        z_array = np.array(z_array)

        z_max = np.max(z_array)

        z_values, distances = self.cumulitive(z_max, n)

        f = interp1d(z_values, distances)

        return f(z_array)

    def distance_moduli(self, z_array, n=2000):
        """
        This method calculates the distance moduli at all the redshift values in the list
        """
        z_array = np.array(z_array)

        interp1d_distances = self.interpolate_distance(z_array, n)

        omega_k = self.Omega_k

        if omega_k == 0:

            luminosity_distance = (1 + z_array) * interp1d_distances

        else:
            x = np.sqrt(abs(omega_k)) * (self.H0 * interp1d_distances / c)
            # Argument of S(x) is defined here

            if omega_k > 0:
                luminosity_distance = (
                    (1 + z_array)
                    * (c / self.H0)
                    * (1 / np.sqrt(abs(omega_k)))
                    * (np.sinh(x))
                )

            else:
                luminosity_distance = (
                    (1 + z_array)
                    * (c / self.H0)
                    * (1 / np.sqrt(abs(omega_k)))
                    * (np.sin(x))
                )

        # Prevent log10(0) by replacing with NaN (μ(0) is undefined)
        luminosity_distance = np.where(
            luminosity_distance <= 0, np.nan, luminosity_distance
        )

        return (5 * np.log10(luminosity_distance)) + 25


class Likelihood:
    def __init__(self, file):
        """
        Loads txt file containing data
        """
        data = np.loadtxt(file)

        self.z = data[:, 0]
        self.mu_obs = data[:, 1]
        self.mu_err = data[:, 2]

        """
        Takes the columns only and configures them in the instance
        """

    def abs_magnitude(self, theta, n=2000, model="free_lambda"):
        """
        This method predicts distance moduli for a universe (θ) over a range of z values (using
        a method constructed in the Cosmology class) and returns them as an array.
        """
        if model == "free_lambda":
            Omega_m, Omega_lambda, H0 = theta

        elif model == "zero_lambda":
            Omega_m, H0 = theta
            Omega_lambda = 0

        cosmo = Cosmology(H0, Omega_m, Omega_lambda)

        return cosmo.distance_moduli(self.z, n) - 19.3

    def __call__(self, theta, n=2000, model="free_lambda"):
        """
        Computes Gaussian log likelihood for given model parameters.
        """

        mu_model = self.abs_magnitude(theta, n=n, model=model)

        obs_diff = self.mu_obs - mu_model

        log_likelihood = -0.5 * np.sum((obs_diff / self.mu_err) ** 2)

        return log_likelihood


class Metropolis:
    """
    A generalized Metropolis algorithm for any log likelihood funciton.
    """

    def __init__(self, log_like, theta_ini, step_sizes, bounds, seed):
        """
        Bounds are given for each parameter so that, the MCMC does not "walk" out of allowed value i.e. negative Omega_m.
        Theta is used as a general d dimensional vector storing d parameters.
        """
        self.theta = theta_ini
        self.log_like = log_like
        self.step_sizes = np.asarray(step_sizes, dtype=float)
        self.d = self.theta.shape[0]
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)

        self.log_inital = self.evaluate(self.theta)
        self.chain = None
        self.log_trace = None
        self.accepted = None
        self.acc_rate = None
        self.n_steps = None
        self.burn_in = None
        self.thin = None

    def in_bounds(self, theta):
        """
        Simply checks if proposed theta is within given bounds.
        """
        if self.bounds is None:
            return True
        for val, (low, high) in zip(theta, self.bounds):
            if (val < low) or (val > high):
                return False
        return True

    def evaluate(self, theta):
        """
        Evaluates the log likelihood at a proposed theta vector. If theta lies outside the bounds,
        the function returns minus infinity (which corresponds to likelihood of zero in log space).
        This ensures the proposals are automatically rejected by the Metropolis acceptance rule inequality.
        """
        if not self.in_bounds(theta):

            # Returns minus infinity so that it fails the inequality (in ...method/function) and alwyas rejected
            return -np.inf
        return float(self.log_like(theta))

    def propose_theta(self):
        """
        Suggests new parameter values by taking a 'step' in each parameter space.
        """
        return self.theta + self.rng.normal(size=self.d) * self.step_sizes

    def accept(self, log_proposed, log_current, rng):
        """
        Acceptance/Rejection of theta using the Metropolis algorithm.
        """
        if log_proposed > log_current:
            return True
        return np.log(rng.random()) < (log_proposed - log_current)

    def run(self, n_steps=10000, burn_in=0, thin=1):
        """
        This method runs the Metropolis Algorithm. Thinning of samples can be controlled by thin number.
        """
        self.n_steps = int(n_steps)
        self.burn_in = int(burn_in)
        self.thin = int(thin)

        # Determines how many values we will store
        n_store = (self.n_steps + self.thin - 1) // self.thin

        self.chain = np.zeros((n_store, self.d), dtype=float)
        self.log_trace = np.zeros(n_store, dtype=float)
        self.accepted = np.zeros(n_store, dtype=float)

        accept_count = 0
        store_index = 0

        for i in range(self.n_steps):
            theta_proposed = self.propose_theta()
            log_proposed = self.evaluate(theta_proposed)

            if self.accept(log_proposed, self.log_inital, self.rng):
                self.theta = theta_proposed
                self.log_inital = log_proposed
                accept_count += 1
                accepted = True

            else:
                accepted = False

            # Stores value is ith step/suggestion is a multiple of the thinning value
            if (i % self.thin) == 0:
                self.chain[store_index] = self.theta
                self.log_trace[store_index] = self.log_inital
                self.accepted[store_index] = accepted
                store_index += 1

        self.acc_rate = accept_count / self.n_steps
        return self.chain, self.log_trace, self.acc_rate

    def get_samples(self):
        """
        Returns samples after burn in accounted for (i.e. dissmissed). Slices of burn in period while accounting for thinning.
        """
        burnt = self.burn_in // self.thin
        return self.chain[burnt:]

    def get_trace(self, remove_burn=False):
        """
        Returns stored log-likelihood values and optionally removes burn-in.
        """
        if not remove_burn:
            return self.log_trace

        burnt = self.burn_in // self.thin
        return self.log_trace[burnt:]

    def plot_trace(log_trace, burnt=0):
        """
        Plots the log-likelihood values from a Metropolis run. The plot shows where
        the burn in values are cut off.
        """

        plt.figure(figsize=(7, 4))
        plt.plot(log_trace, linewidth=1)
        if burnt > 0:
            plt.axvline(burnt, linestyle="--", linewidth=1)
        plt.xlabel("Stored theta iteration")
        plt.ylabel("log-likelihood")
        plt.title("Burn-in / convergence (log-likelihood trace)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_1D_histograms(samples, names=None, bins=50):
        """
        Where samples is an array with elements pertaining to: (Number of stored samples, number of parameters).
        """
        d = samples.shape[1]
        if names is None:
            names = [f"param {i}" for i in range(d)]

        for j in range(d):
            plt.figure(figsize=(6, 4))
            plt.hist(samples[:, j], bins=bins, density=True, alpha=0.85)
            plt.xlabel(names[j])
            plt.ylabel("Probability density")
            plt.title(f"1D distribution: {names[j]}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    def plot_2D_heatmaps(samples, i, j, names=None, bins=40):
        """
        Heatmaps over 2 parameters at a time (i vs j parameter. where i and j are parameter indicies (0,...,d-1)
        """
        if names is None:
            names = [f"param {k}" for k in range(samples.shape[1])]

        x = samples[:, i]
        y = samples[:, j]

        H, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.figure(figsize=(6, 5))

        # Transpose H to get axis correct using imshow
        plt.imshow(H.T, origin="lower", aspect="auto", extent=extent)
        plt.xlabel(names[i])
        plt.ylabel(names[j])
        plt.colorbar(label="Sample density")
        plt.title(f"2D distribution: {names[i]} vs {names[j]}")
        plt.grid(False)
        plt.tight_layout()
        plt.show()


def run_metropolis_pantheon():
    """
    When this function is called it runs the Metropolis algorithm on the pantheon data,
    plots and shows likelihood values (trace) plot, 1d histograms and 2d heatmaps.
    Burn in is set at 1,000 to ensure converagance of loglikelihood has occured.
    """

    L = Likelihood("pantheon_data.txt")

    def log_like(theta):
        # Fix n and model so metropolis algorithm expects
        # log_like(theta) as the only argument
        return L(theta, n=2000, model="free_lambda")

    theta_initial = np.array([0.33, 0.82, 71])

    step_sizes = np.array([0.03, 0.04, 0.12])

    bounds = [(1e-5, 1.0), (1e-5, 1.0), (10, 200)]

    metropolis_class = Metropolis(log_like, theta_initial, step_sizes, bounds, seed=12)

    n_steps = 80000
    burn_in = 1000
    thin = 1

    chain, log_trace, accept_rate = metropolis_class.run(n_steps, burn_in, thin)

    print(f"Acceptance rate: {accept_rate:.3f}")

    burnt = burn_in // thin
    Metropolis.plot_trace(log_trace, burnt)

    samples = metropolis_class.get_samples()

    names = [r"$\Omega_m$", r"$\Omega_\Lambda$", r"$H_0$"]

    # Summary stats for report
    means = samples.mean(axis=0)
    stds = samples.std(axis=0)
    for name, mu, sig in zip(names, means, stds):
        print(f"{name}: mean={mu:.4f}, std={sig:.4f}")

    Metropolis.plot_1D_histograms(samples, names, bins=40)

    Metropolis.plot_2D_heatmaps(samples, 1, 0, names)
    Metropolis.plot_2D_heatmaps(samples, 2, 0, names)
    Metropolis.plot_2D_heatmaps(samples, 2, 1, names)


def three_d_grid(L, Omega_m_vals, Omega_lambda_vals, H0_vals):
    """
    Builds a 3D likelihood grid, then finds the maximum likelihood and returns the value of each parameter
    at that point.
    """

    G = np.zeros((len(Omega_m_vals), len(Omega_lambda_vals), len(H0_vals)))

    # Each loop assignes a number in the i,j,k th postion within the grid, G.
    for i, Omega_m in enumerate(Omega_m_vals):
        for j, Omega_lambda in enumerate(Omega_lambda_vals):
            for k, H0 in enumerate(H0_vals):

                theta = [Omega_m, Omega_lambda, H0]

                G[i, j, k] = L(theta, n=2000, model="free_lambda")

    # Find the ijk location of the maximum value of the grid
    best_i, best_j, best_k = np.unravel_index(np.argmax(G), G.shape)

    best_omega_m = Omega_m_vals[best_i]
    best_omega_lambda = Omega_lambda_vals[best_j]
    best_H0 = H0_vals[best_k]

    print("logL min/max:", np.min(G), np.max(G))
    print("Best parameters:", best_omega_m, best_omega_lambda, best_H0)

    return G, best_omega_m, best_omega_lambda, best_H0


def marginalised(G):
    """
    This function uses the 3D grid from earlier, then marginalizes over one parameter at a time.
    Then goes on to marginalize over two parameters at a time. Resuluting in 3 2D grids and 3 1D grids,
    which are all returned.
    """
    G_max = np.max(G)
    P_arg = np.exp(G - G_max)

    # Marginalization over 1 parameter
    P_lambda_H = P_arg.sum(axis=0)
    P_m_H = P_arg.sum(axis=1)
    P_m_lambda = P_arg.sum(axis=2)

    # Marginilization over 2 paramters
    P_m = P_arg.sum(axis=(1, 2))
    P_lambda = P_arg.sum(axis=(0, 2))
    P_H = P_arg.sum(axis=(0, 1))

    return P_arg, P_lambda_H, P_m_H, P_m_lambda, P_m, P_lambda, P_H


def grid_visualisation(
    Omega_m_vals,
    Omega_lambda_vals,
    H0_vals,
    P_lambda_H,
    P_m_H,
    P_m_lambda,
    P_m,
    P_lambda,
    P_H,
):
    """
    This function creates 2D marginal plots of (Omega_lambda vs H0), (Omega_m vs H0), (Omega_m vs Omega_lambda) as heat maps
    and 1D marginalized plots of Omega_m, Omega_lambda, H0 as a line graphs
    """
    # Omega_lambda vs H0
    plt.figure(figsize=(6, 5))
    plt.imshow(
        P_lambda_H,
        origin="lower",
        aspect="auto",
        extent=[
            H0_vals.min(),
            H0_vals.max(),
            Omega_lambda_vals.min(),
            Omega_lambda_vals.max(),
        ],
    )
    plt.colorbar(label="Marginalised probability")
    plt.xlabel(r"$H_0$")
    plt.ylabel(r"$\Omega_\Lambda$")
    plt.title(r"Marginalised: $\Omega_\Lambda$ vs $H_0$ (sum over $\Omega_m$)")
    plt.tight_layout()
    plt.show()

    # Omega_m vs H0
    plt.figure(figsize=(6, 5))
    plt.imshow(
        P_m_H,
        origin="lower",
        aspect="auto",
        extent=[H0_vals.min(), H0_vals.max(), Omega_m_vals.min(), Omega_m_vals.max()],
    )
    plt.colorbar(label="Marginalised probability")
    plt.xlabel(r"$H_0$")
    plt.ylabel(r"$\Omega_m$")
    plt.title(r"Marginalised: $\Omega_m$ vs $H_0$ (sum over $\Omega_\Lambda$)")
    plt.tight_layout()
    plt.show()

    # Omega_m vs Omega_lambda
    plt.figure(figsize=(6, 5))
    plt.imshow(
        P_m_lambda,
        origin="lower",
        aspect="auto",
        extent=[
            Omega_lambda_vals.min(),
            Omega_lambda_vals.max(),
            Omega_m_vals.min(),
            Omega_m_vals.max(),
        ],
    )
    plt.colorbar(label="Marginalised probability")
    plt.xlabel(r"$\Omega_\Lambda$")
    plt.ylabel(r"$\Omega_m$")
    plt.title(r"Marginalised: $\Omega_m$ vs $\Omega_\Lambda$ (sum over $H_0$)")
    plt.tight_layout()
    plt.show()

    # 1D Marginalised plots
    plt.figure(figsize=(6, 4))
    plt.plot(Omega_m_vals, P_m)
    plt.xlabel(r"$\Omega_m$")
    plt.ylabel("Marginalised probability")
    plt.title(r"1D marginal: $\Omega_m$ (sum over $\Omega_\Lambda, H_0$)")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(Omega_lambda_vals, P_lambda)
    plt.xlabel(r"$\Omega_\Lambda$")
    plt.ylabel("Marginalised probability")
    plt.title(r"1D marginal: $\Omega_\Lambda$ (sum over $\Omega_m, H_0$)")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(H0_vals, P_H)
    plt.xlabel(r"$H_0$")
    plt.ylabel("Marginalised probability")
    plt.title(r"1D marginal: $H_0$ (sum over $\Omega_m, \Omega_\Lambda$)")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.show()


def main():
    """
    Executes full parameter analysis using both the grid and metropolis methods on the pantheon data set.
    """

    Omega_m_vals = np.linspace(0.275, 0.425, 40)
    Omega_lambda_vals = np.linspace(0.7, 0.95, 40)
    H0_vals = np.linspace(71.5, 72.8, 40)

    L = Likelihood("pantheon_data.txt")
    G, best_omega_m, best_omega_lambda, best_H0 = three_d_grid(
        L, Omega_m_vals, Omega_lambda_vals, H0_vals
    )
    P_arg, P_lambda_H, P_m_H, P_m_lambda, P_m, P_lambda, P_H = marginalised(G)
    grid_visualisation(
        Omega_m_vals,
        Omega_lambda_vals,
        H0_vals,
        P_lambda_H,
        P_m_H,
        P_m_lambda,
        P_m,
        P_lambda,
        P_H,
    )

    run_metropolis_pantheon()


if __name__ == "__main__":
    main()
