Cosmological Parameter Estimation with Supernova Data

This project estimates key cosmological parameters using observational supernova data from the Pantheon dataset. It implements two complementary statistical approaches to explore the likelihood surface of the cosmological model parameters:

Brute-force likelihood grid search

Metropolis Monte Carlo Markov Chain (MCMC)

The aim is to determine the most probable values and uncertainties of the parameters:

> Matter density parameter Ωₘ

> Dark energy density parameter ΩΛ

> Hubble constant H₀

This project was developed as part of Unit 5: Parameter Uncertainties in the Computer Modelling course.

**Overview**

The project models cosmological distances from supernova redshift data and evaluates how well different cosmological parameter values explain the observations.

**Two approaches are implemented:**

1️⃣ Likelihood Grid Method

A full 3D likelihood grid is constructed across ranges of:

> Ωₘ

> ΩΛ

> H₀

The likelihood is evaluated at every point in the grid and then marginalized to produce:

> 2D likelihood heatmaps

> 1D marginalized probability distributions

This provides an intuitive visualization of the parameter space.

However, the grid approach scales poorly since the number of likelihood evaluations grows exponentially with the number of parameters.

2️⃣ Metropolis MCMC Method

A generalized Metropolis algorithm is implemented to sample the parameter space efficiently.

The algorithm:

1. Proposes new parameter values by taking Gaussian steps in parameter space

2. Accepts or rejects the proposal based on the likelihood ratio

3. Generates a Markov chain whose samples follow the posterior distribution

This approach allows efficient exploration of parameter space and scales well to higher dimensions.

**Results:**

Both methods produced consistent estimates of the cosmological parameters:

Parameter	Estimated Value
Ωₘ	0.3479 ± 0.0384
ΩΛ	0.8246 ± 0.0657
H₀	72.1231 ± 0.3037

The marginalized distributions show clear peaks around these values, indicating the region of highest likelihood in parameter space.

The Metropolis sampling also reveals correlations between parameters through 2D density plots.

Project Structure
cosmological-parameter-estimation/
│
├── cosmological parameter estimation.py   # Main analysis script
├── pantheon_data.txt                      # Supernova dataset
├── Output Analysis.pdf                    # Results document with plots
└── README.md                              # Project description

The Python script contains several classes:

**Cosmology**

> Computes cosmological distances for given parameters using numerical integration methods:

Rectangle rule

Trapezoid rule

Simpson’s rule

Cumulative integration for interpolation

**Likelihood**

> Handles the supernova dataset and computes the log likelihood of a cosmological model given the observed data.

**Metropolis**

> Implements a generic Metropolis MCMC sampler capable of exploring arbitrary likelihood functions.

Features include:

> parameter bounds

> burn-in removal

> thinning

> trace plots

> 1D histograms

> 2D density heatmaps

**Visualisations Produced**

The code generates:

> Likelihood grid results

> 2D marginalized likelihood heatmaps

> 1D marginalized probability curves

> Metropolis results

> log-likelihood trace (burn-in analysis)

> parameter histograms

> 2D parameter correlation plots

These visualizations allow comparison between grid-based and MCMC-based inference.

**How to Run**

Install required packages:

pip install numpy matplotlib scipy

Run the script:

python "cosmological parameter estimation.py"

The program will:

Construct the likelihood grid

Compute marginalized distributions

Run the Metropolis sampler

Generate plots of the results

Key Concepts Demonstrated

Numerical integration in cosmology

Maximum likelihood estimation

Marginalization of multidimensional likelihoods

Monte Carlo Markov Chains

Metropolis algorithm

Parameter uncertainty estimation

**Author**

**Ali Mulholland**
**BSc Physics – University of Edinburgh**
