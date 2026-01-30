# Numerical-Experiments-for-Differentially-Private-Decentralized-Optimization-Algorithms

Benchmarking convergence rate and accuracy of decentralized optimization algorithms based on real-world and synthetic datasets.

The project introduces DP-DOML, a differentially private decentralized stochastic algorithm featuring a novel $K$-gossip skip iterative mode. This approach achieves the theoretical lower bound of Privacy Leakage Frequency (PLF) — $\Omega(\sqrt{\kappa}\log\frac{1}{\epsilon})$ — enhancing security without sacrificing convergence performance.

## 📌 Experiment Overview on Synthetic Datasets:

The core of this project is to analyze how different network structures and algorithm parameters (such as communication probability and local update steps) affect the performance of distributed solvers.

Tested Algorithms:

(DP)-DOML (Our proposed method)

(DP)-MG-ED

(DP)-OGT

(R)-ADMM

(DP)-ProxSkip

(DP)-SGD

## 🔬 Synthetic Dataset Experiments

We conduct the decentralized least squares objective on on a topology of 20 nodes.

## 📂 Main Scripts

1. **main_topology.m**:

Function: Evaluates the performance (e.g., PLF - Performance Loss Factor) of multiple algorithms across different network connectivity levels (rho).

Key variables: PER (edge probability), n=50 (nodes), d=3 (dimensions).

Output: Comparison plots showing how network structure impacts convergence across all solvers.

2. **main_convergence.m**:

Function: Specifically focuses on the impact of communication probability $p$ on PLF and Itertaion of (DP)-DOML.

Output: Log-scale convergence plots (semilogy) comparing convergence rates over PLF and iterations for different $p$ values.

## 🛠 Usage

1. Prerequisites:

MATLAB (R2020a or later recommended).

2. Running Simulations:

run('main_topology.m');  run('main_convergence.m')


## 📌 Experiment Overview on Real-world Datasets:

The core of this project is to demonstrate the practical utility of the algorithm on large-scale classification tasks.

Data Files:

**For ijcnn1**: 'ijcnn_train_processd.txt' and 'ijcnn_test_processd.txt'.

**For MNIST**: 'train-images.idx3-ubyte' and 'train-labels.idx1-ubyte'.

📢 The core implementation of the training algorithms and learning tasks is integral to our ongoing extended research. To ensure the integrity of the subsequent publication, the full source code will be made publicly available upon the completion of our follow-up work. The complete repository is coming soon.
