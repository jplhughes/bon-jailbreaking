import json
from typing import List

import numpy as np


class CMAEvolutionStrategy:
    def __init__(
        self,
        N: int,
        xmean: np.array,
        sigma: float,
        pc: np.array,
        ps: np.array,
        B: np.array,
        D: np.array,
        C: np.array,
        invsqrtC: np.array,
        population_size: int = None,
        eigeneval: int = 0,
        counteval: int = 0,
    ):
        self.N = N  # dimension of the problem (here based on the number of augmentations)
        self.xmean = xmean  # objective variables mean
        self.sigma = sigma  # coordinate wise standard deviation (step size)

        # Strategy parameter setting: Selection
        self.lambda_ = population_size or (4 + int(3 * np.log(N)))  # population size
        self.mu, self.weights, self.mueff = self.calculate_selection_parameters(self.lambda_)

        # Strategy parameter setting: Adaptation
        self.cc, self.cs, self.c1, self.cmu, self.damps = self.calculate_adaptation_parameters(self.N, self.mueff)

        # Initialize dynamic (internal) strategy parameters and constants
        self.pc = pc  # evolution path for C
        self.ps = ps  # evolution path for sigma
        self.B = B  # defines the coordinate system
        self.D = D  # diagonal D defines the scaling
        self.C = C  # covariance matrix C
        self.invsqrtC = invsqrtC  # C^-1/2

        self.eigeneval = eigeneval  # number of times eigeneval has been called
        self.counteval = counteval  # number of times counteval has been called
        self.chiN = N**0.5 * (1 - 1 / (4 * N) + 1 / (21 * N**2))  # expectation of ||N(0,I)|| == norm(randn(N,1))

    @staticmethod
    def calculate_selection_parameters(lambda_):
        """
        Args:
            lambda_ (int): Population size.

        Returns:
            tuple: A tuple containing:
                - mu (int): Number of parents to select for recombination.
                - weights (np.array): Weights for the parents.
                - mueff (float): Variance-effectiveness of sum w_i x_i.
        """
        mu = lambda_ // 2
        weights = np.log(lambda_ // 2 + 0.5) - np.log(np.arange(1, lambda_ // 2 + 1))
        weights /= np.sum(weights)
        mueff = np.sum(weights) ** 2 / np.sum(weights**2)
        return mu, weights, mueff

    @staticmethod
    def calculate_adaptation_parameters(N, mueff):
        """
        Args:
            N (int): Dimension of the problem
            mueff (float): Variance-effectiveness of sum w_i x_i.
        Returns:
            tuple: A tuple containing:
                - cc (float): Time constant for cumulation for C.
                - cs (float): T-const for cumulation for sigma control.
                - c1 (float): Learning rate for rank-one update of C.
                - cmu (float): And for rank-mu update.
                - damps (float): Damping for sigma.
        """
        cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
        cs = (mueff + 2) / (N + mueff + 5)
        c1 = 2 / ((N + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + cs
        return cc, cs, c1, cmu, damps

    @classmethod
    def from_scratch(cls, N: int = 5, sigma: float = 0.2, population_size: int = None, init_means: List[float] = None):
        xmean = np.zeros(N)
        if init_means is not None:
            xmean = np.array(init_means)
        pc = ps = np.zeros(N)
        B = np.eye(N)
        D = np.ones(N)
        C = B @ np.diag(D**2) @ B.T
        invsqrtC = B @ np.diag(1 / D) @ B.T
        return cls(N, xmean, sigma, pc, ps, B, D, C, invsqrtC, population_size)

    @classmethod
    def from_state_dict(cls, filename: str):
        with open(filename, "r") as f:
            state = json.load(f)
        return cls(
            state["N"],
            np.array(state["xmean"]),
            state["sigma"],
            np.array(state["pc"]),
            np.array(state["ps"]),
            np.array(state["B"]),
            np.array(state["D"]),
            np.array(state["C"]),
            np.array(state["invsqrtC"]),
            state["population_size"],
            state["eigeneval"],
            state["counteval"],
        )

    def sample_multivariate_normal(self) -> np.array:
        return self.xmean + self.sigma * self.B @ (self.D * np.random.randn(self.N))

    def sort_solutions_based_on_fitness(self, vectors: List[np.array], scores: List[float]) -> np.array:
        return np.array([vectors[i] for i in np.argsort(scores)[::-1]])

    def update_m(self, sorted_vectors) -> np.array:
        return np.dot(sorted_vectors[: self.mu].T, self.weights)

    def update_p_s(self, p_s, m, m_prime) -> np.array:
        return (1 - self.cs) * p_s + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * self.invsqrtC @ (
            m - m_prime
        ) / self.sigma

    def calc_hsig(self, p_s, counteval) -> bool:
        return np.linalg.norm(p_s) / np.sqrt(
            1 - (1 - self.cs) ** (2 * counteval * self.lambda_)
        ) / self.chiN < 1.4 + 2 / (self.N + 1)

    def update_p_c(self, p_c, m, m_prime, hsig) -> np.array:
        return (1 - self.cc) * p_c + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * (m - m_prime) / self.sigma

    def update_C(self, C, p_c, sorted_vectors, m_prime, hsig) -> np.array:
        artmp = (1 / self.sigma) * (np.array(sorted_vectors[: self.mu]) - m_prime)
        return (
            (1 - self.c1 - self.cmu) * C
            + self.c1 * (p_c[:, np.newaxis] @ p_c[np.newaxis, :] + (1 - hsig) * self.cc * (2 - self.cc) * C)
            + self.cmu * artmp.T @ np.diag(self.weights) @ artmp
        )

    def update_s(self, s, p_s) -> float:
        return s * np.exp((self.cs / self.damps) * (np.linalg.norm(p_s) / self.chiN - 1))

    @staticmethod
    def decompose_covariance(C):
        eigvals, eigvecs = np.linalg.eigh(C)
        return eigvecs, np.sqrt(eigvals)

    def apply_decomposition(self) -> bool:
        return self.counteval - self.eigeneval >= self.lambda_ / (self.c1 + self.cmu) / self.N / 10

    def update_decomposition(self, C, B, D, invsqrtC):
        C = (C + C.T) / 2
        B, D = self.decompose_covariance(C)
        invsqrtC = B @ np.diag(1 / D) @ B.T
        return C, B, D, invsqrtC

    def run_step(self, vectors: List[np.array], scores: List[float]):
        self.counteval += self.lambda_

        sorted_vectors = self.sort_solutions_based_on_fitness(vectors, scores)
        m_prime = self.xmean
        self.xmean = self.update_m(sorted_vectors)
        self.ps = self.update_p_s(self.ps, self.xmean, m_prime)
        hsig = self.calc_hsig(self.ps, self.counteval)
        self.pc = self.update_p_c(self.pc, self.xmean, m_prime, hsig)
        self.C = self.update_C(self.C, self.pc, sorted_vectors, m_prime, hsig)
        self.sigma = self.update_s(self.sigma, self.ps)
        if self.apply_decomposition():
            self.eigeneval = self.counteval
            self.C, self.B, self.D, self.invsqrtC = self.update_decomposition(self.C, self.B, self.D, self.invsqrtC)

    def get_vectors(self) -> List[np.array]:
        return [self.sample_multivariate_normal() for _ in range(self.lambda_)]

    def save_state(self, filepath):
        state = {
            "N": self.N,
            "xmean": self.xmean.tolist(),
            "sigma": self.sigma,
            "population_size": self.lambda_,
            "pc": self.pc.tolist(),
            "ps": self.ps.tolist(),
            "C": self.C.tolist(),
            "B": self.B.tolist(),
            "D": self.D.tolist(),
            "invsqrtC": self.invsqrtC.tolist(),
            "eigeneval": self.eigeneval,
            "counteval": self.counteval,
        }
        with open(filepath, "w") as f:
            json.dump(state, f)
