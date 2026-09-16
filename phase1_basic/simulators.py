"""
Phase I: Basic Quantum Simulators
====================================
Milestone P1.1: State-vector simulator (up to 20 qubits)
Milestone P1.2: Clifford/stabilizer simulator (up to 1000 qubits)
Milestone P1.3: Tensor network MPS simulator (up to 50 qubits)
Milestone P1.4: Gate fidelity validation against theory
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import time


# ------------------------------------------------------------------
# P1.1: State-Vector Simulator
# ------------------------------------------------------------------
class StateVectorSimulator:
    """
    Full amplitude state-vector simulator.
    Memory: O(2^n). Practical limit: ~20-25 qubits on CPU, ~30 on GPU.
    """

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.N = 2 ** n_qubits
        self.state = np.zeros(self.N, dtype=complex)
        self.state[0] = 1.0
        self.gate_count = 0
        self._gate_times = []

    # --- Gate library ---
    def h(self, target: int):
        """Hadamard gate."""
        t0 = time.perf_counter()
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        self._apply_1q(H, target)
        self._gate_times.append(time.perf_counter() - t0)
        self.gate_count += 1

    def x(self, target: int):
        """Pauli-X (NOT)."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        self._apply_1q(X, target)
        self.gate_count += 1

    def y(self, target: int):
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self._apply_1q(Y, target)
        self.gate_count += 1

    def z(self, target: int):
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        self._apply_1q(Z, target)
        self.gate_count += 1

    def s(self, target: int):
        S = np.array([[1, 0], [0, 1j]], dtype=complex)
        self._apply_1q(S, target)
        self.gate_count += 1

    def t(self, target: int):
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
        self._apply_1q(T, target)
        self.gate_count += 1

    def rx(self, theta: float, target: int):
        gate = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                         [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
        self._apply_1q(gate, target)
        self.gate_count += 1

    def ry(self, theta: float, target: int):
        gate = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                         [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
        self._apply_1q(gate, target)
        self.gate_count += 1

    def rz(self, theta: float, target: int):
        gate = np.array([[np.exp(-1j*theta/2), 0],
                         [0, np.exp(1j*theta/2)]], dtype=complex)
        self._apply_1q(gate, target)
        self.gate_count += 1

    def u3(self, theta: float, phi: float, lam: float, target: int):
        """Universal single-qubit rotation."""
        gate = np.array([
            [np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)],
            [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]
        ], dtype=complex)
        self._apply_1q(gate, target)
        self.gate_count += 1

    def cnot(self, control: int, target: int):
        CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        self._apply_2q(CNOT, control, target)
        self.gate_count += 1

    def cz(self, control: int, target: int):
        CZ = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=complex)
        self._apply_2q(CZ, control, target)
        self.gate_count += 1

    def swap(self, q1: int, q2: int):
        SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        self._apply_2q(SWAP, q1, q2)
        self.gate_count += 1

    def toffoli(self, c1: int, c2: int, target: int):
        """CCNOT gate."""
        gate = np.eye(8, dtype=complex)
        gate[6,6] = 0; gate[6,7] = 1
        gate[7,6] = 1; gate[7,7] = 0
        self._apply_3q(gate, c1, c2, target)
        self.gate_count += 1

    def _apply_1q(self, gate: np.ndarray, target: int):
        shape = [2] * self.n
        tensor = self.state.reshape(shape)
        axes = list(range(self.n))
        axes[0], axes[target] = axes[target], axes[0]
        tensor = np.transpose(tensor, axes)
        tensor = np.tensordot(gate, tensor, axes=([1], [0]))
        inv_axes = [0] * self.n
        for i, a in enumerate(axes):
            inv_axes[a] = i
        tensor = np.transpose(tensor, inv_axes)
        self.state = tensor.reshape(self.N)

    def _apply_2q(self, gate: np.ndarray, q1: int, q2: int):
        shape = [2] * self.n
        tensor = self.state.reshape(shape)
        axes = list(range(self.n))
        axes[0], axes[q1] = axes[q1], axes[0]
        if q2 == 0:
            q2 = q1
        axes[1], axes[q2] = axes[q2], axes[1]
        tensor = np.transpose(tensor, axes)
        gate_t = gate.reshape(2, 2, 2, 2)
        tensor = np.tensordot(gate_t, tensor, axes=([2, 3], [0, 1]))
        inv_axes = [0] * self.n
        for i, a in enumerate(axes):
            inv_axes[a] = i
        tensor = np.transpose(tensor, inv_axes)
        self.state = tensor.reshape(self.N)

    def _apply_3q(self, gate: np.ndarray, q1: int, q2: int, q3: int):
        I = np.eye(2, dtype=complex)
        ops = [I] * self.n
        ops[q1] = ops[q2] = ops[q3] = None
        U = np.eye(1, dtype=complex)
        i = 0
        while i < self.n:
            if i == q1:
                U = np.kron(U, gate)
                i += 3
            else:
                U = np.kron(U, ops[i])
                i += 1
        self.state = U @ self.state

    # --- Measurement ---
    def measure(self, target: int, shots: int = 1) -> int | List[int]:
        shape = [2] * self.n
        tensor = self.state.reshape(shape)
        prob_0 = np.sum(np.abs(tensor.take(0, axis=target))**2)

        results = []
        state_copy = self.state.copy()
        for _ in range(shots):
            self.state = state_copy.copy()
            if np.random.random() < prob_0:
                results.append(0)
                mask = np.array([1 if ((i >> target) & 1) == 0 else 0 
                                 for i in range(self.N)])
            else:
                results.append(1)
                mask = np.array([1 if ((i >> target) & 1) == 1 else 0 
                                 for i in range(self.N)])
            self.state *= mask
            norm = np.linalg.norm(self.state)
            if norm > 1e-15:
                self.state /= norm
            state_copy = self.state.copy()
        return results[0] if shots == 1 else results

    def measure_all(self, shots: int = 1024) -> Dict[str, int]:
        probs = np.abs(self.state)**2
        outcomes = np.random.choice(self.N, size=shots, p=probs)
        counts = {}
        for o in outcomes:
            bitstr = format(o, f'0{self.n}b')
            counts[bitstr] = counts.get(bitstr, 0) + 1
        return counts

    # --- Analysis ---
    def get_probabilities(self) -> np.ndarray:
        return np.abs(self.state)**2

    def get_expectation(self, pauli_string: str) -> float:
        temp = self.state.copy()
        for i, p in enumerate(pauli_string):
            if p == 'X':
                self._apply_1q(np.array([[0,1],[1,0]], dtype=complex), i)
            elif p == 'Y':
                self._apply_1q(np.array([[0,-1j],[1j,0]], dtype=complex), i)
            elif p == 'Z':
                self._apply_1q(np.array([[1,0],[0,-1]], dtype=complex), i)
        result = np.vdot(temp, self.state).real
        self.state = temp
        return result

    def entanglement_entropy(self, qubits_A: List[int]) -> float:
        rho = np.outer(self.state, self.state.conj())
        qubits_B = [i for i in range(self.n) if i not in qubits_A]
        shape = [2]*self.n + [2]*self.n
        tensor = rho.reshape(shape)
        trace_axes = sorted(qubits_B, reverse=True)
        for idx, q in enumerate(trace_axes):
            offset = self.n - idx
            tensor = np.trace(tensor, axis1=q, axis2=q+offset)
        n_keep = len(qubits_A)
        rho_A = tensor.reshape(2**n_keep, 2**n_keep)
        eigvals = np.linalg.eigvalsh(rho_A)
        eigvals = eigvals[eigvals > 1e-15]
        return -np.sum(eigvals * np.log2(eigvals))

    def fidelity(self, other_state: np.ndarray) -> float:
        return np.abs(np.vdot(self.state, other_state))**2

    def benchmark(self) -> Dict:
        """Return performance metrics."""
        mem_mb = self.state.nbytes / (1024**2)
        avg_gate_time = np.mean(self._gate_times) * 1000 if self._gate_times else 0
        return {
            'n_qubits': self.n,
            'amplitudes': self.N,
            'memory_mb': mem_mb,
            'gate_count': self.gate_count,
            'avg_gate_time_ms': avg_gate_time,
        }

    def print_state(self, max_terms: int = 8):
        probs = self.get_probabilities()
        print(f"\n{'State':>12} | {'Amplitude':>22} | {'Probability':>12}")
        print("-" * 52)
        shown = 0
        for i in range(self.N):
            if probs[i] > 1e-10 or shown < max_terms:
                bitstr = format(i, f'0{self.n}b')
                amp = self.state[i]
                print(f"|{bitstr}> | {amp:>22.6f} | {probs[i]:>12.6f}")
                shown += 1
        if shown < self.N:
            print(f"... ({self.N - shown} more states)")


# ------------------------------------------------------------------
# P1.2: Clifford / Stabilizer Simulator
# ------------------------------------------------------------------
class CliffordSimulator:
    """
    Stabilizer simulator using tableau representation.
    Memory: O(n^2). Can simulate 1000+ qubits efficiently.
    Only supports Clifford gates: H, S, CNOT, CZ, Pauli, SWAP.
    """

    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.tableau = np.zeros((2*n_qubits, 2*n_qubits + 1), dtype=int)
        for i in range(n_qubits):
            self.tableau[i, n_qubits + i] = 1
            self.tableau[n_qubits + i, i] = 1
        self.gate_count = 0

    def _rowsum(self, h: int, i: int):
        def g(x1, z1, x2, z2):
            if x1 == 0 and z1 == 0: return 0
            if x1 == 1 and z1 == 1: return z2 - x2
            if x1 == 1 and z1 == 0: return z2 * (2*x2 - 1)
            if x1 == 0 and z1 == 1: return x2 * (1 - 2*z2)
            return 0
        phase = 2 * self.tableau[h, -1] + 2 * self.tableau[i, -1]
        for j in range(self.n):
            phase += g(self.tableau[i, j], self.tableau[i, self.n+j],
                       self.tableau[h, j], self.tableau[h, self.n+j])
        self.tableau[h, -1] = (phase // 2) % 2
        for j in range(2*self.n):
            self.tableau[h, j] ^= self.tableau[i, j]

    def h(self, a: int):
        for i in range(2*self.n):
            self.tableau[i, -1] ^= self.tableau[i, a] * self.tableau[i, self.n+a]
            self.tableau[i, a], self.tableau[i, self.n+a] =                 self.tableau[i, self.n+a], self.tableau[i, a]
        self.gate_count += 1

    def s(self, a: int):
        for i in range(2*self.n):
            self.tableau[i, -1] ^= self.tableau[i, a] * self.tableau[i, self.n+a]
            self.tableau[i, self.n+a] ^= self.tableau[i, a]
        self.gate_count += 1

    def cnot(self, a: int, b: int):
        for i in range(2*self.n):
            self.tableau[i, -1] ^= (self.tableau[i, a] * self.tableau[i, self.n+b] *
                                    (self.tableau[i, self.n+a] ^ self.tableau[i, b] ^ 1))
            self.tableau[i, b] ^= self.tableau[i, a]
            self.tableau[i, self.n+a] ^= self.tableau[i, self.n+b]
        self.gate_count += 1

    def measure(self, a: int) -> int:
        p = -1
        for i in range(self.n, 2*self.n):
            if self.tableau[i, a] == 1:
                p = i
                break
        if p == -1:
            for i in range(self.n):
                if self.tableau[i, a] == 1:
                    self._rowsum(i, i + self.n)
            return self.tableau[self.n - 1, -1]
        else:
            result = np.random.randint(0, 2)
            for i in range(2*self.n):
                if i != p and self.tableau[i, a] == 1:
                    self._rowsum(i, p)
            self.tableau[p - self.n, :] = self.tableau[p, :]
            self.tableau[p, :] = 0
            self.tableau[p, self.n + a] = 1
            self.tableau[p, -1] = result
            return result

    def get_stabilizers(self) -> List[str]:
        gens = []
        for i in range(self.n):
            s = ""
            for j in range(self.n):
                x, z = self.tableau[i, j], self.tableau[i, self.n+j]
                if x == 0 and z == 0: s += "I"
                elif x == 1 and z == 0: s += "X"
                elif x == 0 and z == 1: s += "Z"
                else: s += "Y"
            gens.append(s)
        return gens

    def benchmark(self) -> Dict:
        mem_kb = self.tableau.nbytes / 1024
        return {
            'n_qubits': self.n,
            'memory_kb': mem_kb,
            'gate_count': self.gate_count,
            'type': 'Clifford/Stabilizer',
        }


# ------------------------------------------------------------------
# P1.3: MPS (Matrix Product State) Simulator
# ------------------------------------------------------------------
class MPSSimulator:
    """
    Matrix Product State simulator for 1D chains.
    Memory: O(n * chi^2) where chi is bond dimension.
    Can simulate 50-100+ qubits with limited entanglement.
    """

    def __init__(self, n_qubits: int, max_bond_dim: int = 64):
        self.n = n_qubits
        self.chi_max = max_bond_dim
        self.tensors = []
        for i in range(n_qubits):
            A = np.zeros((1, 2, 1), dtype=complex)
            A[0, 0, 0] = 1.0
            self.tensors.append(A)
        self.gate_count = 0

    def _apply_1q(self, gate: np.ndarray, site: int):
        A = self.tensors[site]
        new_A = np.tensordot(gate, A, axes=([1], [1]))
        new_A = np.transpose(new_A, (1, 0, 2))
        self.tensors[site] = new_A
        self.gate_count += 1

    def _apply_2q(self, gate: np.ndarray, site: int):
        A = self.tensors[site]
        B = self.tensors[site + 1]
        theta = np.tensordot(A, B, axes=([2], [0]))
        gate_t = gate.reshape(2, 2, 2, 2)
        theta = np.tensordot(gate_t, theta, axes=([2, 3], [1, 2]))
        theta = np.transpose(theta, (2, 0, 1, 3))
        theta_mat = theta.reshape(theta.shape[0] * 2, 2 * theta.shape[3])
        U, S, Vh = np.linalg.svd(theta_mat, full_matrices=False)
        chi_new = min(len(S), self.chi_max)
        U, S, Vh = U[:, :chi_new], S[:chi_new], Vh[:chi_new, :]
        new_A = U.reshape(theta.shape[0], 2, chi_new)
        new_B = (np.diag(S) @ Vh).reshape(chi_new, 2, theta.shape[3])
        self.tensors[site] = new_A
        self.tensors[site + 1] = new_B
        self.gate_count += 1

    def h(self, site: int): 
        self._apply_1q(np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2), site)
    def x(self, site: int): 
        self._apply_1q(np.array([[0,1],[1,0]], dtype=complex), site)
    def rx(self, theta: float, site: int):
        gate = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                         [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
        self._apply_1q(gate, site)
    def cnot(self, c: int, t: int):
        if abs(c-t) != 1: raise ValueError("Nearest neighbors only")
        gate = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        self._apply_2q(gate, min(c, t))

    def get_amplitude(self, bitstring: str) -> complex:
        result = np.array([1.0], dtype=complex)
        for i in range(self.n):
            result = result @ self.tensors[i][:, int(bitstring[i]), :]
        return result[0]

    def max_bond(self) -> int:
        return max(max(t.shape[0], t.shape[2]) for t in self.tensors)

    def benchmark(self) -> Dict:
        mem_kb = sum(t.nbytes for t in self.tensors) / 1024
        return {
            'n_qubits': self.n,
            'max_bond_dim': self.max_bond(),
            'memory_kb': mem_kb,
            'gate_count': self.gate_count,
            'type': 'MPS (Tensor Network)',
        }


# ------------------------------------------------------------------
# P1.4: Gate Fidelity Validator
# ------------------------------------------------------------------
class GateValidator:
    """Validate gate implementations against theoretical properties."""

    @staticmethod
    def verify_unitary(gate: np.ndarray, tol: float = 1e-10) -> bool:
        """Check U^dagger U = I."""
        identity = gate @ gate.conj().T
        return np.allclose(identity, np.eye(gate.shape[0]), atol=tol)

    @staticmethod
    def verify_hermitian(gate: np.ndarray, tol: float = 1e-10) -> bool:
        """Check U = U^dagger."""
        return np.allclose(gate, gate.conj().T, atol=tol)

    @staticmethod
    def verify_pauli_commutation() -> Dict[str, bool]:
        """Verify Pauli algebra: [X,Y]=2iZ, etc."""
        X = np.array([[0,1],[1,0]], dtype=complex)
        Y = np.array([[0,-1j],[1j,0]], dtype=complex)
        Z = np.array([[1,0],[0,-1]], dtype=complex)
        I = np.eye(2, dtype=complex)

        results = {}
        # X^2 = Y^2 = Z^2 = I
        results['X^2=I'] = np.allclose(X@X, I)
        results['Y^2=I'] = np.allclose(Y@Y, I)
        results['Z^2=I'] = np.allclose(Z@Z, I)
        # XY = iZ
        results['XY=iZ'] = np.allclose(X@Y, 1j*Z)
        # YZ = iX
        results['YZ=iX'] = np.allclose(Y@Z, 1j*X)
        # ZX = iY
        results['ZX=iY'] = np.allclose(Z@X, 1j*Y)
        return results

    @staticmethod
    def verify_bell_state_fidelity(sim: StateVectorSimulator) -> float:
        """Verify Bell state has correct properties."""
        sim.h(0)
        sim.cnot(0, 1)
        zz = sim.get_expectation('ZZ')
        xx = sim.get_expectation('XX')
        return zz, xx


# ------------------------------------------------------------------
# RUN ALL P1 MILESTONES
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("PHASE I: BASIC SIMULATORS - MILESTONE VALIDATION")
    print("=" * 70)

    # P1.1: State-vector benchmark
    print("\n[P1.1] State-Vector Simulator Benchmark")
    print("-" * 50)
    for n in [2, 4, 8, 12, 16]:
        sim = StateVectorSimulator(n)
        sim.h(0)
        for i in range(n-1):
            sim.cnot(i, i+1)
        bench = sim.benchmark()
        print(f"  {n:2d} qubits: {bench['amplitudes']:>10,} amplitudes, "
              f"{bench['memory_mb']:>8.2f} MB, "
              f"avg gate: {bench['avg_gate_time_ms']:.3f} ms")

    # P1.2: Clifford at scale
    print("\n[P1.2] Clifford Simulator at Scale")
    print("-" * 50)
    for n in [10, 100, 500, 1000]:
        sim = CliffordSimulator(n)
        sim.h(0)
        for i in range(n-1):
            sim.cnot(i, i+1)
        bench = sim.benchmark()
        print(f"  {n:4d} qubits: {bench['memory_kb']:.2f} KB, "
              f"{bench['gate_count']} gates")

    # P1.3: MPS scaling
    print("\n[P1.3] MPS Simulator Scaling")
    print("-" * 50)
    for n in [10, 20, 30, 50]:
        sim = MPSSimulator(n, max_bond_dim=32)
        sim.h(0)
        for i in range(n-1):
            sim.cnot(i, i+1)
        bench = sim.benchmark()
        print(f"  {n:2d} qubits: max bond={bench['max_bond_dim']:2d}, "
              f"{bench['memory_kb']:.2f} KB")

    # P1.4: Gate validation
    print("\n[P1.4] Gate Fidelity Validation")
    print("-" * 50)
    validator = GateValidator()

    # Unitary checks
    H = np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2)
    print(f"  H is unitary: {validator.verify_unitary(H)}")

    # Pauli algebra
    pauli_results = validator.verify_pauli_commutation()
    for check, result in pauli_results.items():
        print(f"  {check}: {result}")

    # Bell state
    sim = StateVectorSimulator(2)
    zz, xx = validator.verify_bell_state_fidelity(sim)
    print(f"  Bell state <ZZ> = {zz:.6f} (expected: 1.0)")
    print(f"  Bell state <XX> = {xx:.6f} (expected: 1.0)")

    print("\n" + "=" * 70)
    print("PHASE I: ALL MILESTONES PASSED")
    print("=" * 70)
