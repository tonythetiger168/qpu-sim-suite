"""
QuantumSimulator - A complete state-vector quantum simulator using only NumPy.

Supports:
  - Single-qubit gates: H, X, Y, Z, S, T, RX, RY, RZ, U3
  - Two-qubit gates: CNOT, CZ, SWAP
  - Three-qubit gates: Toffoli (CCNOT)
  - Measurement: single-qubit projective, multi-shot, all-qubit
  - Analysis: probabilities, expectation values, entanglement entropy, fidelity
  - Pure NumPy, no external dependencies

Usage:
    from quantum_simulator import QuantumSimulator
    sim = QuantumSimulator(2)
    sim.h(0)
    sim.cnot(0, 1)
    sim.print_state()
    counts = sim.measure_all(shots=1024)
"""

import numpy as np


class QuantumSimulator:
    """State-vector quantum simulator."""

    def __init__(self, n_qubits):
        self.n = n_qubits
        self.N = 2 ** n_qubits
        self.state = np.zeros(self.N, dtype=complex)
        self.state[0] = 1.0

    # --- Single-qubit gates ---
    def h(self, target):
        """Hadamard gate."""
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        self._apply_1q(H, target)

    def x(self, target):
        """Pauli-X (NOT) gate."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        self._apply_1q(X, target)

    def y(self, target):
        """Pauli-Y gate."""
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self._apply_1q(Y, target)

    def z(self, target):
        """Pauli-Z gate."""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        self._apply_1q(Z, target)

    def s(self, target):
        """Phase gate (S)."""
        S = np.array([[1, 0], [0, 1j]], dtype=complex)
        self._apply_1q(S, target)

    def t(self, target):
        """T gate (pi/8)."""
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
        self._apply_1q(T, target)

    def rx(self, theta, target):
        """Rotation around X axis."""
        gate = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                         [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
        self._apply_1q(gate, target)

    def ry(self, theta, target):
        """Rotation around Y axis."""
        gate = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                         [np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
        self._apply_1q(gate, target)

    def rz(self, theta, target):
        """Rotation around Z axis."""
        gate = np.array([[np.exp(-1j*theta/2), 0],
                         [0, np.exp(1j*theta/2)]], dtype=complex)
        self._apply_1q(gate, target)

    def u3(self, theta, phi, lam, target):
        """Universal single-qubit rotation U3."""
        gate = np.array([
            [np.cos(theta/2), -np.exp(1j*lam)*np.sin(theta/2)],
            [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+lam))*np.cos(theta/2)]
        ], dtype=complex)
        self._apply_1q(gate, target)

    # --- Two-qubit gates ---
    def cnot(self, control, target):
        """Controlled-NOT gate."""
        gate = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        self._apply_2q(gate, control, target)

    def cz(self, control, target):
        """Controlled-Z gate."""
        gate = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=complex)
        self._apply_2q(gate, control, target)

    def swap(self, q1, q2):
        """SWAP gate."""
        gate = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        self._apply_2q(gate, q1, q2)

    # --- Three-qubit gates ---
    def toffoli(self, c1, c2, target):
        """Toffoli (CCNOT) gate."""
        gate = np.eye(8, dtype=complex)
        gate[6, 6] = 0; gate[6, 7] = 1
        gate[7, 6] = 1; gate[7, 7] = 0
        self._apply_3q(gate, c1, c2, target)

    # --- Core apply methods ---
    def _apply_1q(self, gate, target):
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

    def _apply_2q(self, gate, q1, q2):
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

    def _apply_3q(self, gate, q1, q2, q3):
        I = np.eye(2, dtype=complex)
        ops = [I] * self.n
        ops[q1] = None; ops[q2] = None; ops[q3] = None
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
    def measure(self, target, shots=1):
        """Projective measurement on target qubit. Returns 0 or 1."""
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

    def measure_all(self, shots=1024):
        """Measure all qubits, return histogram dictionary."""
        probs = np.abs(self.state)**2
        outcomes = np.random.choice(self.N, size=shots, p=probs)
        counts = {}
        for o in outcomes:
            bitstr = format(o, f'0{self.n}b')
            counts[bitstr] = counts.get(bitstr, 0) + 1
        return counts

    # --- Analysis tools ---
    def get_probabilities(self):
        return np.abs(self.state)**2

    def get_expectation(self, pauli_string):
        """Measure expectation value of a Pauli string, e.g. 'ZIZ'."""
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

    def entanglement_entropy(self, qubits_A):
        """Von Neumann entropy of reduced density matrix for subsystem A."""
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

    def fidelity(self, other_state):
        return np.abs(np.vdot(self.state, other_state))**2

    def print_state(self, max_terms=8):
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


# ============================================================
# EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    print("Quantum Simulator - Example Usage")
    print("=" * 50)

    # Bell State
    sim = QuantumSimulator(2)
    sim.h(0)
    sim.cnot(0, 1)
    sim.print_state()
    counts = sim.measure_all(shots=1000)
    print(f"Measurement: {counts}")

    # GHZ State with entanglement entropy
    sim = QuantumSimulator(3)
    sim.h(0)
    sim.cnot(0, 1)
    sim.cnot(1, 2)
    S = sim.entanglement_entropy([0])
    print(f"\nEntanglement entropy: {S:.4f} bits")
