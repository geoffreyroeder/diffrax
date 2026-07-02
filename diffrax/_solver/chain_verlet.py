from collections.abc import Callable
from typing import ClassVar
from typing_extensions import TypeAlias

import jax.numpy as jnp
import jax.scipy.linalg
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import ArrayLike, Float, PyTree

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm
from .base import AbstractSolver


_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None

Ya: TypeAlias = PyTree[Float[ArrayLike, "?*y"], " Y"]


def chain_matrix(coefficients: tuple) -> jnp.ndarray:
    r"""The $K \times K$ chain-coupling matrix $C$ of the order-$K$ Langevin
    hierarchy: $C_{0,1} = c_0$; $C_{i,i+1} = c_i$, $C_{i+1,i} = -c_i$ for
    $i \ge 1$; zero elsewhere (in particular $C_{1,0} = 0$: the force enters
    through the kick, not the linear flow). Traceless, so its flow
    $e^{hC}$ has unit determinant; flip-antisymmetric under the alternating
    flip $F = \mathrm{diag}(+1, -1, +1, \ldots)$, i.e. $FCF = -C$, which is
    what makes the palindromic step reversible.
    """
    order = len(coefficients) + 1
    C = jnp.zeros((order, order))
    C = C.at[0, 1].set(coefficients[0])
    for i in range(1, order - 1):
        C = C.at[i, i + 1].set(coefficients[i])
        C = C.at[i + 1, i].set(-coefficients[i])
    return C


class ChainVerlet(AbstractSolver):
    r"""Palindromic splitting for the order-$K$ Langevin chain hierarchy.

    The state is a 2-tuple `(x, momenta)` with `momenta` a tuple of $K - 1$
    momentum blocks, each a pytree like `x`. Over a step of size
    $h = t_1 - t_0$ it computes

    $\mathrm{Kick}_{h/2} \circ (e^{hC} \otimes I) \circ \mathrm{Kick}_{h/2}$

    where the kick is the shear $p_1 \mathrel{+}= \frac{h}{2}\,c_0 f(t, x)$
    with $f$ the force term (for sampling, $f = \nabla\log\tilde\pi$), and
    $e^{hC}$ is the exact flow of the linear chain dynamics $z' = Cz$ on the
    blocks $(x, p_1, \ldots, p_{K-1})$, computed as a $K \times K$ matrix
    exponential. Volume-preserving ($\operatorname{tr} C = 0$ and the kick is
    a shear) and time-reversible under the alternating flip
    $F = \mathrm{diag}(+1, -1, +1, \ldots)$ of the momentum blocks — exactly
    the two properties Metropolis adjustment with a bare energy-difference
    acceptance requires (see [`diffrax.ChainLangevin`][]).

    At $K = 2$, $c_0 = 1$ the chain matrix is nilpotent, $e^{hC}$ is the
    leapfrog drift, and this solver is [`diffrax.VelocityVerlet`][] exactly
    (with the state's momentum wrapped in a 1-tuple). 2nd order method. Does
    not support adaptive step sizing.

    The term should be the single force vector field `f(t, x, args)`
    driving $p_1$; the chain couplings are constructor coefficients, not
    terms, because their flow is computed exactly rather than integrated.
    """

    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    coefficients: tuple

    def order(self, terms):
        return 2

    def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: tuple[Ya, tuple],
        args: Args,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: tuple[Ya, tuple],
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[tuple[Ya, tuple], _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        x0, momenta = y0
        order = len(momenta) + 1
        if len(self.coefficients) != order - 1:
            raise ValueError(
                f"ChainVerlet has {len(self.coefficients)} coefficients but the "
                f"state carries {len(momenta)} momentum blocks."
            )
        c0 = self.coefficients[0]
        control = terms.contr(t0, t1)
        h = t1 - t0
        flow = jax.scipy.linalg.expm(h * chain_matrix(self.coefficients))

        # First half-kick: p_1 += (h/2) c_0 f(x).
        p1_half = (
            momenta[0] ** ω + 0.5 * c0 * terms.vf_prod(t0, x0, args, control) ** ω
        ).ω
        blocks = (x0, p1_half) + momenta[1:]

        # Exact linear chain flow: block_i <- sum_j flow[i, j] block_j.
        def mix(i):
            return jtu.tree_map(
                lambda *leaves: sum(flow[i, j] * leaves[j] for j in range(order)),
                *blocks,
            )

        x1 = mix(0)
        new_momenta = tuple(mix(i) for i in range(1, order))

        # Second half-kick at the new position.
        p1_full = (
            new_momenta[0] ** ω + 0.5 * c0 * terms.vf_prod(t1, x1, args, control) ** ω
        ).ω
        y1 = (x1, (p1_full,) + new_momenta[1:])

        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: tuple[Ya, tuple],
        args: Args,
    ) -> VF:
        x0, momenta = y0
        order = len(momenta) + 1
        C = chain_matrix(self.coefficients)
        blocks = (x0,) + momenta
        linear = tuple(
            jtu.tree_map(
                lambda *leaves: sum(C[i, j] * leaves[j] for j in range(order)),
                *blocks,
            )
            for i in range(order)
        )
        force = terms.vf(t0, x0, args)
        p1_dot = (linear[1] ** ω + self.coefficients[0] * force**ω).ω
        return (linear[0], (p1_dot,) + linear[2:])


ChainVerlet.__init__.__doc__ = """**Arguments:**

- `coefficients`: The chain couplings `(c_0, ..., c_{K-2})` of the order-`K`
    hierarchy; `c_0` also scales the force in the kick. `(1.0,)` at `K = 2`
    reproduces [`diffrax.VelocityVerlet`][] exactly.
"""
