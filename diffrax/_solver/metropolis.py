r"""Metropolis-adjusted wrapped solvers, for building MCMC kernels out of
differential equation solvers.

The construction is "Metropolis-corrected simulation": simulate dynamics whose
stationary law is (an extension of) an unnormalized target density for one step
with a numerical integrator, then correct the discretization error with a
Metropolis--Hastings accept/reject step so that the discrete chain leaves the
target invariant exactly.

**PRNG key threading.** The accept/reject step (and, for [`diffrax.GHMC`][],
the momentum refresh) consumes randomness. diffrax's
`AbstractSolver.init`/`step` interface has no key parameter, so these solvers
store an initial key as a field: `init` seeds the solver state with it, and
each `step` splits the key carried in the solver state
`(inner_solver_state, key)`. This composes with
[`diffrax.diffeqsolve`][] (unlike passing a key to `init`, which would require
driving the solver manually). Alternatively `step` may be driven directly in a
manual scan loop, supplying `(inner_state, key)` yourself. Use a constant step
size: an adaptive controller that rejects and retries steps would reuse
randomness.
"""

import operator
from collections.abc import Callable
from typing import Any, Optional, TYPE_CHECKING
from typing_extensions import TypeAlias

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu


if TYPE_CHECKING:
    from typing import ClassVar as AbstractVar
else:
    from equinox import AbstractVar
from equinox.internal import ω
from jaxtyping import PRNGKeyArray, PyTree

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from .._solution import RESULTS
from .._term import AbstractTerm
from .base import AbstractSolver, AbstractWrappedSolver


_SolverState: TypeAlias = tuple[Any, PRNGKeyArray]


def _sum_squares(tree: PyTree) -> RealScalarLike:
    return jtu.tree_reduce(
        operator.add, jtu.tree_map(lambda z: jnp.sum(jnp.square(z)), tree)
    )


def _normal_like(key: PRNGKeyArray, tree: PyTree) -> PyTree:
    leaves, treedef = jtu.tree_flatten(tree)
    keys = jr.split(key, len(leaves))
    return jtu.tree_unflatten(
        treedef,
        [
            jr.normal(k, jnp.shape(leaf), jnp.result_type(leaf))
            for k, leaf in zip(keys, leaves)
        ],
    )


def _log_acceptance(log_ratio: RealScalarLike) -> RealScalarLike:
    # The acceptance probability is computed as `exp(min(0, log_ratio))`: this has
    # bitwise-identical forward values to `clip(exp(log_ratio), max=1)`, but its
    # derivative is exactly 0 in the saturated branch, instead of the `0 * inf`
    # NaN that the clip-after-exp form produces under reverse-mode AD at overflow.
    # NaN energies/log-ratios (diverged proposals) are rejected, mirroring
    # blackjax's `safe_energy_diff`.
    log_ratio = jnp.where(jnp.isnan(log_ratio), -jnp.inf, log_ratio)
    return jnp.minimum(0.0, log_ratio)


def _tree_where(pred: BoolScalarLike, a: PyTree, b: PyTree) -> PyTree:
    return jtu.tree_map(lambda ai, bi: jnp.where(pred, ai, bi), a, b)


class AbstractMetropolisSolver(AbstractWrappedSolver[_SolverState]):
    """Abstract base class for Metropolis-adjusted wrapped solvers.

    Wraps an inner solver whose step produces the MCMC proposal, and threads a
    PRNG key through the solver state for the accept/reject randomness. The
    solver state is the 2-tuple `(inner_solver_state, key)`, seeded by `init`
    from the `key` field.
    """

    solver: AbstractVar[AbstractSolver]
    key: AbstractVar[PRNGKeyArray]

    @property
    def term_structure(self):
        return self.solver.term_structure

    @property
    def interpolation_cls(self):  # pyright: ignore
        return self.solver.interpolation_cls

    @property
    def term_compatible_contr_kwargs(self):
        return self.solver.term_compatible_contr_kwargs

    def order(self, terms: PyTree[AbstractTerm]) -> Optional[int]:
        return self.solver.order(terms)

    def strong_order(self, terms: PyTree[AbstractTerm]) -> Optional[RealScalarLike]:
        return self.solver.strong_order(terms)

    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _SolverState:
        return self.solver.init(terms, t0, t1, y0, args), self.key

    def func(
        self, terms: PyTree[AbstractTerm], t0: RealScalarLike, y0: Y, args: Args
    ) -> VF:
        return self.solver.func(terms, t0, y0, args)


class MetropolisAdjusted(AbstractMetropolisSolver):
    r"""Wraps another solver and Metropolis-adjusts each of its steps, using
    energy-difference acceptance.

    Each step makes a proposal $y_1$ with the wrapped solver and accepts it with
    probability $\exp(\min(0, -(E(y_1) - E(y_0))))$, where $E$ is `energy_fn`.
    On rejection the next state is `flip(y0)` (by default just $y_0$).

    This is the correct Metropolis correction when the wrapped solver's step is
    a deterministic, volume-preserving map that is an involution up to `flip`
    -- e.g. [`diffrax.VelocityVerlet`][] on Hamiltonian dynamics for the
    extended energy $E(x, p) = -\log\tilde\pi(x) + \frac{1}{2}|p|^2$, with
    `flip` the momentum flip $(x, p) \mapsto (x, -p)$: then the chain leaves
    $\tilde\pi(x)\,\mathcal{N}(p; 0, I)$ invariant. With full momentum
    refreshment between steps this is Hamiltonian Monte Carlo (Duane et al.
    1987; Neal 2011); with persistent momentum the flip on rejection is
    required for invariance (Horowitz 1991).

    See the module note in `diffrax/_solver/metropolis.py` (or
    [`diffrax.GHMC`][]) for how the PRNG key is threaded.

    ??? cite "References"

        ```bibtex
        @article{duane1987hybrid,
            title={Hybrid Monte Carlo},
            author={Duane, Simon and Kennedy, Anthony D and Pendleton, Brian J
                    and Roweth, Duncan},
            journal={Physics Letters B},
            volume={195},
            number={2},
            pages={216--222},
            year={1987},
        }

        @incollection{neal2011mcmc,
            title={{MCMC} using {H}amiltonian dynamics},
            author={Neal, Radford M},
            booktitle={Handbook of Markov Chain Monte Carlo},
            pages={113--162},
            year={2011},
            publisher={Chapman and Hall/CRC},
        }

        @article{horowitz1991generalized,
            title={A generalized guided {M}onte {C}arlo algorithm},
            author={Horowitz, Alan M},
            journal={Physics Letters B},
            volume={268},
            number={2},
            pages={247--252},
            year={1991},
        }
        ```
    """

    solver: AbstractSolver
    energy_fn: Callable[[Y, Args], RealScalarLike]
    key: PRNGKeyArray
    flip: Optional[Callable[[Y], Y]] = None

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        inner_state, key = solver_state
        y1, y_error, dense_info, inner_state, result = self.solver.step(
            terms, t0, t1, y0, args, inner_state, made_jump
        )
        if y_error is not None:
            raise NotImplementedError(
                "`MetropolisAdjusted` does not support inner solvers with error "
                "estimates."
            )

        delta_energy = self.energy_fn(y1, args) - self.energy_fn(y0, args)
        log_accept = _log_acceptance(-delta_energy)

        next_key, accept_key = jr.split(key)
        u = jr.uniform(accept_key)
        accept = u < jnp.exp(log_accept)

        y_reject = y0 if self.flip is None else self.flip(y0)
        y_next = _tree_where(accept, y1, y_reject)
        dense_info = dict(dense_info, y1=y_next)
        return y_next, None, dense_info, (inner_state, next_key), result


MetropolisAdjusted.__init__.__doc__ = """**Arguments:**

- `solver`: The solver to wrap; its step is the proposal.
- `energy_fn`: The total energy `E(y, args)` of the extended state, e.g.
    `H(x, p) = -log pi(x) + 0.5 |p|^2`. The target of the chain is
    proportional to `exp(-E)`.
- `key`: A PRNG key, used to seed the accept/reject randomness.
- `flip`: State transform applied on rejection, e.g. the momentum flip
    `(x, p) -> (x, -p)`. If `None` then the state is left unchanged on
    rejection (plain Metropolis on a symmetric/involutive proposal).
"""


class MetropolisHastingsAdjusted(AbstractMetropolisSolver):
    r"""Wraps another solver and Metropolis--Hastings-adjusts each of its steps,
    using the MALA Hastings ratio.

    The wrapped solver's step over $[t_0, t_1]$, $h = t_1 - t_0$, is assumed to
    be an Euler--Maruyama step of the overdamped Langevin SDE

    $\mathrm{d}y = \nabla \log\tilde\pi(y)\,\mathrm{d}t
        + \sqrt{2}\,\mathrm{d}W_t,$

    i.e. a draw from the proposal
    $q(y_1 | y_0) = \mathcal{N}(y_1;\, y_0 + h \nabla\log\tilde\pi(y_0),\, 2h I)$.
    The step accepts with probability $\exp(\min(0, \log\rho))$ where

    $\log\rho = \log\tilde\pi(y_1) - \log\tilde\pi(y_0)
        + \log q(y_0 | y_1) - \log q(y_1 | y_0),$

    and keeps $y_0$ otherwise. This is the Metropolis-adjusted Langevin
    algorithm (MALA): the chain leaves $\tilde\pi$ invariant exactly.

    See the module note in `diffrax/_solver/metropolis.py` (or
    [`diffrax.GHMC`][]) for how the PRNG key is threaded.

    ??? cite "Reference"

        ```bibtex
        @article{roberts1996exponential,
            title={Exponential convergence of {L}angevin distributions and
                   their discrete approximations},
            author={Roberts, Gareth O and Tweedie, Richard L},
            journal={Bernoulli},
            volume={2},
            number={4},
            pages={341--363},
            year={1996},
        }
        ```
    """

    solver: AbstractSolver
    logdensity_fn: Callable[[Y, Args], RealScalarLike]
    key: PRNGKeyArray

    def _propose(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        inner_state,
        made_jump: BoolScalarLike,
    ):
        y1, y_error, dense_info, inner_state, result = self.solver.step(
            terms, t0, t1, y0, args, inner_state, made_jump
        )
        if y_error is not None:
            raise NotImplementedError(
                "`MetropolisHastingsAdjusted` does not support inner solvers with "
                "error estimates."
            )

        h = t1 - t0
        logp0, grad0 = jax.value_and_grad(self.logdensity_fn)(y0, args)
        logp1, grad1 = jax.value_and_grad(self.logdensity_fn)(y1, args)
        # log q(y0 | y1) - log q(y1 | y0), with
        # q(y' | y) = N(y'; y + h grad(y), 2h I).
        fwd = (y1**ω - y0**ω - h * grad0**ω).ω
        bwd = (y0**ω - y1**ω - h * grad1**ω).ω
        log_ratio = logp1 - logp0 + (_sum_squares(fwd) - _sum_squares(bwd)) / (4 * h)
        return y1, dense_info, inner_state, result, log_ratio

    def acceptance_probability(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> RealScalarLike:
        """The Metropolis--Hastings acceptance probability
        `exp(min(0, log rho))` of the step over `[t0, t1]` from `y0` -- the
        exact quantity `step` evaluates before drawing its accept uniform
        (same code path, via the same proposal computation). Differentiable,
        including where the acceptance saturates at 1 (the log-domain clamp
        gives derivative 0 there instead of the clip-after-exp NaN). Only
        deterministic (and only equal to the acceptance used by a given
        `step` call) if the proposal noise is carried by the terms, e.g. a
        fixed Brownian path, since this re-makes the wrapped solver's step.
        """
        inner_state = self.solver.init(terms, t0, t1, y0, args)
        *_, log_ratio = self._propose(terms, t0, t1, y0, args, inner_state, False)
        return jnp.exp(_log_acceptance(log_ratio))

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        inner_state, key = solver_state
        y1, dense_info, inner_state, result, log_ratio = self._propose(
            terms, t0, t1, y0, args, inner_state, made_jump
        )
        log_accept = _log_acceptance(log_ratio)

        next_key, accept_key = jr.split(key)
        u = jr.uniform(accept_key)
        accept = u < jnp.exp(log_accept)

        y_next = _tree_where(accept, y1, y0)
        dense_info = dict(dense_info, y1=y_next)
        return y_next, None, dense_info, (inner_state, next_key), result


MetropolisHastingsAdjusted.__init__.__doc__ = """**Arguments:**

- `solver`: The solver to wrap, e.g. [`diffrax.Euler`][] on the overdamped
    Langevin SDE with drift `grad(log pi)` and diffusion `sqrt(2)`.
- `logdensity_fn`: The unnormalized target log density `log pi(y, args)`.
- `key`: A PRNG key, used to seed the accept/reject randomness.
"""


class GHMC(AbstractMetropolisSolver):
    r"""Generalized Hamiltonian Monte Carlo: Metropolis-adjusted kinetic
    (underdamped) Langevin dynamics.

    The order-2 member of the Langevin hierarchy (Mou et al. 2021), on the
    extended state $y = (x, p)$ with extended energy
    $H(x, p) = -\log\tilde\pi(x) + \frac{1}{2}|p|^2$, targetting the extended
    Gibbs measure $\tilde\pi(x)\,\mathcal{N}(p; 0, I)$. Each step of size
    $h = t_1 - t_0$ performs, with $\alpha = \exp(-\gamma h / 2)$:

    1. An exact Ornstein--Uhlenbeck partial momentum refresh over time $h/2$:
       $p \leftarrow \alpha p + \sqrt{1 - \alpha^2}\,\xi$,
       $\xi \sim \mathcal{N}(0, I)$. This leaves $\mathcal{N}(0, I)$ invariant.
    2. One step of the wrapped solver (typically
       [`diffrax.VelocityVerlet`][]), Metropolis-corrected with acceptance
       $\exp(\min(0, -\Delta H))$; on rejection the state is $(x, -p)$ -- the
       momentum flip is required for invariance with persistent momentum.
    3. A second OU partial refresh over time $h/2$.

    This is the OBABO splitting of underdamped Langevin dynamics with the
    leapfrog core corrected by Metropolis: Horowitz's generalized/guided HMC,
    also known as Metropolis-adjusted kinetic Langevin, and (over multi-step
    trajectories) Metropolis Adjusted Langevin Trajectories (MALT). The
    $\gamma \to \infty$ corner is HMC with full momentum refreshment.

    See the module docstring in `diffrax/_solver/metropolis.py` for how the
    PRNG key is threaded (a `key` field seeds the solver state, so this
    composes with [`diffrax.diffeqsolve`][] at constant step size; `step` can
    also be driven directly in a manual scan loop).

    ??? cite "References"

        ```bibtex
        @article{horowitz1991generalized,
            title={A generalized guided {M}onte {C}arlo algorithm},
            author={Horowitz, Alan M},
            journal={Physics Letters B},
            volume={268},
            number={2},
            pages={247--252},
            year={1991},
        }

        @article{bourabee2017randomized,
            title={Randomized {H}amiltonian {M}onte {C}arlo},
            author={Bou-Rabee, Nawaf and Sanz-Serna, Jes{\'u}s Mar{\'\i}a},
            journal={The Annals of Applied Probability},
            volume={27},
            number={4},
            pages={2159--2194},
            year={2017},
        }

        @article{rioudurand2022metropolis,
            title={Metropolis adjusted {L}angevin trajectories: a robust
                   alternative to {H}amiltonian {M}onte {C}arlo},
            author={Riou-Durand, Lionel and Vogrinc, Jure},
            journal={arXiv:2202.13230},
            year={2022},
        }

        @article{mou2021highorder,
            title={High-order {L}angevin diffusion yields an accelerated
                   {MCMC} algorithm},
            author={Mou, Wenlong and Ma, Yi-An and Wainwright, Martin J and
                    Bartlett, Peter L and Jordan, Michael I},
            journal={Journal of Machine Learning Research},
            volume={22},
            number={42},
            pages={1--41},
            year={2021},
        }
        ```
    """

    solver: AbstractSolver
    energy_fn: Callable[[Y, Args], RealScalarLike]
    gamma: RealScalarLike
    key: PRNGKeyArray

    def step(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Optional[Y], DenseInfo, _SolverState, RESULTS]:
        inner_state, key = solver_state
        next_key, refresh1_key, accept_key, refresh2_key = jr.split(key, 4)

        h = t1 - t0
        alpha = jnp.exp(-self.gamma * h * 0.5)
        beta = jnp.sqrt(1 - alpha**2)

        # O: partial momentum refresh (exact OU flow over time h/2).
        x0, p0 = y0
        xi1 = _normal_like(refresh1_key, p0)
        p_refreshed = (alpha * p0**ω + beta * xi1**ω).ω
        y_refreshed = (x0, p_refreshed)

        # BAB: leapfrog proposal with the wrapped solver, Metropolis-corrected;
        # momentum flip on rejection.
        y1, y_error, _, inner_state, result = self.solver.step(
            terms, t0, t1, y_refreshed, args, inner_state, made_jump
        )
        if y_error is not None:
            raise NotImplementedError(
                "`GHMC` does not support inner solvers with error estimates."
            )
        delta_energy = self.energy_fn(y1, args) - self.energy_fn(y_refreshed, args)
        log_accept = _log_acceptance(-delta_energy)
        u = jr.uniform(accept_key)
        accept = u < jnp.exp(log_accept)
        y_flipped = (x0, jtu.tree_map(jnp.negative, p_refreshed))
        x_next, p_next = _tree_where(accept, y1, y_flipped)

        # O: second partial refresh.
        xi2 = _normal_like(refresh2_key, p_next)
        p_next = (alpha * p_next**ω + beta * xi2**ω).ω

        y_next = (x_next, p_next)
        dense_info = dict(y0=y0, y1=y_next)
        return y_next, None, dense_info, (inner_state, next_key), result


GHMC.__init__.__doc__ = """**Arguments:**

- `solver`: The solver used for the Hamiltonian part, typically
    [`diffrax.VelocityVerlet`][] (whose 2-tuple terms should hold the
    Hamiltonian vector fields `f1(t, p, args) = p` and
    `f2(t, x, args) = grad(log pi)(x)`).
- `energy_fn`: The extended energy `H((x, p), args)`, e.g.
    `-log pi(x) + 0.5 |p|^2`.
- `gamma`: The friction coefficient of the underdamped Langevin dynamics.
    Each half-step OU refresh uses `alpha = exp(-gamma * h / 2)`. `gamma = 0`
    gives fully persistent momentum; `gamma -> infinity` gives full refresh
    (HMC).
- `key`: A PRNG key, used to seed the refresh and accept/reject randomness.
"""
