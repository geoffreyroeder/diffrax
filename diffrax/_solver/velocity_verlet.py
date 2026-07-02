from collections.abc import Callable
from typing import ClassVar
from typing_extensions import TypeAlias

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
Yb: TypeAlias = PyTree[Float[ArrayLike, "?*y"], " Y"]


class VelocityVerlet(AbstractSolver):
    r"""Velocity Verlet method.

    The Störmer--Verlet (leapfrog) integrator in kick-drift-kick form, for
    separable Hamiltonian systems. 2nd order method. Symplectic,
    volume-preserving, and time-reversible under the momentum flip
    $(x, p) \mapsto (x, -p)$. Does not support adaptive step sizing. Uses 1st
    order local linear interpolation for dense/ts output.

    As with [`diffrax.SemiImplicitEuler`][], the terms and state should be a
    2-tuple $(x, p)$, with the first term the vector field
    $f_1(t, p) = \partial H/\partial p$ driving $x$ and the second term the
    vector field $f_2(t, x) = -\partial H/\partial x$ driving $p$. Over a step
    of size $h$ it computes

    $p_{1/2} = p_0 + \frac{h}{2} f_2(t_0, x_0)$

    $x_1 = x_0 + h f_1(t_0 + \frac{h}{2}, p_{1/2})$

    $p_1 = p_{1/2} + \frac{h}{2} f_2(t_1, x_1)$

    ??? cite "References"

        This is the standard integrator of Hamiltonian Monte Carlo; see:

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
        ```
    """

    term_structure: ClassVar = (AbstractTerm, AbstractTerm)
    interpolation_cls: ClassVar[
        Callable[..., LocalLinearInterpolation]
    ] = LocalLinearInterpolation

    def order(self, terms):
        return 2

    def init(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[tuple[Ya, Yb], _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        term_1, term_2 = terms
        x0, p0 = y0
        control1 = term_1.contr(t0, t1)
        control2 = term_2.contr(t0, t1)
        tmid = t0 + 0.5 * (t1 - t0)

        p_half = (p0**ω + 0.5 * term_2.vf_prod(t0, x0, args, control2) ** ω).ω
        x1 = (x0**ω + term_1.vf_prod(tmid, p_half, args, control1) ** ω).ω
        p1 = (p_half**ω + 0.5 * term_2.vf_prod(t1, x1, args, control2) ** ω).ω

        y1 = (x1, p1)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: tuple[AbstractTerm, AbstractTerm],
        t0: RealScalarLike,
        y0: tuple[Ya, Yb],
        args: Args,
    ) -> VF:
        term_1, term_2 = terms
        y0_1, y0_2 = y0
        f1 = term_1.vf(t0, y0_2, args)
        f2 = term_2.vf(t0, y0_1, args)
        return f1, f2
