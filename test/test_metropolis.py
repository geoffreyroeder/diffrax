"""Tests for `VelocityVerlet` and the Metropolis-adjusted wrapped solvers.

Most chain tests drive `solver.step(...)` in a manual `lax.scan` loop, threading
the `(inner_state, key)` solver state by hand.  This mirrors how the solvers are
used in the differentiable-MCMC estimator prototypes.  The solvers also compose
with `diffeqsolve` (the PRNG key is stored as a field and seeded into the solver
state by `init`); this is checked by `test_ghmc_diffeqsolve` and the
`VelocityVerlet` tests.
"""

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
from diffrax import ControlTerm, MultiTerm, ODETerm, VirtualBrownianTree
from jax import lax


def _harmonic_terms():
    # H(x, p) = 0.5 x^2 + 0.5 p^2
    return (
        ODETerm(lambda t, p, args: p),
        ODETerm(lambda t, x, args: -x),
    )


def _std_normal_logdensity(y, args):
    return -0.5 * jnp.sum(y**2)


def _harmonic_energy(y, args):
    x, p = y
    return 0.5 * jnp.sum(x**2) + 0.5 * jnp.sum(p**2)


#
# VelocityVerlet
#


def test_velocity_verlet_step_matches_kdk():
    solver = diffrax.VelocityVerlet()
    terms = _harmonic_terms()
    x0 = jnp.array(1.2)
    p0 = jnp.array(-0.7)
    h = 0.1
    state = solver.init(terms, 0.0, h, (x0, p0), None)
    (x1, p1), err, _, _, result = solver.step(
        terms, 0.0, h, (x0, p0), None, state, False
    )
    # Hand-computed kick-drift-kick update.
    p_half = p0 + 0.5 * h * (-x0)
    x1_expected = x0 + h * p_half
    p1_expected = p_half + 0.5 * h * (-x1_expected)
    assert err is None
    assert result == diffrax.RESULTS.successful
    assert jnp.allclose(x1, x1_expected, atol=1e-14)
    assert jnp.allclose(p1, p1_expected, atol=1e-14)


def test_velocity_verlet_energy_bounded():
    y0 = (jnp.array(1.0), jnp.array(0.5))
    e0 = _harmonic_energy(y0, None)
    h = 0.05
    n = 1000

    sol = diffrax.diffeqsolve(
        _harmonic_terms(),
        diffrax.VelocityVerlet(),
        t0=0.0,
        t1=n * h,
        dt0=h,
        y0=y0,
        saveat=diffrax.SaveAt(steps=True),
        max_steps=n,
    )
    xs, ps = sol.ys
    energies = 0.5 * (xs**2 + ps**2)  # per-step energies
    vv_err = jnp.max(jnp.abs(energies - e0))
    assert vv_err < 1e-3  # symplectic: bounded energy error over long runs

    # Explicit Euler on the same (joint) system drifts.
    euler_term = ODETerm(lambda t, y, args: (y[1], -y[0]))
    sol_euler = diffrax.diffeqsolve(
        euler_term,
        diffrax.Euler(),
        t0=0.0,
        t1=n * h,
        dt0=h,
        y0=y0,
        saveat=diffrax.SaveAt(steps=True),
        max_steps=n,
    )
    xs_e, ps_e = sol_euler.ys
    euler_err = jnp.max(jnp.abs(0.5 * (xs_e**2 + ps_e**2) - e0))
    assert euler_err > 1.0


def test_velocity_verlet_time_reversible():
    solver = diffrax.VelocityVerlet()
    terms = _harmonic_terms()
    y0 = (jnp.array(0.8), jnp.array(-1.3))
    h = 0.37
    state = solver.init(terms, 0.0, h, y0, None)
    (x1, p1), _, _, _, _ = solver.step(terms, 0.0, h, y0, None, state, False)
    # Flip the momentum, integrate forwards again, flip back: recovers y0.
    (x2, p2), _, _, _, _ = solver.step(terms, 0.0, h, (x1, -p1), None, state, False)
    assert jnp.allclose(x2, y0[0], atol=1e-12)
    assert jnp.allclose(-p2, y0[1], atol=1e-12)


def test_exports_and_instance_check():
    for name in (
        "VelocityVerlet",
        "MetropolisAdjusted",
        "MetropolisHastingsAdjusted",
        "GHMC",
    ):
        assert hasattr(diffrax, name)
    solver = diffrax.MetropolisAdjusted(
        diffrax.VelocityVerlet(), _harmonic_energy, key=jr.PRNGKey(0)
    )
    # AbstractWrappedSolver forwards isinstance checks to the wrapped solver.
    assert isinstance(solver, diffrax.VelocityVerlet)
    assert not isinstance(solver, diffrax.Euler)


#
# MetropolisHastingsAdjusted (MALA)
#


def test_mala_single_step_acceptance():
    h = 1.2
    y0 = jnp.array([1.4, -0.3])
    vbt = VirtualBrownianTree(0.0, h, tol=h / 8, shape=(2,), key=jr.PRNGKey(0))
    diffusion = lambda t, y, args: lx.DiagonalLinearOperator(
        jnp.full((2,), jnp.sqrt(2.0))
    )
    terms = MultiTerm(ODETerm(lambda t, y, args: -y), ControlTerm(diffusion, vbt))

    # The proposal the wrapped Euler solver makes, reconstructed by hand.
    dw = vbt.evaluate(0.0, h)
    y1 = y0 + h * (-y0) + jnp.sqrt(2.0) * dw

    # Analytic MALA acceptance, from first principles:
    # q(y' | y) = N(y'; y + h grad(y), 2h I).
    grad_ld = jax.grad(_std_normal_logdensity)

    def logq(y_to, y_from):
        return -jnp.sum((y_to - y_from - h * grad_ld(y_from, None)) ** 2) / (4 * h)

    log_ratio = (
        _std_normal_logdensity(y1, None)
        - _std_normal_logdensity(y0, None)
        + logq(y0, y1)
        - logq(y1, y0)
    )
    a = jnp.exp(jnp.minimum(0.0, log_ratio))
    assert 0.05 < a < 0.95  # the configuration actually exercises accept/reject

    def single_step(key):
        solver = diffrax.MetropolisHastingsAdjusted(
            diffrax.Euler(), _std_normal_logdensity, key=key
        )
        state = solver.init(terms, 0.0, h, y0, None)
        y_next, _, _, _, _ = solver.step(terms, 0.0, h, y0, None, state, False)
        accepted = jnp.allclose(y_next, y1, atol=1e-10)
        rejected = jnp.allclose(y_next, y0, atol=1e-10)
        return accepted, rejected

    accepted, rejected = jax.vmap(single_step)(jr.split(jr.PRNGKey(0), 4000))
    assert jnp.all(accepted | rejected)  # output is always proposal or current state
    freq = jnp.mean(accepted.astype(jnp.float64))
    assert jnp.abs(freq - a) < 4 * jnp.sqrt(a * (1 - a) / 4000) + 1e-3


def test_mala_chain_moments():
    h = 0.5
    n_steps = 2000
    n_burn = 300
    diffusion = lambda t, y, args: lx.DiagonalLinearOperator(
        jnp.full((1,), jnp.sqrt(2.0))
    )

    def run_chain(chain_key):
        vbt_key, solver_key = jr.split(chain_key)
        vbt = VirtualBrownianTree(
            0.0, n_steps * h, tol=h / 4, shape=(1,), key=vbt_key
        )
        terms = MultiTerm(
            ODETerm(lambda t, y, args: -y), ControlTerm(diffusion, vbt)
        )
        solver = diffrax.MetropolisHastingsAdjusted(
            diffrax.Euler(), _std_normal_logdensity, key=solver_key
        )
        y0 = jnp.zeros((1,))
        state = solver.init(terms, 0.0, h, y0, None)

        def body(carry, t):
            y, s = carry
            y_next, _, _, s_next, _ = solver.step(terms, t, t + h, y, None, s, False)
            return (y_next, s_next), y_next

        ts = h * jnp.arange(n_steps, dtype=jnp.float64)
        _, ys = lax.scan(body, (y0, state), ts)
        return ys[n_burn:, 0]

    xs = jax.vmap(run_chain)(jr.split(jr.PRNGKey(7), 8))
    assert jnp.abs(jnp.mean(xs)) < 0.1
    assert jnp.abs(jnp.var(xs) - 1.0) < 0.15


#
# MetropolisAdjusted(VelocityVerlet) == HMC / generalized-Metropolis
#


def test_metropolis_adjusted_acceptance_matches_energy_difference():
    terms = _harmonic_terms()
    h = 1.9
    x0 = jnp.array(1.5)
    p0 = jnp.array(1.5)

    # Manual leapfrog + energy difference.
    p_half = p0 + 0.5 * h * (-x0)
    x1 = x0 + h * p_half
    p1 = p_half + 0.5 * h * (-x1)
    delta_h = _harmonic_energy((x1, p1), None) - _harmonic_energy((x0, p0), None)
    a = jnp.exp(jnp.minimum(0.0, -delta_h))
    assert 0.05 < a < 0.95

    flip = lambda y: (y[0], -y[1])

    def single_step(key):
        solver = diffrax.MetropolisAdjusted(
            diffrax.VelocityVerlet(), _harmonic_energy, key=key, flip=flip
        )
        state = solver.init(terms, 0.0, h, (x0, p0), None)
        (xn, pn), _, _, _, _ = solver.step(
            terms, 0.0, h, (x0, p0), None, state, False
        )
        accepted = jnp.allclose(xn, x1, atol=1e-10) & jnp.allclose(
            pn, p1, atol=1e-10
        )
        rejected = jnp.allclose(xn, x0, atol=1e-10) & jnp.allclose(
            pn, -p0, atol=1e-10
        )
        return accepted, rejected

    accepted, rejected = jax.vmap(single_step)(jr.split(jr.PRNGKey(1), 4000))
    assert jnp.all(accepted | rejected)  # on rejection, the momentum flips
    freq = jnp.mean(accepted.astype(jnp.float64))
    assert jnp.abs(freq - a) < 4 * jnp.sqrt(a * (1 - a) / 4000) + 1e-3


def test_metropolis_adjusted_hmc_stationarity():
    # Full momentum refresh each step (the gamma -> infinity corner): plain HMC
    # with L=1. The x-marginal should be N(0, 1).
    terms = _harmonic_terms()
    h = 1.2
    n_steps = 4000
    n_burn = 200
    solver = diffrax.MetropolisAdjusted(
        diffrax.VelocityVerlet(),
        _harmonic_energy,
        key=jr.PRNGKey(0),
        flip=lambda y: (y[0], -y[1]),
    )

    def body(carry, _):
        x, key = carry
        key, p_key, step_key = jr.split(key, 3)
        p = jr.normal(p_key)
        (x_next, _), _, _, _, _ = solver.step(
            terms, 0.0, h, (x, p), None, (None, step_key), False
        )
        return (x_next, key), x_next

    _, xs = lax.scan(body, (jnp.array(0.0), jr.PRNGKey(3)), None, length=n_steps)
    xs = xs[n_burn:]
    assert jnp.abs(jnp.mean(xs)) < 0.08
    assert jnp.abs(jnp.var(xs) - 1.0) < 0.12


#
# GHMC (Metropolis-adjusted kinetic Langevin)
#


def test_ghmc_refresh_alpha():
    # Zero-gradient target: the leapfrog step is exactly x1 = x0 + h p', p1 = p'
    # with DeltaH = 0, so every step accepts. Then
    #     p' = alpha p0 + sqrt(1 - alpha^2) xi1,     alpha = exp(-gamma h / 2),
    #     p_out = alpha p' + sqrt(1 - alpha^2) xi2,
    # so E[(x_out - x0)/h] = alpha p0 and E[p_out] = alpha^2 p0 = exp(-gamma h) p0.
    gamma = 0.8
    h = 0.5
    alpha = jnp.exp(-gamma * h / 2)
    terms = (
        ODETerm(lambda t, p, args: p),
        ODETerm(lambda t, x, args: jnp.zeros_like(x)),
    )
    energy = lambda y, args: 0.5 * jnp.sum(y[1] ** 2)
    solver = diffrax.GHMC(diffrax.VelocityVerlet(), energy, gamma, key=jr.PRNGKey(0))
    x0 = jnp.array(0.0)
    p0 = jnp.array(4.0)

    def single_step(key):
        (x_out, p_out), _, _, _, _ = solver.step(
            terms, 0.0, h, (x0, p0), None, (None, key), False
        )
        return (x_out - x0) / h, p_out

    p_mid, p_out = jax.vmap(single_step)(jr.split(jr.PRNGKey(11), 20000))
    assert jnp.abs(jnp.mean(p_mid) - alpha * p0) < 0.03
    assert jnp.abs(jnp.mean(p_out) - alpha**2 * p0) < 0.03


def test_ghmc_rejection_flips_momentum():
    # gamma = 0: the OU refreshes are the identity, so a forced rejection (a huge
    # leapfrog step blows up the energy) must return exactly (x0, -p0).
    terms = _harmonic_terms()
    solver = diffrax.GHMC(
        diffrax.VelocityVerlet(), _harmonic_energy, 0.0, key=jr.PRNGKey(0)
    )
    x0 = jnp.array(1.2)
    p0 = jnp.array(0.7)
    (x_out, p_out), _, _, _, _ = solver.step(
        terms, 0.0, 50.0, (x0, p0), None, (None, jr.PRNGKey(4)), False
    )
    assert x_out == x0
    assert p_out == -p0


def test_ghmc_stationarity_2d():
    # 2-d Gaussian target with variances (1, 4).
    variances = jnp.array([1.0, 4.0])
    terms = (
        ODETerm(lambda t, p, args: p),
        ODETerm(lambda t, x, args: -x / variances),
    )
    energy = lambda y, args: 0.5 * jnp.sum(y[0] ** 2 / variances) + 0.5 * jnp.sum(
        y[1] ** 2
    )
    solver = diffrax.GHMC(diffrax.VelocityVerlet(), energy, 1.0, key=jr.PRNGKey(0))
    h = 0.5
    n_steps = 3000
    n_burn = 500

    def run_chain(chain_key):
        y0 = (jnp.zeros(2), jnp.zeros(2))

        def body(carry, _):
            y, s = carry
            y_next, _, _, s_next, _ = solver.step(
                terms, 0.0, h, y, None, s, False
            )
            return (y_next, s_next), y_next[0]

        _, xs = lax.scan(body, (y0, (None, chain_key)), None, length=n_steps)
        return xs[n_burn:]

    xs = jax.vmap(run_chain)(jr.split(jr.PRNGKey(20), 8))
    xs = xs.reshape(-1, 2)
    mean = jnp.mean(xs, axis=0)
    var = jnp.var(xs, axis=0)
    assert jnp.all(jnp.abs(mean) < 0.15)
    assert jnp.all(jnp.abs(var / variances - 1.0) < 0.2)


def test_ghmc_grad_finite():
    # Reverse-mode differentiability of a short chain w.r.t. a target parameter
    # (the mean of a Gaussian target), driving `step` in a manual scan.
    h = 0.5

    def loss(mu):
        energy = lambda y, args: 0.5 * jnp.sum((y[0] - args) ** 2) + 0.5 * jnp.sum(
            y[1] ** 2
        )
        terms = (
            ODETerm(lambda t, p, args: p),
            ODETerm(lambda t, x, args: -(x - args)),
        )
        solver = diffrax.GHMC(diffrax.VelocityVerlet(), energy, 1.0, key=jr.PRNGKey(5))
        y0 = (jnp.zeros(2), jnp.zeros(2))
        state = solver.init(terms, 0.0, h, y0, mu)

        def body(carry, _):
            y, s = carry
            y_next, _, _, s_next, _ = solver.step(terms, 0.0, h, y, mu, s, False)
            return (y_next, s_next), None

        (y_final, _), _ = lax.scan(body, (y0, state), None, length=30)
        return jnp.sum(y_final[0])

    grad = jax.grad(loss)(0.7)
    assert jnp.isfinite(grad)


def test_ghmc_diffeqsolve():
    # The stored-key-field design composes with diffeqsolve.
    terms = _harmonic_terms()
    solver = diffrax.GHMC(
        diffrax.VelocityVerlet(), _harmonic_energy, 1.0, key=jr.PRNGKey(8)
    )
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0=0.0,
        t1=10.0,
        dt0=0.5,
        y0=(jnp.array(0.3), jnp.array(-0.2)),
    )
    x_final, p_final = sol.ys
    assert jnp.all(jnp.isfinite(x_final))
    assert jnp.all(jnp.isfinite(p_final))
