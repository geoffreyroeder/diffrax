r"""Parity of diffrax MALA (`MetropolisHastingsAdjusted(Euler)` on overdamped
Langevin terms) with `blackjax.mcmc.mala`, ported from the retired diffmala
test suite (diffmala @ c4501e6).

Old test -> new test mapping:

- tests/test_mala_matches_blackjax.py::test_one_step_matches_blackjax
    -> test_one_step_matches_blackjax
- tests/test_mala_matches_blackjax.py::test_short_scan_matches_blackjax
    -> test_short_scan_matches_blackjax
- tests/test_mala_matches_blackjax.py::test_proposal_matches_blackjax_integrator
    -> test_proposal_matches_blackjax_integrator
- tests/test_mala_matches_blackjax.py::test_jit_matches_eager
    -> test_jit_matches_eager
- tests/test_mcmc_convergence.py::test_mala_converges_to_univariate_normal
    -> test_mala_converges_to_univariate_normal (original values: N(1, 2),
       45000 steps, burn-in 5000, step 0.2, seed 12, mean/var rtol 1e-1)
- tests/test_mcmc_convergence.py::test_mala_converges_on_linear_regression
    -> test_mala_converges_on_linear_regression (ORIGINAL MALA version:
       STEP_SIZE=1e-5, 10000 steps, burn-in 3000, blackjax seed 19,
       dict-pytree state {"coefs", "log_scale"}, atol 1e-1)
- tests/test_acceptance_overflow_gradient.py
    ::test_acceptance_gradient_finite_past_exp_overflow
    -> same name (original construction: TAU=0.5, x0=-100)
- tests/test_acceptance_overflow_gradient.py
    ::test_acceptance_gradient_finite_below_exp_overflow
    -> same name (x0=-40)
- tests/test_acceptance_overflow_gradient.py
    ::test_acceptance_gradient_nonzero_below_saturation
    -> same name (x0=-0.2)

**Randomness injection (not a mock).** blackjax's per-step randomness is

    key_integrator, key_mh = jax.random.split(step_key)
    eps = blackjax.util.generate_gaussian_noise(key_integrator, position)
    is_accepted = jax.random.bernoulli(key_mh, p_accept)   # = uniform(key_mh) < p

The identical `eps` is fed to the diffrax solver through
`_PrescribedIncrement`, a bona fide `diffrax.AbstractPath` control whose
increment over the step is exactly `eps` (a legitimate diffrax input; the
solver code under test is untouched). The accept uniform needs no injection:
`MetropolisHastingsAdjusted.step` draws `u = uniform(split(state_key)[1])`,
so driving each step with `state_key = step_key` makes `u` bitwise equal to
blackjax's bernoulli uniform (`jax.random.bernoulli(key, p)` is
`uniform(key, (), dtype(p)) < p`, and `split(step_key)[1] == key_mh`).

**Bitwise vs tolerance.** The terms are constructed so that every *product*
is bitwise identical to blackjax's proposal expression
`p + step_size * g + jnp.sqrt(2 * step_size) * n`:
the drift term computes `fl(step_size * g)` (ODETerm contracts the same
python-float control against the same gradient) and the diffusion term
computes `fl(fl(sqrt(2 * step_size)) * eps)` leaf-wise (same
`jnp.sqrt(2 * step_size)` expression, same eps). The ONLY difference is
addition association:

    blackjax:       x1_B = fl( fl(x0 + d) + s )      (left-to-right tree_map)
    diffrax Euler:  x1_D = fl( x0 + fl(d + s) )      (y0 + MultiTerm sum)

with d = fl(step_size * g(x0)), s = fl(sqrt(2*step_size) * eps). Since fp
addition is not associative, bitwise equality of the positions is
unattainable through Euler's `y0 + vf_prod` structure. Each rounded addition
has relative error <= u = 2^-53, so with S = x0 + d + s (exact) and
M = |x0| + |d| + |s|:

    |x1_B - S| <= 2 u M,   |x1_D - S| <= 2 u M   =>   |x1_B - x1_D| <= 4 u M,

i.e. a maximal deviation of ~4.4e-16 * M per component per step (M = O(10)
on this fixture grid). Position/logdensity/gradient/acceptance asserts
therefore use allclose at rtol=1e-13 (with atol=1e-14 for near-zero
components), which covers one step and 5 accumulated steps with two orders
of magnitude of margin. The acceptance probability additionally differs by
association inside the log-ratio (diffrax computes
`logp1 - logp0 + (|fwd|^2 - |bwd|^2)/(4 h)`; blackjax computes the
difference of two transition energies `-logp + 0.25 * (1/h) * |theta|^2`),
which is the same real number under a different bracketing: same 1e-13
bucket. The accept/reject *decision* is asserted with exact equality: the
uniforms are bitwise identical, and the acceptance probabilities agree to
~1e-15, so the decisions agree unless |u - a| < 1e-15 (does not occur for
these seeds; a mismatch would also blow up the position asserts).

Like the retired suite, each case prints a one-line diagnostic under
`pytest -s` whose fields are the exact values being asserted.
"""

import functools
import operator
from typing import Any, Callable

import blackjax.mcmc.diffusions as bj_diffusions
import blackjax.mcmc.mala as bj_mala
import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats
import jax.tree_util as jtu
import numpy as np
import pytest
from blackjax.util import generate_gaussian_noise
from diffrax import MultiTerm, ODETerm


STEP_SIZE = 0.1

RTOL = 1e-13
ATOL = 1e-14


# ---------------------------------------------------------------------------
# diffrax-side plumbing: prescribed-increment control + leaf-wise diffusion
# ---------------------------------------------------------------------------


class _PrescribedIncrement(diffrax.AbstractPath):
    """A control path with a prescribed increment over the single step
    `[0, interval]`. A bona fide diffrax control, used to feed the solver
    exactly the Gaussian increment blackjax draws."""

    increment: Any
    interval: float

    @property
    def t0(self):
        return 0.0

    @property
    def t1(self):
        return self.interval

    def evaluate(self, t0, t1=None, left=True):
        del left
        if t1 is None:
            raise ValueError("Only increments are defined for this path.")
        return self.increment


class _LeafwiseControlTerm(diffrax.AbstractTerm):
    """Diffusion term `f(t, y, args) dW` with a diagonal vector field
    represented leaf-wise: `prod` multiplies vf and control leaf by leaf (the
    semantics of the deprecated `diffrax.WeaklyDiagonalControlTerm`, restated
    without the deprecation warning). This reproduces blackjax's
    `sqrt(2 * step_size) * noise` products bitwise."""

    vector_field: Callable
    control: diffrax.AbstractPath

    def vf(self, t, y, args):
        return self.vector_field(t, y, args)

    def contr(self, t0, t1, **kwargs):
        return self.control.evaluate(t0, t1, **kwargs)

    def prod(self, vf, control):
        return jtu.tree_map(operator.mul, vf, control)


def _langevin_terms(logdensity_fn, eps, step_size):
    """Overdamped Langevin terms `dy = grad log pi dt + sqrt(2) dW` whose
    Euler step reproduces blackjax's MALA proposal product-for-product (see
    module docstring). `eps ~ N(0, I)` enters as the prescribed control
    increment with diffusion coefficient `sqrt(2 * step_size)` -- exactly
    blackjax's `sqrt(2 * step_size) * eps`."""
    drift = ODETerm(lambda t, y, args: jax.grad(logdensity_fn)(y, args))
    coeff = jnp.sqrt(2 * step_size)  # blackjax's exact expression
    diffusion = _LeafwiseControlTerm(
        lambda t, y, args: jtu.tree_map(lambda _: coeff, y),
        _PrescribedIncrement(increment=eps, interval=step_size),
    )
    return MultiTerm(drift, diffusion)


def _bj_step_eps(step_key, position_template):
    """The Gaussian noise blackjax's MALA kernel consumes at `step_key`."""
    key_integrator, _ = jr.split(step_key)
    return generate_gaussian_noise(key_integrator, position_template)


def _tree_array_equal(a, b):
    return all(
        jtu.tree_leaves(jtu.tree_map(lambda x, y: bool(jnp.array_equal(x, y)), a, b))
    )


def _tree_allclose(a, b, rtol=RTOL, atol=ATOL):
    return all(
        jtu.tree_leaves(
            jtu.tree_map(
                lambda x, y: bool(jnp.allclose(x, y, rtol=rtol, atol=atol)), a, b
            )
        )
    )


def _tree_max_abs_diff(a, b):
    return max(
        jtu.tree_leaves(jtu.tree_map(lambda x, y: float(jnp.max(jnp.abs(x - y))), a, b))
    )


# ---------------------------------------------------------------------------
# Fixture builders: verbatim from diffmala tests/test_mala_matches_blackjax.py,
# with the DecayingGaussian fixture inlined from tests/_fixtures/gaussians.py
# (diffmala @ c4501e6) -- provenance: copied rather than imported because the
# diffmala package no longer ships the retired MALA kernel or its fixtures.
# ---------------------------------------------------------------------------


def _make_decaying_gaussian(D, alpha, target_seed):
    """`Sigma = W diag(exp(log_v - i/alpha)) W^T` rotated Gaussian
    (tests/_fixtures/gaussians.py::make_decaying_gaussian, logdensity only)."""
    key = jax.random.key(target_seed)
    W, _ = jnp.linalg.qr(jax.random.normal(key, (D, D)))

    def logdensity_fn(theta, x):
        log_v = theta[D:]
        # `jnp.arange(D) / alpha` in the original; cast for the strict dtype
        # promotion this test suite (diffrax's conftest) runs under.
        log_d = log_v - jnp.arange(D, dtype=x.dtype) / alpha
        diff = x - theta[:D]
        y = W.T @ diff
        quad = jnp.sum(y * y * jnp.exp(-log_d))
        return -0.5 * (D * jnp.log(2.0 * jnp.pi) + jnp.sum(log_d) + quad)

    return logdensity_fn


def _isotropic(D, seed):
    """Isotropic D-dim Gaussian: log pi(theta, x) = -1/2 ||x - theta||^2."""
    key = jax.random.key(seed)
    k_theta, k_x0 = jax.random.split(key)
    theta = jax.random.uniform(k_theta, (D,), minval=-1.0, maxval=1.0)
    x0 = jax.random.normal(k_x0, (D,))
    log_pi = lambda th, x: -0.5 * jnp.sum((x - th) ** 2)
    return f"iso_D{D}_s{seed}", log_pi, theta, x0


def _decaying(D, seed):
    """Full-covariance rotated Gaussian with decaying eigenvalues at theta=0."""
    logdensity_fn = _make_decaying_gaussian(D=D, alpha=2.0, target_seed=seed)
    theta = jnp.zeros(2 * D)  # mu = 0, log_v = 0
    x0 = jax.random.normal(jax.random.key(seed + 1000), (D,))
    return f"decay_D{D}_s{seed}", logdensity_fn, theta, x0


FIXTURES = [
    _isotropic(D=1, seed=0),
    _isotropic(D=1, seed=1),
    _isotropic(D=3, seed=0),
    _isotropic(D=3, seed=1),
    _isotropic(D=7, seed=0),
    _decaying(D=5, seed=0),
    _decaying(D=5, seed=1),
]
FIXTURE_IDS = [f[0] for f in FIXTURES]


def _our_solver(target, key):
    """The system under test: diffrax MALA."""
    return diffrax.MetropolisHastingsAdjusted(
        diffrax.Euler(), lambda y, args: target(y), key=key
    )


def _our_step(solver, target, y0, step_key, step_size):
    """One diffrax-MALA step consuming blackjax's randomness at `step_key`."""
    eps = _bj_step_eps(step_key, y0)
    terms = _langevin_terms(solver.logdensity_fn, eps, step_size)
    y_next, _, _, _, _ = solver.step(
        terms, 0.0, step_size, y0, None, (None, step_key), False
    )
    accept_prob = solver.acceptance_probability(terms, 0.0, step_size, y0, None)
    return y_next, accept_prob


# ---------------------------------------------------------------------------
# Test group 2 (old numbering): one-step equivalence over the fixture grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, log_pi, theta, x0", FIXTURES, ids=FIXTURE_IDS)
def test_one_step_matches_blackjax(name, log_pi, theta, x0):
    """position, logdensity, logdensity_grad, acceptance_rate, is_accepted."""
    target = lambda x: log_pi(theta, x)
    key = jax.random.key(42)

    bj_state = bj_mala.init(x0, target)
    bj_kernel = bj_mala.build_kernel()
    bj_next, bj_info = bj_kernel(key, bj_state, target, STEP_SIZE)

    solver = _our_solver(target, key)
    our_pos, our_accept_prob = _our_step(solver, target, x0, key, STEP_SIZE)
    our_accepted = not _tree_array_equal(our_pos, x0)
    our_logdensity, our_grad = jax.value_and_grad(target)(our_pos)

    pos_diff = _tree_max_abs_diff(our_pos, bj_next.position)
    lpd_diff = float(jnp.abs(our_logdensity - bj_next.logdensity))
    grad_diff = _tree_max_abs_diff(our_grad, bj_next.logdensity_grad)
    a_diff = float(jnp.abs(our_accept_prob - bj_info.acceptance_rate))

    print(
        f"[one_step]  {name:14s}  a={float(our_accept_prob):.4f}  "
        f"pos_diff={pos_diff}  lpd_diff={lpd_diff}  grad_diff={grad_diff}  "
        f"a_diff={a_diff}"
    )

    # Association-only deviation: see module docstring for the derivation.
    assert _tree_allclose(our_pos, bj_next.position)
    assert jnp.allclose(our_logdensity, bj_next.logdensity, rtol=RTOL, atol=ATOL)
    assert _tree_allclose(our_grad, bj_next.logdensity_grad)
    assert jnp.allclose(our_accept_prob, bj_info.acceptance_rate, rtol=RTOL, atol=ATOL)
    # Bitwise: same uniform, decision must match exactly.
    assert our_accepted == bool(bj_info.is_accepted)
    if not our_accepted:
        # Rejection returns the current state bitwise on both sides.
        assert _tree_array_equal(our_pos, bj_next.position)


# ---------------------------------------------------------------------------
# Test group 3 (old numbering): 5-step trajectory equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name, log_pi, theta, x0", FIXTURES, ids=FIXTURE_IDS)
def test_short_scan_matches_blackjax(name, log_pi, theta, x0):
    """position, logdensity, logdensity_grad, acceptance_rate, is_accepted
    match at every one of 5 chained steps."""
    target = lambda x: log_pi(theta, x)
    NUM_STEPS = 5
    rng_key = jax.random.key(7)

    bj_state = bj_mala.init(x0, target)
    bj_kernel = bj_mala.build_kernel()
    solver = _our_solver(target, rng_key)
    our_pos = x0

    step_keys = jax.random.split(rng_key, NUM_STEPS)
    for step_idx, key in enumerate(step_keys):
        prev_pos = our_pos
        bj_state, bj_info = bj_kernel(key, bj_state, target, STEP_SIZE)
        our_pos, our_accept_prob = _our_step(solver, target, our_pos, key, STEP_SIZE)
        our_accepted = not _tree_array_equal(our_pos, prev_pos)
        our_logdensity, our_grad = jax.value_and_grad(target)(our_pos)

        pos_diff = _tree_max_abs_diff(our_pos, bj_state.position)
        lpd_diff = float(jnp.abs(our_logdensity - bj_state.logdensity))
        grad_diff = _tree_max_abs_diff(our_grad, bj_state.logdensity_grad)

        print(
            f"[scan]      {name:14s}  step={step_idx + 1}/{NUM_STEPS}  "
            f"a={float(our_accept_prob):.4f}  "
            f"pos_diff={pos_diff}  lpd_diff={lpd_diff}  grad_diff={grad_diff}"
        )

        assert _tree_allclose(
            our_pos, bj_state.position
        ), f"position mismatch at step {step_idx} ({name})"
        assert jnp.allclose(
            our_logdensity, bj_state.logdensity, rtol=RTOL, atol=ATOL
        ), f"logdensity mismatch at step {step_idx} ({name})"
        assert _tree_allclose(
            our_grad, bj_state.logdensity_grad
        ), f"logdensity_grad mismatch at step {step_idx} ({name})"
        assert jnp.allclose(
            our_accept_prob, bj_info.acceptance_rate, rtol=RTOL, atol=ATOL
        ), f"acceptance_rate mismatch at step {step_idx} ({name})"
        assert our_accepted == bool(
            bj_info.is_accepted
        ), f"is_accepted mismatch at step {step_idx} ({name})"


# ---------------------------------------------------------------------------
# Test group 4 (old numbering): the proposal the wrapped Euler solver makes
# matches BlackJAX's integrator at the integrator subkey.
# ---------------------------------------------------------------------------


def test_proposal_matches_blackjax_integrator():
    """The inner Euler proposal (the exact code path `step` uses) matches the
    DiffusionState from BlackJAX's integrator subkey."""
    D = 3
    theta = jnp.ones(D)
    x0 = jnp.array([0.5, -1.0, 2.0])
    log_pi = lambda th, x: -0.5 * jnp.sum((x - th) ** 2)
    target = lambda x: log_pi(theta, x)
    key = jax.random.key(2)

    # Replicate the kernel's internal key split to obtain the integrator subkey.
    key_integrator, _ = jax.random.split(key)
    grad_fn = jax.value_and_grad(target)
    integrator = bj_diffusions.overdamped_langevin(grad_fn)
    expected = integrator(key_integrator, bj_mala.init(x0, target), STEP_SIZE)

    solver = _our_solver(target, key)
    eps = _bj_step_eps(key, x0)
    terms = _langevin_terms(solver.logdensity_fn, eps, STEP_SIZE)
    # `solver.step`/`solver.acceptance_probability` obtain the proposal from
    # exactly this call on the wrapped solver.
    our_proposal, _, _, _, _ = solver.solver.step(
        terms, 0.0, STEP_SIZE, x0, None, None, False
    )
    our_logdensity, our_grad = jax.value_and_grad(target)(our_proposal)

    pos_diff = _tree_max_abs_diff(our_proposal, expected.position)
    lpd_diff = float(jnp.abs(our_logdensity - expected.logdensity))
    grad_diff = _tree_max_abs_diff(our_grad, expected.logdensity_grad)

    print(
        f"[proposal]  iso_D{D}         pos_diff={pos_diff}  "
        f"lpd_diff={lpd_diff}  grad_diff={grad_diff}"
    )

    assert _tree_allclose(our_proposal, expected.position)
    assert jnp.allclose(our_logdensity, expected.logdensity, rtol=RTOL, atol=ATOL)
    assert _tree_allclose(our_grad, expected.logdensity_grad)


# ---------------------------------------------------------------------------
# JIT-vs-eager kernel parity (tolerances as the old file)
# ---------------------------------------------------------------------------

JIT_RTOL = 1e-6
JIT_ATOL = 1e-6


@pytest.mark.parametrize("name, log_pi, theta, x0", FIXTURES, ids=FIXTURE_IDS)
def test_jit_matches_eager(name, log_pi, theta, x0):
    """jax.jit(step) matches the eager step up to fused-op reordering."""
    target = lambda x: log_pi(theta, x)
    key = jax.random.key(99)
    solver = _our_solver(target, key)

    def one_step(step_key, y0):
        return _our_step(solver, target, y0, step_key, STEP_SIZE)

    eager_pos, eager_a = one_step(key, x0)
    jit_pos, jit_a = jax.jit(one_step)(key, x0)

    pos_diff = _tree_max_abs_diff(jit_pos, eager_pos)
    a_diff = float(jnp.abs(jit_a - eager_a))
    eager_accepted = not _tree_array_equal(eager_pos, x0)
    jit_accepted = not _tree_array_equal(jit_pos, x0)

    print(
        f"[jit_eager] {name:14s}                          "
        f"pos_diff={pos_diff}  a_diff={a_diff}  (tol={JIT_ATOL})"
    )

    assert _tree_allclose(jit_pos, eager_pos, rtol=JIT_RTOL, atol=JIT_ATOL)
    assert jnp.allclose(jit_a, eager_a, rtol=JIT_RTOL, atol=JIT_ATOL)
    assert jit_accepted == eager_accepted


# ---------------------------------------------------------------------------
# Convergence (ported from tests/test_mcmc_convergence.py, original values)
# ---------------------------------------------------------------------------


def _run_chain(solver, logdensity_fn, y0, keys, step_size):
    """Drive the diffrax-MALA chain in a scan, one blackjax step key per step
    (integrator subkey -> prescribed control increment; the accept uniform
    comes from the solver's own split of the step key, which equals
    blackjax's)."""
    eps_all = jax.vmap(lambda k: _bj_step_eps(k, y0))(keys)

    def body(y, xs):
        key, eps = xs
        terms = _langevin_terms(logdensity_fn, eps, step_size)
        y_next, _, _, _, _ = solver.step(
            terms, 0.0, step_size, y, None, (None, key), False
        )
        return y_next, y_next

    _, positions = jax.lax.scan(body, y0, (keys, eps_all))
    return positions


def test_mala_converges_to_univariate_normal():
    """Imitates blackjax UnivariateNormalTest::test_mala: target N(1.0, 2.0),
    45000 steps, burn-in 5000, step_size 0.2, blackjax seed 12.

    Run under float32, the dtype the blackjax/diffmala suites ran under
    (diffrax's conftest enables x64 globally). This is load-bearing for the
    seed, not a fudge: blackjax's own kernel at this seed gives
    mean = 0.9042 at f32 (passes) but mean = 1.1428 at f64 (fails rtol 1e-1)
    -- the original test's pass is a property of the f32 chain, which the
    diffrax solver reproduces here.
    """
    from jax.experimental import disable_x64

    normal_logprob = lambda x, args: jstats.norm.logpdf(x, loc=1.0, scale=2.0)

    NUM_STEPS, BURNIN, STEP_SIZE_ = 45_000, 5_000, 0.2
    with disable_x64():
        y0 = jnp.array(1.0)
        keys = jax.random.split(jax.random.key(12), NUM_STEPS)  # blackjax seed
        solver = diffrax.MetropolisHastingsAdjusted(
            diffrax.Euler(), normal_logprob, key=jax.random.key(12)
        )
        positions = _run_chain(solver, normal_logprob, y0, keys, STEP_SIZE_)
    samples = positions[BURNIN:]
    moved = jnp.mean((samples[1:] != samples[:-1]).astype(jnp.float64))

    print(
        f"\n  [univariate]   N={NUM_STEPS}  burnin={BURNIN}  step={STEP_SIZE_}  "
        f"acc(realised)={float(moved):.4f}  "
        f"mean={float(samples.mean()):.4f} (truth 1.0)  "
        f"var={float(samples.var()):.4f}  (truth 4.0)"
    )

    np.testing.assert_allclose(samples.mean(), 1.0, rtol=1e-1)
    np.testing.assert_allclose(samples.var(), 4.0, rtol=1e-1)


def test_mala_converges_on_linear_regression():
    """Imitates blackjax LinearRegressionTest::test_mala: coefs/scale
    posterior, 10000 steps, burn-in 3000, STEP_SIZE=1e-5, blackjax seed 19.
    The dict-pytree state {"coefs", "log_scale"} doubles as a pytree-state
    test of the solver."""

    def regression_logprob(log_scale, coefs, preds, x):
        scale = jnp.exp(log_scale)
        scale_prior = jstats.expon.logpdf(scale, 0, 1) + log_scale
        coefs_prior = jstats.norm.logpdf(coefs, 0, 5)
        y = jnp.dot(x, coefs)
        loglik = jstats.norm.logpdf(preds, y, scale)
        return sum(t.sum() for t in (scale_prior, coefs_prior, loglik))

    KEY = jax.random.key(19)  # blackjax fixture seed
    init_key0, init_key1, inference_key = jax.random.split(KEY, 3)
    x_data = jax.random.normal(init_key0, shape=(1000, 1))
    y_data = 3 * x_data + jax.random.normal(init_key1, shape=x_data.shape)

    log_pi_partial = functools.partial(regression_logprob, x=x_data, preds=y_data)
    log_pi = lambda state, args: log_pi_partial(**state)

    NUM_STEPS, BURNIN, STEP_SIZE_ = 10_000, 3_000, 1e-5
    y0 = {"coefs": jnp.asarray(1.0), "log_scale": jnp.asarray(1.0)}
    keys = jax.random.split(inference_key, NUM_STEPS)
    solver = diffrax.MetropolisHastingsAdjusted(diffrax.Euler(), log_pi, key=KEY)
    positions = _run_chain(solver, log_pi, y0, keys, STEP_SIZE_)
    coefs_samples = positions["coefs"][BURNIN:]
    scale_samples = jnp.exp(positions["log_scale"][BURNIN:])
    moved = jnp.mean((coefs_samples[1:] != coefs_samples[:-1]).astype(jnp.float64))

    print(
        f"\n  [regression]   N={NUM_STEPS}  burnin={BURNIN}  step={STEP_SIZE_}  "
        f"acc(realised)={float(moved):.4f}  "
        f"mean(coefs)={float(coefs_samples.mean()):.4f} (truth 3.0)  "
        f"mean(scale)={float(scale_samples.mean()):.4f} (truth 1.0)"
    )

    np.testing.assert_allclose(scale_samples.mean(), 1.0, atol=1e-1)
    np.testing.assert_allclose(coefs_samples.mean(), 3.0, atol=1e-1)


# ---------------------------------------------------------------------------
# Acceptance-probability gradient at exp overflow
# (ported from tests/test_acceptance_overflow_gradient.py, original
# constructions: TAU = 0.5, x0 in {-100, -40, -0.2})
# ---------------------------------------------------------------------------

TAU = 0.5
OVERFLOW_KEY = jax.random.key(0)


def _acceptance(theta, x0):
    """The differentiated quantity: a single step's acceptance probability,
    computed by the solver's own code path (`acceptance_probability` is what
    `step` evaluates before drawing the accept uniform)."""
    target = lambda x, th: -0.5 * th * jnp.sum(x**2)
    y0 = jnp.array([x0])
    eps = _bj_step_eps(OVERFLOW_KEY, y0)
    solver = diffrax.MetropolisHastingsAdjusted(
        diffrax.Euler(), target, key=OVERFLOW_KEY
    )
    terms = _langevin_terms(target, eps, TAU)
    return solver.acceptance_probability(terms, 0.0, TAU, y0, theta)


def test_acceptance_gradient_finite_past_exp_overflow():
    """x0 = -100: log_p_accept ~ 940, past the f64 exp-overflow threshold.

    With the clip(exp(.)) form this gradient is NaN; the log-domain clamp
    gives the exact saturated-branch derivative, 0.0, with the forward value
    still exactly 1.0."""
    a, grad = jax.value_and_grad(_acceptance)(1.0, x0=-100.0)
    print(f"\n  [overflow] accept = {float(a)}  d accept / d theta = {float(grad)}")
    assert a == 1.0
    assert grad == 0.0


def test_acceptance_gradient_finite_below_exp_overflow():
    """x0 = -40: log_p_accept ~ 150, saturated but no exp overflow at f64.

    Same saturated forward value and the same exact derivative as the
    overflow case: the fix makes the two regimes indistinguishable, as the
    true function min(1, e^L) demands for any L > 0."""
    a, grad = jax.value_and_grad(_acceptance)(1.0, x0=-40.0)
    print(f"\n  [saturated] accept = {float(a)}  d accept / d theta = {float(grad)}")
    assert a == 1.0
    assert grad == 0.0


def test_acceptance_gradient_nonzero_below_saturation():
    """A rejected-regime step (log_p_accept < 0): the acceptance gradient is
    finite and nonzero -- the fix must not kill the gradient the estimator
    needs where the true derivative d/dL e^L = a is nonzero."""
    a, grad = jax.value_and_grad(_acceptance)(1.0, x0=-0.2)
    print(f"\n  [interior]  accept = {float(a)}  d accept / d theta = {float(grad)}")
    assert 0.0 < a <= 1.0
    assert jnp.isfinite(grad)
    if a < 1.0:
        assert grad != 0.0
