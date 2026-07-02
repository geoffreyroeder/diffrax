"""Order-K Metropolis-adjusted Langevin in diffrax: ChainVerlet +
ChainLangevin.

Parity target: the diffmala order-K engine (diffmala/higher_order.py on the
same branch). Its deterministic core is ported verbatim below as
`_reference_integrate` (provenance: diffmala @ the hmc-langevin-replacement
branch; copied rather than imported because diffmala is not a dependency of
this repository) and ChainVerlet is checked against it to machine
precision, alongside the in-library K = 2 identity with VelocityVerlet, the
structural properties the Metropolis correction requires (unit Jacobian,
alternating-flip involution), and the stationarity of the full
ChainLangevin kernel at K = 3.
"""
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

jax.config.update("jax_enable_x64", True)

import diffrax


def _logdensity(x):
    return -0.5 * jnp.sum(x**2) - 0.1 * jnp.sum(x**4)


def _force_term(logdensity_fn):
    return diffrax.ODETerm(lambda t, x, args: jax.grad(logdensity_fn)(x))


def _energy(y, args):
    x, momenta = y
    return -_logdensity(x) + 0.5 * sum(jnp.sum(p**2) for p in momenta)


def _step_chain_verlet(solver, logdensity_fn, y0, h, num_steps=1):
    term = _force_term(logdensity_fn)
    y = y0
    for i in range(num_steps):
        y, _, _, _, _ = solver.step(term, i * h, (i + 1) * h, y, None, None, False)
    return y


# ---------------------------------------------------------------------------
# Reference: diffmala.higher_order.integrate, ported verbatim (flat arrays,
# stacked momenta) minus the divergence saturation, which is bitwise identity
# on the healthy trajectories used here.
# ---------------------------------------------------------------------------


def _reference_integrate(x, P, logdensity_fn, step_size, num_steps, order,
                         coefficients=None):
    if coefficients is None:
        coefficients = jnp.ones(order - 1)
    C = jnp.zeros((order, order))
    C = C.at[0, 1].set(coefficients[0])
    for i in range(1, order - 1):
        C = C.at[i, i + 1].set(coefficients[i])
        C = C.at[i + 1, i].set(-coefficients[i])
    flow = jax.scipy.linalg.expm(step_size * C)
    c0 = coefficients[0]
    half_kick = 0.5 * step_size * c0

    grad = jax.grad(logdensity_fn)(x)
    for _ in range(num_steps):
        P = P.at[0].add(half_kick * grad)
        z = flow @ jnp.concatenate([x[None, :], P], axis=0)
        x, P = z[0], z[1:]
        grad = jax.grad(logdensity_fn)(x)
        P = P.at[0].add(half_kick * grad)
    return x, P


@pytest.mark.parametrize("order,num_steps", [(2, 1), (3, 1), (3, 4), (5, 2)])
def test_chain_verlet_matches_diffmala_reference(order, num_steps):
    D, h = 3, 0.3
    x = jnp.array([0.4, -1.1, 0.7])
    P = 0.5 * jnp.arange(1, (order - 1) * D + 1, dtype=jnp.float64).reshape(
        order - 1, D
    )

    solver = diffrax.ChainVerlet(coefficients=(1.0,) * (order - 1))
    y = _step_chain_verlet(solver, _logdensity, (x, tuple(P)), h, num_steps)
    x_ref, P_ref = _reference_integrate(x, P, _logdensity, h, num_steps, order)

    assert jnp.allclose(y[0], x_ref, rtol=1e-13, atol=1e-14)
    for i in range(order - 1):
        assert jnp.allclose(y[1][i], P_ref[i], rtol=1e-13, atol=1e-14)


def test_chain_verlet_k2_is_velocity_verlet():
    """K = 2, c = (1,): identical to VelocityVerlet up to the exactness of
    the 2x2 nilpotent matrix exponential."""
    h = 0.37
    x = jnp.array([0.4, -1.1, 0.7])
    p = jnp.array([0.5, 0.1, -0.4])

    vv_terms = (
        diffrax.ODETerm(lambda t, p_, args: p_),
        diffrax.ODETerm(lambda t, x_, args: jax.grad(_logdensity)(x_)),
    )
    vv = diffrax.VelocityVerlet()
    (x_vv, p_vv), _, _, _, _ = vv.step(vv_terms, 0.0, h, (x, p), None, None, False)

    cv = diffrax.ChainVerlet(coefficients=(1.0,))
    (x_cv, (p_cv,)) = _step_chain_verlet(cv, _logdensity, (x, (p,)), h)

    assert jnp.allclose(x_cv, x_vv, rtol=1e-13, atol=1e-14)
    assert jnp.allclose(p_cv, p_vv, rtol=1e-13, atol=1e-14)


def _alternating_flip(momenta):
    return tuple(-p if i % 2 == 0 else p for i, p in enumerate(momenta))


@pytest.mark.parametrize("order", [2, 3, 5])
def test_chain_verlet_flip_involution(order):
    """S = F . Phi^L is an involution: integrate, alternating-flip, repeat,
    and return to the start."""
    D, h = 3, 0.25
    x = jax.random.normal(jax.random.key(0), (D,))
    momenta = tuple(
        0.3 * jax.random.normal(jax.random.key(i + 1), (D,))
        for i in range(order - 1)
    )
    solver = diffrax.ChainVerlet(coefficients=(1.0,) * (order - 1))

    def S(y):
        x1, m1 = _step_chain_verlet(solver, _logdensity, y, h, num_steps=2)
        return x1, _alternating_flip(m1)

    y2 = S(S((x, momenta)))
    assert jnp.allclose(y2[0], x, rtol=1e-10, atol=1e-10)
    for a, b in zip(y2[1], momenta):
        assert jnp.allclose(a, b, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("order", [3, 4])
def test_chain_verlet_unit_jacobian(order):
    D, h = 2, 0.3
    solver = diffrax.ChainVerlet(coefficients=(1.0,) * (order - 1))

    def flat_step(zflat):
        x = zflat[:D]
        momenta = tuple(
            zflat[D * (i + 1) : D * (i + 2)] for i in range(order - 1)
        )
        x1, m1 = _step_chain_verlet(solver, _logdensity, (x, momenta), h)
        return jnp.concatenate([x1, *m1])

    z0 = jnp.concatenate(
        [jnp.array([0.4, -1.1])]
        + [0.3 * jnp.ones(D) * (i + 1) for i in range(order - 1)]
    )
    J = jax.jacfwd(flat_step)(z0)
    assert jnp.allclose(jnp.abs(jnp.linalg.det(J)), 1.0, rtol=1e-12)


def test_chain_langevin_k3_converges_to_gaussian_moments():
    """The K = 3 ChainLangevin chain on N(1, 4) reproduces mean and
    variance — the same statistical pin the diffmala order-K suite uses."""
    target = lambda x: -0.5 * jnp.sum((x - 1.0) ** 2) / 4.0
    term = diffrax.ODETerm(lambda t, x, args: jax.grad(target)(x))

    def energy(y, args):
        x, momenta = y
        return -target(x) + 0.5 * sum(jnp.sum(p**2) for p in momenta)

    h = 0.8
    gamma = -jnp.log(0.7) / h  # alpha = 0.7, matching the diffmala test
    solver = diffrax.ChainLangevin(
        solver=diffrax.ChainVerlet(coefficients=(1.0, 1.0)),
        energy_fn=energy,
        gamma=gamma,
        key=jax.random.key(42),
    )

    y0 = (jnp.zeros((1,)), (jnp.zeros((1,)), jnp.zeros((1,))))
    state0 = solver.init(term, 0.0, h, y0, None)

    def one(carry, _):
        y, state, t = carry
        y1, _, _, state1, _ = solver.step(term, t, t + h, y, None, state, False)
        return (y1, state1, t + h), y1[0]

    (_, _, _), positions = jax.lax.scan(one, (y0, state0, 0.0), None, length=30_000)
    samples = positions[5_000:, 0]
    assert jnp.allclose(samples.mean(), 1.0, rtol=1e-1)
    assert jnp.allclose(samples.var(), 4.0, rtol=1e-1)


def test_chain_langevin_acceptance_matches_energy_difference():
    """One step's accept indicator is driven by exp(min(0, -dH)) computed
    from the wrapped solver's endpoint — checked by reconstructing the
    step's internals with the same key split."""
    import jax.random as jr

    h = 0.45
    key = jax.random.key(3)
    x = jnp.array([0.3, -0.7, 1.2])
    momenta = (jnp.array([0.5, 0.1, -0.4]), jnp.array([0.2, -0.9, 0.05]))
    term = _force_term(_logdensity)

    solver = diffrax.ChainLangevin(
        solver=diffrax.ChainVerlet(coefficients=(1.0, 1.0)),
        energy_fn=_energy,
        gamma=0.5,
        key=key,
    )
    y0 = (x, momenta)
    state0 = solver.init(term, 0.0, h, y0, None)
    y1, _, _, _, _ = solver.step(term, 0.0, h, y0, None, state0, False)

    # Reconstruct: same splits as ChainLangevin.step.
    _, refresh_key, accept_key = jr.split(key, 3)
    alpha = jnp.exp(-0.5 * h)
    keys = jr.split(refresh_key, 1)
    xi = jr.normal(keys[0], momenta[-1].shape, momenta[-1].dtype)
    top = alpha * momenta[-1] + jnp.sqrt(1 - alpha**2) * xi
    refreshed = (momenta[0], top)
    y_prop = _step_chain_verlet(
        diffrax.ChainVerlet(coefficients=(1.0, 1.0)), _logdensity,
        (x, refreshed), h,
    )
    delta = _energy(y_prop, None) - _energy((x, refreshed), None)
    accept = jr.uniform(accept_key) < jnp.exp(jnp.minimum(0.0, -delta))

    if accept:
        expected = y_prop
    else:
        expected = (x, (-refreshed[0], refreshed[1]))
    assert jnp.allclose(y1[0], expected[0], rtol=1e-13, atol=1e-14)
    for a, b in zip(y1[1], expected[1]):
        assert jnp.allclose(a, b, rtol=1e-13, atol=1e-14)
