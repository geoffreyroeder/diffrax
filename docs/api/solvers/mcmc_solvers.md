# MCMC solvers

These solvers wrap another solver and Metropolis-adjust each of its steps, so that the resulting discrete chain leaves a target density invariant exactly. They can be used to build MCMC kernels (HMC, MALA, generalized HMC, higher-order Langevin) out of differential equation solvers.

!!! info "PRNG key threading"

    The accept/reject step (and, where applicable, the momentum refresh) consumes randomness. These solvers store an initial PRNG key as a field: `init` seeds the solver state with it, and each `step` splits the key carried in the solver state. This composes with [`diffrax.diffeqsolve`][]. Use a constant step size: an adaptive controller that rejects and retries steps would reuse randomness.

---

::: diffrax.MetropolisAdjusted
    selection:
        members:
            - __init__

---

::: diffrax.MetropolisHastingsAdjusted
    selection:
        members:
            - __init__

---

::: diffrax.GHMC
    selection:
        members:
            - __init__

---

::: diffrax.ChainLangevin
    selection:
        members:
            - __init__
