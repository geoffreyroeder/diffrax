import functools as ft
import warnings
from typing import Optional

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp

from .adjoint import (
    AbstractAdjoint,
    BacksolveAdjoint,
    NoAdjoint,
    RecursiveCheckpointAdjoint,
)
from .brownian import AbstractBrownianPath, UnsafeBrownianPath
from .custom_types import Array, Bool, Int, PyTree, Scalar
from .global_interpolation import DenseInterpolation
from .misc import (
    bounded_while_loop,
    branched_error_if,
    error_if,
    HadInplaceUpdate,
    unvmap_all,
    unvmap_max,
)
from .saveat import SaveAt
from .solution import RESULTS, Solution
from .solver import (
    AbstractAdaptiveSDESolver,
    AbstractItoSolver,
    AbstractSolver,
    AbstractStratonovichSolver,
    Euler,
)
from .step_size_controller import (
    AbstractAdaptiveStepSizeController,
    AbstractStepSizeController,
    ConstantStepSize,
    StepTo,
)
from .term import AbstractTerm, WrapTerm


class _State(eqx.Module):
    # Evolving state during the solve
    y: Array["state"]  # noqa: F821
    tprev: Scalar
    tnext: Scalar
    made_jump: Bool
    solver_state: PyTree
    controller_state: PyTree
    result: RESULTS
    num_steps: Int
    num_accepted_steps: Int
    num_rejected_steps: Int
    # Output that is .at[].set() updated during the solve (and their indices)
    saveat_ts_index: Scalar
    ts: Array["times"]  # noqa: F821
    ys: PyTree[Array["times", ...]]  # noqa: F821
    save_index: Int
    dense_ts: Optional[Array["times + 1"]]  # noqa: F821
    dense_infos: Optional[PyTree[Array["times", ...]]]  # noqa: F821
    dense_save_index: Int


class _InnerState(eqx.Module):
    saveat_ts_index: Int
    ts: Array["times"]  # noqa: F821
    ys: PyTree[Array["times", ...]]  # noqa: F821
    save_index: Int


def _save(state: _State, t: Scalar) -> _State:
    ts = state.ts
    ys = state.ys
    save_index = state.save_index
    y = state.y

    ts = ts.at[save_index].set(t)
    ys = jax.tree_map(lambda ys_, y_: ys_.at[save_index].set(y_), ys, y)
    save_index = save_index + 1

    return eqx.tree_at(
        lambda s: [s.ts, s.save_index] + jax.tree_leaves(s.ys),
        state,
        [ts, save_index] + jax.tree_leaves(ys),
    )


def _clip_to_end(tnext, t1):
    return jnp.where(tnext > t1 - 1e-6, t1, tnext)


def loop(
    *,
    solver,
    stepsize_controller,
    saveat,
    t0,
    t1,
    dt0,
    max_steps,
    terms,
    args,
    init_state,
    is_bounded,
):

    if saveat.t0:
        init_state = _save(init_state, t0)
    if saveat.dense:
        dense_ts = init_state.dense_ts
        dense_ts = dense_ts.at[0].set(t0)
        init_state = eqx.tree_at(lambda s: s.dense_ts, init_state, dense_ts)

    def cond_fun(state):
        return (state.tprev < t1) & (state.result == RESULTS.successful)

    def body_fun(state, inplace):

        #
        # Actually do some differential equation solving! Make numerical steps, adapt
        # step sizes, all that jazz.
        #

        (y, y_error, dense_info, solver_state, solver_result) = solver.step(
            terms,
            state.tprev,
            state.tnext,
            state.y,
            args,
            state.solver_state,
            state.made_jump,
        )

        local_order = _get_local_order(terms, solver)
        (
            keep_step,
            tprev,
            tnext,
            made_jump,
            controller_state,
            stepsize_controller_result,
        ) = stepsize_controller.adapt_step_size(
            state.tprev,
            state.tnext,
            state.y,
            y,
            args,
            y_error,
            local_order,
            state.controller_state,
        )
        assert jnp.result_type(keep_step) is jnp.dtype(bool)

        #
        # Do some book-keeping.
        #

        # The 1e-6 tolerance means that we don't end up with too-small intervals for
        # dense output, which then gives numerically unstable answers due to floating
        # point errors.
        tnext = _clip_to_end(tnext, t1)
        tprev = jnp.minimum(tprev, t1)

        # The other parts of the mutable state are kept/not-kept (based on whether the
        # step was accepted) by the stepsize controller. But it doesn't get access to
        # these parts, so we do them here.
        keep = lambda a, b: jnp.where(keep_step, a, b)
        y = jax.tree_map(keep, y, state.y)
        solver_state = jax.tree_map(keep, solver_state, state.solver_state)
        made_jump = keep(made_jump, state.made_jump)
        solver_result = keep(solver_result, RESULTS.successful)

        # TODO: if we ever support events, then they should go in here.
        # In particular the thing to be careful about is in the `if saveat.steps`
        # branch below, where we want to make sure that it is the value of `y` at
        # `tprev` that is actually saved. (And not just the value of `y` at the
        # previous step's `tnext`, i.e. immediately before the jump.)

        # Store the first unsuccessful result we get whilst iterating (if any).
        result = state.result
        result = jnp.where(result == RESULTS.successful, solver_result, result)
        result = jnp.where(
            result == RESULTS.successful, stepsize_controller_result, result
        )

        # Count the number of steps, just for statistical purposes.
        num_steps = state.num_steps + 1
        num_accepted_steps = state.num_accepted_steps + keep_step
        # Not just ~keep_step, which does the wrong thing when keep_step is a non-array
        # bool True/False.
        num_rejected_steps = state.num_rejected_steps + jnp.invert(keep_step)

        #
        # Store the output produced from this numerical step.
        # This is a bit involved, and uses the `inplace` function passed as an argument
        # to this body function.
        # This is because we need to make in-place updates to store our results, but
        # doing is a bit of a hassle inside `bounded_while_loop`. (See its docstring
        # for details.)
        #

        saveat_ts_index = state.saveat_ts_index
        ts = state.ts
        ys = state.ys
        save_index = state.save_index
        dense_ts = state.dense_ts
        dense_infos = state.dense_infos
        dense_save_index = state.dense_save_index
        made_inplace_update = False

        if saveat.ts is not None:
            made_inplace_update = True

            _interpolator = solver.interpolation_cls(
                t0=state.tprev, t1=state.tnext, **dense_info
            )

            def _saveat_get(_saveat_ts_index):
                return saveat.ts[jnp.minimum(_saveat_ts_index, len(saveat.ts) - 1)]

            def _cond_fun(_state):
                _saveat_ts_index = _state.saveat_ts_index
                _saveat_t = _saveat_get(_saveat_ts_index)
                return (
                    keep_step
                    & (_saveat_t <= state.tnext)
                    & (_saveat_ts_index < len(saveat.ts))
                )

            def _body_fun(_state, _inplace):
                _saveat_ts_index = _state.saveat_ts_index
                _ts = _state.ts
                _ys = _state.ys
                _save_index = _state.save_index

                _saveat_t = _saveat_get(_saveat_ts_index)
                _saveat_y = _interpolator.evaluate(_saveat_t)

                # VOODOO MAGIC
                #
                # Okay, time for some voodoo that I absolutely don't understand.
                #
                # Shown in the comment is what I would to write:
                #
                #  _inplace = _inplace.merge(inplace)
                #  _ts = _inplace(_ts).at[_save_index].set(_saveat_t)
                #  _ys = jax.tree_map(lambda __ys, __saveat_y: _inplace(__ys).at[_save_index].set(__saveat_y), _ys, _saveat_y)  # noqa: E501
                #
                # Seems reasonable, right? Just updating a value.
                #
                # Below is what we actually run:

                _inplace.merge(inplace)
                _pred = cond_fun(state) & _cond_fun(_state)
                _ts = _ts.at[_save_index].set(
                    jnp.where(_pred, _saveat_t, _ts[_save_index])
                )
                _ys = jax.tree_map(
                    lambda __ys, __saveat_y: __ys.at[_save_index].set(
                        jnp.where(_pred, __saveat_y, __ys[_save_index])
                    ),
                    _ys,
                    _saveat_y,
                )

                # Some immediate questions you might have:
                #
                # - Isn't this essentially equivalent to the commented-out version?
                #   - Nitpick: the commented-out version includes an enhanced cond_fun
                #     that checks the step count, but it shouldn't matter here.
                # - It looks like `_inplace.merge(inplace)` isn't even used?
                #   - I think it will appear in the jaxpr, interestingly, based off of
                #     the toy example:
                #     >>> def f(x, y):
                #     ...     x & y
                #     ...     return x + 1
                #     >>> jax.make_jaxpr(f)(1, 2)
                #     Which is presumably how this manages to affect anything at all.
                #
                # And you are right. Those are both reasonable questions, at least as
                # far as I can see.
                #
                # And yet for some reason this version will run substantially faster.
                # (At time of writing: on the `small_neural_ode.py` benchmark, on the
                # CPU.)
                #
                # ~VOODOO MAGIC

                _saveat_ts_index = _saveat_ts_index + 1
                _save_index = _save_index + 1

                _ts = HadInplaceUpdate(_ts)
                _ys = jax.tree_map(HadInplaceUpdate, _ys)

                return _InnerState(
                    saveat_ts_index=_saveat_ts_index,
                    ts=_ts,
                    ys=_ys,
                    save_index=_save_index,
                )

            init_inner_state = _InnerState(
                saveat_ts_index=saveat_ts_index, ts=ts, ys=ys, save_index=save_index
            )

            final_inner_state = bounded_while_loop(
                _cond_fun, _body_fun, init_inner_state, max_steps=len(saveat.ts)
            )

            saveat_ts_index = final_inner_state.saveat_ts_index
            ts = final_inner_state.ts
            ys = final_inner_state.ys
            save_index = final_inner_state.save_index

        def maybe_inplace(i, x, u):
            return inplace(x).at[i].set(jnp.where(keep_step, u, x[i]))

        if saveat.steps:
            made_inplace_update = True
            ts = maybe_inplace(save_index, ts, tprev)
            ys = jax.tree_map(ft.partial(maybe_inplace, save_index), ys, y)
            save_index = save_index + keep_step

        if saveat.dense:
            made_inplace_update = True
            dense_ts = maybe_inplace(dense_save_index + 1, dense_ts, tprev)
            dense_infos = jax.tree_map(
                ft.partial(maybe_inplace, dense_save_index),
                dense_infos,
                dense_info,
            )
            dense_save_index = dense_save_index + keep_step

        if made_inplace_update:
            ts = HadInplaceUpdate(ts)
            ys = jax.tree_map(HadInplaceUpdate, ys)
            dense_ts = HadInplaceUpdate(dense_ts)
            dense_infos = jax.tree_map(HadInplaceUpdate, dense_infos)

        new_state = _State(
            y=y,
            tprev=tprev,
            tnext=tnext,
            made_jump=made_jump,
            solver_state=solver_state,
            controller_state=controller_state,
            result=result,
            num_steps=num_steps,
            num_accepted_steps=num_accepted_steps,
            num_rejected_steps=num_rejected_steps,
            saveat_ts_index=saveat_ts_index,
            ts=ts,
            ys=ys,
            save_index=save_index,
            dense_ts=dense_ts,
            dense_infos=dense_infos,
            dense_save_index=dense_save_index,
        )

        return new_state

    if is_bounded:
        # Some privileged optimisations, but for common use cases.
        # TODO: make these a method on an AbstractFixedStepSizeController?
        #
        # These optimisations depend on implementations details of `ConstantStepSize`,
        # `StepTo`, and `bounded_while_loop`.
        #
        # We try to determine the exact number of integration steps that will be made.
        # If this is possible then we can use a single `lax.scan`, rather than the
        # recursive construction of `bounded_while_loop`. This primarily reduces
        # compilation times.
        if max_steps is None:
            # `bounded_while_loop(..., max_steps=None)` lowers to `lax.while_loop`
            # anyway; this is already fast. Don't try to determine the number of steps
            # needed.
            compiled_num_steps = None
        elif isinstance(stepsize_controller, ConstantStepSize) and (
            stepsize_controller.compile_steps is None
            or stepsize_controller.compile_steps is True
        ):
            # We can determine the number of steps quite easily with constant step
            # size.
            #
            # We do so using a `lax.while_loop`.
            # - Not just a (t1 - t0)/dt0 division, to avoid floating point errors.
            # - lax.while_loop, not just a Python one, to ensure that we match the
            #   behaviour at runtime; no funny edge cases.
            with jax.ensure_compile_time_eval():

                def _is_finite(_t):
                    all_finite = unvmap_all(jnp.isfinite(_t))
                    return not isinstance(all_finite, jax.core.Tracer) and all_finite

                if _is_finite(t0) and _is_finite(t1) and _is_finite(dt0):

                    def _cond_fun(_state):
                        _, _t = _state
                        return _t < t1

                    def _body_fun(_state):
                        _step, _t = _state
                        return _step + 1, _clip_to_end(_t + dt0, t1)

                    compiled_num_steps, _ = lax.while_loop(
                        _cond_fun, _body_fun, (0, t0)
                    )
                    compiled_num_steps = unvmap_max(compiled_num_steps)
                else:
                    if stepsize_controller.compile_steps is None:
                        compiled_num_steps = None
                    else:
                        assert stepsize_controller.compile_steps is True
                        raise ValueError(
                            "Could not determine exact number of steps, but "
                            "`stepsize_controller.compile_steps=True`"
                        )
        elif isinstance(stepsize_controller, StepTo) and (
            stepsize_controller.compile_steps is None
            or stepsize_controller.compile_steps is True
        ):
            # The user has explicitly specified the number of steps.
            compiled_num_steps = len(stepsize_controller.ts) - 1
        else:
            # Else we can't determine the number of steps.
            compiled_num_steps = None

        if compiled_num_steps is None or isinstance(
            compiled_num_steps, jax.core.Tracer
        ):
            # If we couldn't determine the number of steps then use the default
            # recursive construction.
            compiled_num_steps = None
            base = 16
        else:
            if isinstance(compiled_num_steps, jnp.ndarray):
                compiled_num_steps = compiled_num_steps.item()
            base = compiled_num_steps
            max_steps = min(max_steps, compiled_num_steps)

        final_state = bounded_while_loop(
            cond_fun, body_fun, init_state, max_steps, base=base
        )
    else:
        compiled_num_steps = None

        if max_steps is None:
            _cond_fun = cond_fun
        else:

            def _cond_fun(state):
                return cond_fun(state) & (state.num_steps < max_steps)

        final_state = bounded_while_loop(
            _cond_fun, body_fun, init_state, max_steps=None
        )

    if saveat.t1 and not saveat.steps:
        # if saveat.steps then the final value is already saved.
        final_state = _save(final_state, t1)
    result = jnp.where(
        cond_fun(final_state), RESULTS.max_steps_reached, final_state.result
    )
    aux_stats = dict(compiled_num_steps=compiled_num_steps)
    return eqx.tree_at(lambda s: s.result, final_state, result), aux_stats


# Assumes that the SDE-ness is interpretable by finding AbstractBrownianPath.
# In principle a user could re-create terms, controls, etc. without going via this,
# though. So this is a bit imperfect.
#
# Fortunately, at time of writing this is used for two things:
# - _get_local_order
# - error checking
# The former can be overriden by `PIDController(local_order=...)` and the latter is
# really just to catch common errors.
# That is, for the power user who implements enough to bypass this check -- probably
# they know what they're doing and can handle both of these cases appropriately.
def _is_sde(terms: PyTree[AbstractTerm]) -> bool:
    is_brownian = lambda x: isinstance(x, AbstractBrownianPath)
    leaves, _ = jax.tree_flatten(terms, is_leaf=is_brownian)
    return any(is_brownian(leaf) for leaf in leaves)


def _is_unsafe_sde(terms: PyTree[AbstractTerm]) -> bool:
    is_brownian = lambda x: isinstance(x, UnsafeBrownianPath)
    leaves, _ = jax.tree_flatten(terms, is_leaf=is_brownian)
    return any(is_brownian(leaf) for leaf in leaves)


def _get_local_order(terms: PyTree[AbstractTerm], solver: AbstractSolver) -> Scalar:
    """Guess the local order of convergence.

    The error estimate is assumed to come from the difference of two methods. If these
    two methods have orders `p` and `q` then the local order of the error estimate is
    `min(p, q) + 1` for an ODE and `min(p, q) + 0.5` for an SDE.

    - In the SDE case then we assume `p == q == solver.strong_order`.
    - In the ODE case then we assume `p == q + 1 == solver.order`.
    - We assume that non-SDE/ODE cases do not arise.

    This is imperfect as these assumptions may not be true. In addition in the SDE
    case, then solvers will sometimes exhibit higher orders of convergence for specific
    noise types (see issue #47).
    """
    if _is_sde(terms):
        return solver.strong_order + 0.5
    else:
        return solver.order


@eqx.filter_jit
def diffeqsolve(
    terms: PyTree[AbstractTerm],
    solver: AbstractSolver,
    t0: Scalar,
    t1: Scalar,
    dt0: Optional[Scalar],
    y0: PyTree,
    args: Optional[PyTree] = None,
    *,
    saveat: SaveAt = SaveAt(t1=True),
    stepsize_controller: AbstractStepSizeController = ConstantStepSize(),
    adjoint: AbstractAdjoint = RecursiveCheckpointAdjoint(),
    max_steps: Optional[int] = 16**3,
    throw: bool = True,
    solver_state: Optional[PyTree] = None,
    controller_state: Optional[PyTree] = None,
    made_jump: Optional[Bool] = None,
) -> Solution:
    """Solves a differential equation.

    This function is the main entry point for solving all kinds of initial value
    problems, whether they are ODEs, SDEs, or CDEs.

    The differential equation is integrated from `t0` to `t1`.

    **Main arguments:**

    These are the arguments most commonly used day-to-day.

    - `terms`: The terms of the differential equation. This specifies the vector field.
        (For non-ordinary differential equations (SDEs, CDEs), this also specifies the
        Brownian motion or the control.)
    - `solver`: The solver for the differential equation. See the guide on [how to
        choose a solver](../usage/how-to-choose-a-solver.md).
    - `t0`: The start of the region of integration.
    - `t1`: The end of the region of integration.
    - `dt0`: The step size to use for the first step. If using fixed step sizes then
        this will also be the step size for all other steps. (Except the last one,
        which may be slightly smaller and clipped to `t1`.) If set as `None` then the
        initial step size will be determined automatically if possible.
    - `y0`: The initial value. This can be any PyTree of JAX arrays. (Or types that
        can be coerced to JAX arrays, like Python floats.)
    - `args`: Any additional arguments to pass to the vector field.
    - `saveat`: What times to save the solution of the differential equation. Defaults
        to just the last time `t1`. (Keyword-only argument.)
    - `stepsize_controller`: How to change the step size as the integration progresses.
        Defaults to using a fixed constant step size. (Keyword-only argument.)

    **Other arguments:**

    These arguments are infrequently used, and for most purposes you shouldn't need to
    understand these. All of these are keyword-only arguments.

    - `adjoint`: How to backpropagate (and compute forward-mode autoderivatives) of
        `diffeqsolve`. Defaults to discretise-then-optimise with recursive
        checkpointing, which is usually the best option for most problems. See the page
        on [Adjoints](./adjoints.md) for more information.

    - `max_steps`: The maximum number of steps to take before quitting the computation
        unconditionally.

        Can also be set to `None` to allow an arbitrary number of steps, although this
        will disable backpropagation via discretise-then-optimise (backpropagation via
        optimise-then-discretise will still work), and also disables
        `saveat.steps=True` and `saveat.dense=True`.

        Note that (a) compile times; and (b) backpropagation run times; will increase
        as `max_steps` increases. (Specifically, each time `max_steps` passes a power
        of 16.) You can reduce these times by using the smallest value of `max_steps`
        that is reasonable for your problem.

    - `throw`: Whether to raise an exception if the integration fails for any reason.

        If `True` then an integration failure will either raise a `ValueError` (when
        not using `jax.jit`) or print a warning message (when using `jax.jit`).

        If `False` then the returned solution object will have a `result` field
        indicating whether any failures occurred.

        Possible failures include for example hitting `max_steps`, or the problem
        becoming too stiff to integrate. (For most purposes these failures are
        unusual.)

        !!! note

            Note that when `jax.vmap`-ing a differential equation solve, then
            `throw=True` means that an exception will be raised if any batch element
            fails. You may prefer to set `throw=False` and inspect the `result` field
            of the returned solution object, to determine which batch elements
            succeeded and which failed.

    - `solver_state`: Some initial state for the solver. Can be useful when for example
        using a reversible solver to recompute a solution. Generally obtained by
        `SaveAt(solver_state=True)`. It is unlikely you will need to use this option.

    - `controller_state`: Some initial state for the step size controller. Generally
        obtained by `SaveAt(controller_state=True)`. It is unlikely you will need to
        use this option.

    - `made_jump`: Whether a jump has just been made at `t0`. Used to update
        `solver_state` (if passed). It is unlikely you will need to use this option.

    **Returns:**

    Returns a [`diffrax.Solution`][] object specifying the solution to the differential
    equation.

    **Raises:**

    - `ValueError` for bad inputs.
    - `RuntimeError` if `throw=True` and the integration fails (e.g. hitting the
        maximum number of steps).

    !!! note

        It is possible to have `t1 < t0`, in which case integration proceeds backwards
        in time.
    """

    #
    # Initial set-up
    #

    # Error checking
    if dt0 is not None:
        error_if(lambda: (t1 - t0) * dt0 <= 0, "Must have (t1 - t0) * dt0 > 0")

    # Error checking
    term_leaves, term_structure = jax.tree_flatten(
        terms, is_leaf=lambda x: isinstance(x, AbstractTerm)
    )
    raises = False
    for leaf in term_leaves:
        if not isinstance(leaf, AbstractTerm):
            raises = True
        del leaf
    if term_structure != solver.term_structure:
        raises = True
    if raises:
        raise ValueError(
            "`terms` must be a PyTree of `AbstractTerms` (such as `ODETerm`), with "
            f"structure {solver.term_structure}"
        )
    del term_leaves, term_structure, raises

    if _is_sde(terms):
        if not isinstance(solver, (AbstractItoSolver, AbstractStratonovichSolver)):
            warnings.warn(
                f"`{solver.__name__}` is not marked as converging to either the Itô "
                "or the Stratonovich solution."
            )
        if isinstance(adjoint, BacksolveAdjoint):
            if isinstance(solver, AbstractItoSolver):
                raise NotImplementedError(
                    f"`{solver.__name__}` converges to the Itô solution. However "
                    "`BacksolveAdjoint` currently only supports Stratonovich SDEs."
                )
            elif not isinstance(solver, AbstractStratonovichSolver):
                warnings.warn(
                    f"{solver.__name__} is not marked as converging to either the Itô "
                    "or the Stratonovich solution. Note that BacksolveAdjoint will "
                    "only produce the correct solution for Stratonovich SDEs."
                )
        if isinstance(stepsize_controller, AbstractAdaptiveStepSizeController):
            # Specific check to not work even if using HalfSolver(Euler())
            if isinstance(solver, Euler):
                raise ValueError(
                    "An SDE should not be solved with adaptive step sizes with Euler's "
                    "method; it will not converge to the correct solution."
                )
            if not isinstance(solver, AbstractAdaptiveSDESolver):
                raise ValueError(
                    "An adaptive step size controller is being used with a solver "
                    "that does not provide error estimates suitable for SDEs."
                )
    if _is_unsafe_sde(terms):
        if isinstance(stepsize_controller, AbstractAdaptiveStepSizeController):
            raise ValueError(
                "`UnsafeBrownianPath` cannot be used with adaptive step sizes."
            )
        if not isinstance(adjoint, NoAdjoint):
            raise ValueError(
                "`UnsafeBrownianPath` can only be used with `adjoint=NoAdjoint()`."
            )

    # Allow setting e.g. t0 as an int with dt0 as a float. (We need consistent
    # types for JAX to be happy with the bounded_while_loop below.)
    # Use compile-time-eval to avoid turning timelikes into spurious tracers, which
    # inhibit optimisation via compile-time number-of-step inference.
    with jax.ensure_compile_time_eval():
        timelikes = (jnp.array(0.0), t0, t1, dt0, saveat.ts)
        timelikes = [x for x in timelikes if x is not None]
        dtype = jnp.result_type(*timelikes)
        t0 = jnp.asarray(t0, dtype=dtype)
        t1 = jnp.asarray(t1, dtype=dtype)
        if dt0 is not None:
            dt0 = jnp.asarray(dt0, dtype=dtype)
        if saveat.ts is not None:
            saveat = eqx.tree_at(lambda s: s.ts, saveat, saveat.ts.astype(dtype))

    # Time will affect state, so need to promote the state dtype as well if necessary.
    def _promote(yi):
        _dtype = jnp.result_type(yi, *timelikes)  # noqa: F821
        return jnp.asarray(yi, dtype=_dtype)

    y0 = jax.tree_map(_promote, y0)
    del timelikes, dtype

    # Normalises time: if t0 > t1 then flip things around.
    # Once again use compile-time-eval to keep the timelikes non-tracer if possible.
    with jax.ensure_compile_time_eval():
        direction = jnp.where(t0 < t1, 1, -1)
        t0 = t0 * direction
        t1 = t1 * direction
        if dt0 is not None:
            dt0 = dt0 * direction
        if saveat.ts is not None:
            saveat = eqx.tree_at(lambda s: s.ts, saveat, saveat.ts * direction)
    stepsize_controller = stepsize_controller.wrap(direction)
    terms = jax.tree_map(
        lambda t: WrapTerm(t, direction),
        terms,
        is_leaf=lambda x: isinstance(x, AbstractTerm),
    )

    # Stepsize controller gets an opportunity to modify the solver.
    # Note that at this point the solver could be anything so we must check any
    # abstract base classes of the solver before this.
    solver = stepsize_controller.wrap_solver(solver)

    # Error checking
    if saveat.ts is not None:
        error_if(
            saveat.ts[1:] < saveat.ts[:-1],
            "saveat.ts must be increasing or decreasing.",
        )
        error_if(
            (saveat.ts > t1) | (saveat.ts < t0), "saveat.ts must lie between t0 and t1."
        )

    # Initialise states
    tprev = t0
    local_order = _get_local_order(terms, solver)
    if controller_state is None:
        (tnext, controller_state) = stepsize_controller.init(
            terms, t0, t1, y0, dt0, args, solver.func_for_init, local_order
        )
    else:
        if dt0 is None:
            (tnext, _) = stepsize_controller.init(
                terms, t0, t1, y0, dt0, args, solver.func_for_init, local_order
            )
        else:
            tnext = t0 + dt0
    tnext = jnp.minimum(tnext, t1)
    if solver_state is None:
        solver_state = solver.init(terms, t0, tnext, y0, args)

    # Allocate memory to store output.
    out_size = 0
    if saveat.t0:
        out_size += 1
    if saveat.ts is not None:
        out_size += len(saveat.ts)
    if saveat.steps:
        # We have no way of knowing how many steps we'll actually end up taking, and
        # XLA doesn't support dynamic shapes. So we just have to allocate the maximum
        # amount of steps we can possibly take.
        error_if(
            max_steps is None,
            "`max_steps=None` is incompatible with `saveat.steps=True`",
        )
        out_size += max_steps
    if saveat.t1 and not saveat.steps:
        out_size += 1
    num_steps = 0
    num_accepted_steps = 0
    num_rejected_steps = 0
    saveat_ts_index = 0
    save_index = 0
    made_jump = False if made_jump is None else made_jump
    ts = jnp.full(out_size, jnp.inf)
    ys = jax.tree_map(lambda y: jnp.full((out_size,) + jnp.shape(y), jnp.inf), y0)
    result = jnp.array(RESULTS.successful)
    if saveat.dense:
        error_if(t0 == t1, "Cannot save dense output if t0 == t1")
        error_if(
            max_steps is None,
            "`max_steps=None` is incompatible with `saveat.dense=True`",
        )
        (
            _,
            _,
            dense_info,
            _,
            _,
        ) = solver.step(terms, tprev, tnext, y0, args, solver_state, made_jump)
        dense_ts = jnp.full(max_steps + 1, jnp.nan)
        _make_full = lambda x: jnp.full((max_steps,) + jnp.shape(x), jnp.nan)
        dense_infos = jax.tree_map(_make_full, dense_info)
        dense_save_index = 0
    else:
        dense_ts = None
        dense_infos = None
        dense_save_index = None

    # Initialise state
    init_state = _State(
        y=y0,
        tprev=tprev,
        tnext=tnext,
        made_jump=made_jump,
        solver_state=solver_state,
        controller_state=controller_state,
        result=result,
        num_steps=num_steps,
        num_accepted_steps=num_accepted_steps,
        num_rejected_steps=num_rejected_steps,
        saveat_ts_index=saveat_ts_index,
        ts=ts,
        ys=ys,
        save_index=save_index,
        dense_ts=dense_ts,
        dense_infos=dense_infos,
        dense_save_index=dense_save_index,
    )

    #
    # Main loop
    #

    final_state, aux_stats = adjoint.loop(
        args=args,
        terms=terms,
        solver=solver,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        t0=t0,
        t1=t1,
        dt0=dt0,
        max_steps=max_steps,
        throw=throw,
        init_state=init_state,
    )

    #
    # Finish up
    #

    if saveat.t0 or saveat.t1 or saveat.steps or (saveat.ts is not None):
        ts = final_state.ts
        ts = ts * direction
        ys = final_state.ys
        # It's important that we don't do any further postprocessing on `ys` here, as
        # it is the `final_state` value that is used when backpropagating via
        # optimise-then-discretise.
    else:
        ts = None
        ys = None
    if saveat.controller_state:
        controller_state = final_state.controller_state
    else:
        controller_state = None
    if saveat.solver_state:
        solver_state = final_state.solver_state
    else:
        solver_state = None
    if saveat.made_jump:
        made_jump = final_state.made_jump
    else:
        made_jump = None
    if saveat.dense:
        interpolation = DenseInterpolation(
            ts=final_state.dense_ts,
            ts_size=final_state.dense_save_index,
            interpolation_cls=solver.interpolation_cls,
            infos=final_state.dense_infos,
            direction=direction,
        )
    else:
        interpolation = None

    t0 = t0 * direction
    t1 = t1 * direction

    # Store metadata
    compiled_num_steps = aux_stats["compiled_num_steps"]
    stats = {
        "num_steps": final_state.num_steps,
        "num_accepted_steps": final_state.num_accepted_steps,
        "num_rejected_steps": final_state.num_rejected_steps,
        "max_steps": max_steps,
        "compiled_num_steps": compiled_num_steps,
    }
    result = final_state.result
    error_index = unvmap_max(result)
    branched_error_if(
        throw & (result != RESULTS.successful),
        error_index,
        RESULTS.reverse_lookup,
        RuntimeError,
    )

    return Solution(
        t0=t0,
        t1=t1,
        ts=ts,
        ys=ys,
        interpolation=interpolation,
        stats=stats,
        result=result,
        solver_state=solver_state,
        controller_state=controller_state,
        made_jump=made_jump,
    )
