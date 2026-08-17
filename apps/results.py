"""Read-only Marimo dashboard for saved orbit-superposition runs.

This app deliberately has no path back into the optimization or orbit-integration
workflow.  Missing artifacts are reported in the UI and are never recomputed.
"""

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    from collections.abc import Mapping
    from pathlib import Path

    import marimo as mo
    import numpy as np

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    return Mapping, Path, mo, np, plt


@app.cell
def _():
    try:
        import halo_mw_lmc.artifacts as artifact_api
    except (ImportError, AttributeError) as exc:
        artifact_api = None
        artifact_api_error = f"Artifact reader is unavailable: {exc}"
    else:
        required = (
            "discover_runs",
            "load_run_summary",
            "load_best_evaluation",
        )
        missing = [name for name in required if not hasattr(artifact_api, name)]
        if missing:
            artifact_api_error = (
                "Artifact reader is incomplete; missing: " + ", ".join(missing)
            )
        else:
            artifact_api_error = None

    def discover_run_artifacts(root):
        if artifact_api is None or artifact_api_error is not None:
            return []
        return artifact_api.discover_runs(root)

    def read_run_summary(run_dir):
        if artifact_api is None or artifact_api_error is not None:
            raise RuntimeError(artifact_api_error or "Artifact reader is unavailable")
        return artifact_api.load_run_summary(run_dir)

    def read_best_evaluation(run_dir):
        if artifact_api is None or artifact_api_error is not None:
            raise RuntimeError(artifact_api_error or "Artifact reader is unavailable")
        return artifact_api.load_best_evaluation(run_dir)

    return (
        artifact_api_error,
        discover_run_artifacts,
        read_best_evaluation,
        read_run_summary,
    )


@app.cell
def _(Mapping, np):
    def value_from(value, *names, default=None):
        """Read a field from a mapping, npz-like object, or attribute object."""

        if value is None:
            return default
        for name in names:
            if isinstance(value, Mapping) and name in value:
                return value[name]
            structured_names = (
                getattr(getattr(value, "dtype", None), "names", None) or ()
            )
            if name in structured_names:
                return value[name]
            files = getattr(value, "files", ())
            if name in files:
                return value[name]
            if hasattr(value, name):
                return getattr(value, name)
        return default

    def scalar_text(value, *, digits=5, missing="—"):
        if value is None:
            return missing
        try:
            array = np.asarray(value)
            if array.size != 1:
                return str(value)
            item = array.reshape(()).item()
        except (TypeError, ValueError):
            item = value
        if isinstance(item, (float, np.floating)):
            return f"{float(item):.{digits}g}" if np.isfinite(item) else str(item)
        return str(item)

    def column_from(table, name):
        if table is None:
            return np.array([], dtype=float)
        if isinstance(table, Mapping) and name in table:
            return np.asarray(table[name])
        names = getattr(getattr(table, "dtype", None), "names", None) or ()
        if name in names:
            return np.asarray(table[name])
        if hasattr(table, name):
            return np.asarray(getattr(table, name))
        try:
            rows = list(table)
        except TypeError:
            return np.array([], dtype=float)
        if rows and all(isinstance(row, Mapping) and name in row for row in rows):
            return np.asarray([row[name] for row in rows])
        return np.array([], dtype=float)

    def row_as_mapping(table, index):
        if table is None:
            return {}
        if isinstance(table, Mapping):
            result = {}
            for name, values in table.items():
                try:
                    result[name] = np.asarray(values)[index].item()
                except (IndexError, TypeError, ValueError, AttributeError):
                    continue
            return result
        names = getattr(getattr(table, "dtype", None), "names", None) or ()
        if names:
            row = table[index]
            result = {}
            for name in names:
                item = row[name]
                result[name] = item.item() if hasattr(item, "item") else item
            return result
        try:
            row = list(table)[index]
        except (IndexError, TypeError):
            return {}
        return dict(row) if isinstance(row, Mapping) else {}

    def best_row_from(table):
        objective = column_from(table, "objective")
        if objective.size == 0:
            return {}
        try:
            numeric = np.asarray(objective, dtype=float)
        except (TypeError, ValueError):
            return {}
        finite = np.flatnonzero(np.isfinite(numeric))
        if finite.size == 0:
            return {}
        index = int(finite[np.argmin(numeric[finite])])
        return row_as_mapping(table, index)

    def valid_cube(value):
        if value is None:
            return None
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None
        return array if array.ndim == 3 and array.shape[2] > 0 else None

    return best_row_from, column_from, scalar_text, valid_cube, value_from


@app.cell
def _(Path, mo):
    project_root = Path(__file__).resolve().parents[1]
    configured_runs_root = project_root / "runs"
    default_runs_root = (
        configured_runs_root if configured_runs_root.is_dir() else project_root
    )
    runs_root_input = mo.ui.text(
        value=str(default_runs_root),
        label="Runs root",
        full_width=True,
    )
    direct_run_input = mo.ui.text(
        value="",
        label="Run path override (optional)",
        full_width=True,
    )
    return direct_run_input, runs_root_input


@app.cell
def _(
    Path,
    discover_run_artifacts,
    mo,
    runs_root_input,
    value_from,
):
    runs_root = Path(runs_root_input.value).expanduser()
    try:
        discovered = discover_run_artifacts(runs_root)
        discovery_error = None
    except Exception as exc:
        discovered = []
        discovery_error = f"Could not discover runs under {runs_root}: {exc}"

    discovered_paths = []
    for item in discovered or ():
        candidate = value_from(item, "run_dir", "path", default=item)
        try:
            path = Path(candidate).expanduser().resolve()
        except (TypeError, ValueError, OSError):
            continue
        if path not in discovered_paths:
            discovered_paths.append(path)
    discovered_paths.sort(key=lambda path: path.name)

    run_options = [str(path) for path in discovered_paths]
    run_dropdown = mo.ui.dropdown(
        options=run_options or [""],
        value=run_options[0] if run_options else "",
        label="Discovered run",
        full_width=True,
    )
    return discovery_error, run_dropdown, runs_root


@app.cell
def _(Path, direct_run_input, run_dropdown):
    direct_value = direct_run_input.value.strip()
    selected_value = direct_value or (run_dropdown.value or "").strip()
    try:
        selected_run_dir = (
            Path(selected_value).expanduser().resolve() if selected_value else None
        )
        selection_error = None
    except (OSError, ValueError) as exc:
        selected_run_dir = None
        selection_error = f"Invalid run path: {exc}"
    return selected_run_dir, selection_error


@app.cell
def _(read_best_evaluation, read_run_summary, selected_run_dir):
    if selected_run_dir is None:
        run_summary = None
        summary_error = "Select a saved run to view its results."
        best_evaluation = None
        best_evaluation_error = None
    else:
        try:
            run_summary = read_run_summary(selected_run_dir)
            summary_error = None
        except Exception as exc:
            run_summary = None
            summary_error = f"Could not load run summary: {exc}"

        if run_summary is None:
            best_evaluation = None
            best_evaluation_error = (
                "Best-evaluation display is disabled until the run summary "
                "passes schema and consistency checks."
            )
        else:
            try:
                best_evaluation = read_best_evaluation(selected_run_dir)
                best_evaluation_error = None
            except Exception as exc:
                best_evaluation = None
                best_evaluation_error = (
                    "No readable best-evaluation snapshot is available. "
                    f"Density comparison is shown only when already saved ({exc})."
                )
    return (
        best_evaluation,
        best_evaluation_error,
        run_summary,
        summary_error,
    )


@app.cell
def _(best_row_from, column_from, run_summary, value_from):
    run_config = value_from(
        run_summary,
        "run_config",
        "config",
        "resolved_config",
        default={},
    )
    samples = value_from(
        run_summary,
        "samples",
        "sample_table",
        "trials",
        default=None,
    )
    weight_audit = value_from(
        run_summary,
        "weight_audit",
        "fixed_weight_audit",
        "fixed_weights",
        default=None,
    )
    best_row = value_from(run_summary, "best_sample", "best_row", default=None)
    if best_row is None:
        best_row = best_row_from(samples)
    trial_count = int(column_from(samples, "objective").size)
    return best_row, run_config, samples, trial_count, weight_audit


@app.cell
def _(
    best_evaluation,
    np,
    valid_cube,
    value_from,
    weight_audit,
):
    density_block = value_from(
        best_evaluation,
        "density",
        "density_comparison",
        default=best_evaluation,
    )
    density_grid = value_from(density_block, "grid", default=None)

    target_density = valid_cube(
        value_from(
            density_block,
            "data_density",
            "target_density",
            "density_data_density",
            default=None,
        )
    )
    if target_density is None:
        target_density = valid_cube(
            value_from(weight_audit, "target_density", default=None)
        )
    model_density = valid_cube(
        value_from(
            density_block,
            "model_density",
            "density_model_density",
            default=None,
        )
    )
    residual_density = valid_cube(
        value_from(
            density_block,
            "residual",
            "density_residual",
            default=None,
        )
    )

    cube_candidates = [
        cube for cube in (target_density, model_density, residual_density) if cube is not None
    ]
    density_nphi = cube_candidates[0].shape[2] if cube_candidates else 1
    consistent_cubes = all(cube.shape[2] == density_nphi for cube in cube_candidates)

    r_edges = value_from(
        density_block,
        "r_edges",
        "density_r_edges",
        default=value_from(
            density_grid,
            "r_edges",
            default=value_from(weight_audit, "r_edges", default=None),
        ),
    )
    z_edges = value_from(
        density_block,
        "z_edges",
        "density_z_edges",
        default=value_from(
            density_grid,
            "z_edges",
            default=value_from(weight_audit, "z_edges", default=None),
        ),
    )
    phi_edges = value_from(
        density_block,
        "phi_edges",
        "density_phi_edges",
        default=value_from(
            density_grid,
            "phi_edges",
            default=value_from(weight_audit, "phi_edges", default=None),
        ),
    )
    try:
        r_edges = np.asarray(r_edges, dtype=float) if r_edges is not None else None
        z_edges = np.asarray(z_edges, dtype=float) if z_edges is not None else None
        phi_edges = np.asarray(phi_edges, dtype=float) if phi_edges is not None else None
    except (TypeError, ValueError):
        r_edges = z_edges = phi_edges = None
    return (
        consistent_cubes,
        density_nphi,
        model_density,
        phi_edges,
        r_edges,
        residual_density,
        target_density,
        z_edges,
    )


@app.cell
def _(density_nphi, mo, phi_edges):
    phi_labels = []
    for index in range(density_nphi):
        if phi_edges is not None and phi_edges.size == density_nphi + 1:
            import math

            lower = math.degrees(float(phi_edges[index]))
            upper = math.degrees(float(phi_edges[index + 1]))
            phi_labels.append(f"φ {index}: {lower:.0f}° to {upper:.0f}°")
        else:
            phi_labels.append(f"φ bin {index}")
    phi_selector = mo.ui.dropdown(
        options={label: index for index, label in enumerate(phi_labels)},
        value=phi_labels[0],
        label="Density / weight φ bin",
    )
    return (phi_selector,)


@app.cell
def _(
    best_row,
    mo,
    run_config,
    scalar_text,
    selected_run_dir,
    trial_count,
    value_from,
):
    optimizer_config = value_from(run_config, "optimizer", default={})
    grid_config = value_from(
        run_config,
        "density_grid",
        "grid",
        default={},
    )
    velocity_config = value_from(run_config, "velocity_fit", default={})
    weight_config = value_from(run_config, "weight_model", default={})
    objective_config = value_from(run_config, "objective", default={})
    schema_version = value_from(run_config, "schema_version", default=None)
    include_velocity = value_from(
        run_config,
        "include_velocity",
        default=value_from(velocity_config, "enabled", default=None),
    )
    git_commit = value_from(run_config, "git_commit", default=None)
    configured_phi_edges = value_from(
        grid_config,
        "phi_edges_rad",
        default=None,
    )
    configured_nphi = value_from(grid_config, "n_phi", default=None)
    if configured_nphi is None and configured_phi_edges is not None:
        try:
            configured_nphi = max(len(configured_phi_edges) - 1, 0)
        except TypeError:
            configured_nphi = None

    parameter_names = (
        "qhalo",
        "phalo",
        "rho0",
        "rho0_plus_2logrs",
        "gamma",
    )
    parameter_rows = []
    for name in parameter_names:
        value = value_from(best_row, name, default=None)
        if value is not None:
            parameter_rows.append(f"| `{name}` | {scalar_text(value)} |")
    if not parameter_rows:
        parameter_rows.append("| — | No finite best trial is available |")

    overview_view = mo.md(
        f"""
        ## Overview

        | Field | Value |
        |---|---|
        | Run | `{selected_run_dir or '—'}` |
        | Artifact schema | {scalar_text(schema_version)} |
        | Trials | {trial_count} |
        | Best iteration | {scalar_text(value_from(best_row, 'iteration'))} |
        | Best objective | {scalar_text(value_from(best_row, 'objective'))} |
        | Velocity-only objective | {scalar_text(value_from(best_row, 'objective_velocity'))} |
        | Density + velocity objective | {scalar_text(value_from(best_row, 'objective_density_velocity'))} |
        | Density χ² | {scalar_text(value_from(best_row, 'chi2'))} |
        | Density χ² / fitted bin | {scalar_text(value_from(best_row, 'density_chi2_per_bin'))} |
        | Density scale | {scalar_text(value_from(best_row, 'density_scale'))} |
        | Weight mode | {scalar_text(value_from(weight_config, 'mode'))} |
        | Objective mode | {scalar_text(value_from(objective_config, 'mode'))} |
        | Effective orbit count | {scalar_text(value_from(best_row, 'effective_orbit_count'))} |
        | Maximum weight fraction | {scalar_text(value_from(best_row, 'max_weight_fraction'))} |
        | Active orbit count | {scalar_text(value_from(best_row, 'active_orbit_count'))} |
        | Weight solver converged | {scalar_text(value_from(best_row, 'weight_solver_converged'))} |
        | Velocity enabled | {scalar_text(include_velocity)} |
        | Requested iterations | {scalar_text(value_from(optimizer_config, 'iterations'))} |
        | Density φ bins | {scalar_text(configured_nphi)} |
        | Git commit | `{scalar_text(git_commit)}` |

        ### Best saved parameters

        | Parameter | Value |
        |---|---:|
        {chr(10).join(parameter_rows)}
        """
    )
    return (overview_view,)


@app.cell
def _(column_from, mo, np, plt, samples):
    if plt is None:
        convergence_view = mo.md(
            "## Convergence\n\nMatplotlib is unavailable; the saved samples were not modified."
        )
    else:
        iteration = column_from(samples, "iteration")
        objective = column_from(samples, "objective")
        try:
            iteration = np.asarray(iteration, dtype=float)
            objective = np.asarray(objective, dtype=float)
        except (TypeError, ValueError):
            iteration = objective = np.array([], dtype=float)

        finite = np.isfinite(iteration) & np.isfinite(objective)
        if not np.any(finite):
            convergence_view = mo.md(
                "## Convergence\n\nNo finite saved iteration/objective pairs are available."
            )
        else:
            iteration = iteration[finite]
            objective = objective[finite]
            order = np.argsort(iteration)
            iteration = iteration[order]
            objective = objective[order]
            best_so_far = np.minimum.accumulate(objective)

            convergence_figure, convergence_axis = plt.subplots(
                figsize=(9.0, 4.8), constrained_layout=True
            )
            convergence_axis.scatter(
                iteration,
                objective,
                s=12,
                alpha=0.4,
                label="saved trial",
            )
            convergence_axis.plot(
                iteration,
                best_so_far,
                color="tab:red",
                linewidth=1.8,
                label="best so far",
            )
            convergence_axis.set_xlabel("Iteration")
            convergence_axis.set_ylabel("Objective")
            convergence_axis.set_yscale(
                "log" if np.all(objective > 0) else "symlog"
            )
            convergence_axis.grid(alpha=0.2)
            convergence_axis.legend()
            convergence_axis.set_title("Optimization convergence (saved samples only)")
            convergence_view = mo.vstack(
                [mo.md("## Convergence"), convergence_figure]
            )
    return (convergence_view,)


@app.cell
def _(
    consistent_cubes,
    mo,
    model_density,
    np,
    phi_selector,
    plt,
    r_edges,
    residual_density,
    target_density,
    z_edges,
):
    density_cubes = [
        ("Target log₁₀ density", target_density, "viridis"),
        ("Best model log₁₀ density", model_density, "viridis"),
        ("Standardized residual", residual_density, "RdBu_r"),
    ]
    available_density = [item for item in density_cubes if item[1] is not None]

    if plt is None:
        density_view = mo.md(
            "## Density by φ\n\nMatplotlib is unavailable; no density plot was generated."
        )
    elif not available_density:
        density_view = mo.md(
            "## Density by φ\n\nNo saved density cube is available. The app will not recompute one."
        )
    elif not consistent_cubes:
        density_view = mo.md(
            "## Density by φ\n\nSaved density cubes have inconsistent φ dimensions."
        )
    else:
        phi_index = int(phi_selector.value or 0)
        columns = len(available_density)
        density_figure, density_axes = plt.subplots(
            1,
            columns,
            figsize=(5.0 * columns, 4.5),
            constrained_layout=True,
            squeeze=False,
        )
        axes = density_axes[0]

        extent = None
        first_cube = available_density[0][1]
        if (
            r_edges is not None
            and z_edges is not None
            and r_edges.size == first_cube.shape[0] + 1
            and z_edges.size == first_cube.shape[1] + 1
        ):
            extent = [r_edges[0], r_edges[-1], z_edges[0], z_edges[-1]]

        log_parts = []
        for _, cube, _ in available_density:
            if cube is residual_density:
                continue
            positive = cube[:, :, phi_index]
            positive = positive[np.isfinite(positive) & (positive > 0)]
            if positive.size:
                log_parts.append(np.log10(positive))
        if log_parts:
            all_log_density = np.concatenate(log_parts)
            density_limits = np.nanpercentile(all_log_density, [2, 98])
        else:
            density_limits = (-1.0, 1.0)

        for axis, (title, cube, cmap) in zip(axes, available_density):
            density_values = cube[:, :, phi_index].T
            if cube is residual_density:
                finite_residual = np.abs(
                    density_values[np.isfinite(density_values)]
                )
                limit = (
                    float(np.nanpercentile(finite_residual, 98))
                    if finite_residual.size
                    else 1.0
                )
                limit = max(limit, 1.0e-12)
                shown = density_values
                image = axis.imshow(
                    shown,
                    origin="lower",
                    aspect="auto",
                    extent=extent,
                    cmap=cmap,
                    vmin=-limit,
                    vmax=limit,
                )
            else:
                shown = np.full_like(density_values, np.nan, dtype=float)
                np.log10(
                    density_values,
                    out=shown,
                    where=density_values > 0,
                )
                image = axis.imshow(
                    shown,
                    origin="lower",
                    aspect="auto",
                    extent=extent,
                    cmap=cmap,
                    vmin=float(density_limits[0]),
                    vmax=float(density_limits[1]),
                )
            axis.set_title(title)
            axis.set_xlabel("R [kpc]")
            axis.set_ylabel("z [kpc]")
            density_figure.colorbar(image, ax=axis, shrink=0.82)
        density_figure.suptitle(f"Saved density comparison — φ bin {phi_index}")
        density_view = mo.vstack(
            [mo.md("## Density by φ"), phi_selector, density_figure]
        )
    return (density_view,)


@app.cell
def _(
    best_evaluation,
    mo,
    np,
    phi_selector,
    plt,
    r_edges,
    scalar_text,
    value_from,
    weight_audit,
    z_edges,
):
    best_weight_solution = value_from(
        best_evaluation,
        "weight_solution",
        default=None,
    )
    best_weight_metrics = (
        ("Inner objective", "inner_objective"),
        ("Regularization penalty", "regularization_penalty"),
        ("Effective orbit count", "effective_orbit_count"),
        ("Largest weight fraction", "maximum_weight_fraction"),
        ("Active orbit count", "active_orbit_count"),
        ("Solver converged", "converged"),
        ("Solver status", "status"),
        ("Solver message", "message"),
    )
    best_metric_rows = [
        f"| {label} | {scalar_text(value_from(best_weight_solution, key))} |"
        for label, key in best_weight_metrics
    ]
    weight_metrics = (
        ("Positive seed count", "positive_seed_count"),
        ("Seeds inside density grid", "in_grid_seed_count"),
        ("Total fixed weight", "total_weight"),
        ("Weight inside density grid", "in_grid_weight"),
        ("Global effective seed count", "effective_seed_count"),
        ("Largest global weight fraction", "max_weight_fraction"),
    )
    metric_rows = [
        f"| {label} | {scalar_text(value_from(weight_audit, key))} |"
        for label, key in weight_metrics
    ]

    quantile_levels = value_from(weight_audit, "quantile_levels", default=None)
    weight_quantiles = value_from(weight_audit, "weight_quantiles", default=None)
    quantile_rows = []
    if quantile_levels is not None and weight_quantiles is not None:
        try:
            levels = np.asarray(quantile_levels, dtype=float)
            quantile_values = np.asarray(weight_quantiles, dtype=float)
            if levels.shape == quantile_values.shape:
                quantile_rows = [
                    f"| {100 * level:.0f}% | {scalar_text(value)} |"
                    for level, value in zip(levels, quantile_values)
                ]
        except (TypeError, ValueError):
            quantile_rows = []

    weight_markdown = mo.md(
        f"""
        ## Best-trial weight solution

        | Metric | Value |
        |---|---:|
        {chr(10).join(best_metric_rows)}

        ## Fixed catalogue / input audit

        | Metric | Value |
        |---|---:|
        {chr(10).join(metric_rows)}

        ### Catalogue-weight quantiles

        | Quantile | Fixed weight |
        |---|---:|
        {chr(10).join(quantile_rows) if quantile_rows else '| — | Not saved |'}
        """
    )

    audit_cube = value_from(
        weight_audit,
        "cell_max_weight_fraction",
        default=None,
    )
    try:
        audit_cube = np.asarray(audit_cube, dtype=float)
    except (TypeError, ValueError):
        audit_cube = np.array([])

    if plt is None or audit_cube.ndim != 3 or audit_cube.shape[2] == 0:
        weight_view = weight_markdown
    else:
        audit_phi = min(int(phi_selector.value or 0), audit_cube.shape[2] - 1)
        audit_extent = None
        if (
            r_edges is not None
            and z_edges is not None
            and r_edges.size == audit_cube.shape[0] + 1
            and z_edges.size == audit_cube.shape[1] + 1
        ):
            audit_extent = [r_edges[0], r_edges[-1], z_edges[0], z_edges[-1]]
        weight_figure, weight_axis = plt.subplots(
            figsize=(6.4, 5.0), constrained_layout=True
        )
        weight_image = weight_axis.imshow(
            audit_cube[:, :, audit_phi].T,
            origin="lower",
            aspect="auto",
            extent=audit_extent,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )
        weight_axis.set_xlabel("R [kpc]")
        weight_axis.set_ylabel("z [kpc]")
        weight_axis.set_title(
            f"Largest seed-weight fraction in each cell — φ bin {audit_phi}"
        )
        weight_figure.colorbar(weight_image, ax=weight_axis, label="max / total")
        weight_view = mo.vstack([weight_markdown, weight_figure])
    return (weight_view,)


@app.cell
def _(
    artifact_api_error,
    best_evaluation_error,
    convergence_view,
    density_view,
    direct_run_input,
    discovery_error,
    mo,
    overview_view,
    run_dropdown,
    runs_root,
    runs_root_input,
    selection_error,
    summary_error,
    weight_view,
):
    messages = [
        message
        for message in (
            artifact_api_error,
            discovery_error,
            selection_error,
            summary_error,
            best_evaluation_error,
        )
        if message
    ]
    notices = (
        mo.md("\n".join(f"> **Notice:** {message}" for message in messages))
        if messages
        else mo.md("> Loaded saved artifacts in read-only mode. No model was recomputed.")
    )
    selectors = mo.vstack(
        [
            mo.md(f"**Run discovery root:** `{runs_root}`"),
            runs_root_input,
            run_dropdown,
            direct_run_input,
        ]
    )
    tabs = mo.ui.tabs(
        {
            "Overview": overview_view,
            "Convergence": convergence_view,
            "Density by φ": density_view,
            "Weights": weight_view,
        }
    )
    mo.vstack(
        [
            mo.md(
                "# Orbit-superposition run results\n\n"
                "This dashboard reads saved run artifacts only. It never integrates "
                "orbits, evaluates a potential, or writes into the run directory."
            ),
            selectors,
            notices,
            tabs,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
