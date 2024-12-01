import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_rgb

from almj.utils.power_law import (
    exp_decay_model,
    exp_exp_decay_model,
    exp_exp_poly_model,
    exp_functional_form,
    exp_poly_model,
    exp_power_law,
    exp_power_law_single_term_no_constant,
    exp_power_law_two_terms,
    exp_power_law_two_terms_plus_e,
    functional_form,
    power_law,
    power_law_single_term_no_constant,
    power_law_two_terms,
    power_law_two_terms_plus_e,
)


def plot_mean_and_std(
    ax,
    asr_mean,
    asr_std,
    steps,
    log_scale_y=True,
    log_scale_x=True,
    epsilon=1e-5,
    exp_name="ASR",
    color=None,
    linewidth=1,
    use_line=True,
    plot_std_err=True,
    alpha=1,
    fill_alpha=0.3,
    std_scale_factor=1,
    scatter=False,
    linestyle="-",
):
    plot_kwargs = {"label": exp_name, "linewidth": linewidth}
    if color is not None:
        plot_kwargs["color"] = color

    if log_scale_y:
        if use_line:
            if not scatter:
                (line,) = ax.plot(steps, -np.log(asr_mean), **plot_kwargs, alpha=alpha, linestyle=linestyle)
            else:
                # limits indices but keeps look of log scale
                # Create logarithmically spaced indices for sampling
                ranges = [(1, 10, 1), (10, 100, 6), (100, 1000, 36), (1000, 10000, 156)]
                indices = []
                for start, stop, step in ranges:
                    # Only add indices if they're within array bounds
                    if start - 1 < len(steps):
                        end = min(stop, len(steps))  # Don't add 1 to avoid out of bounds
                        indices.extend(range(start, end, step))
                indices = [x - 1 for x in indices]  # Subtract 1 from indices
                # Filter out any indices that would be out of bounds
                indices = [i for i in indices if i < len(asr_mean)]

                scatter = ax.scatter(
                    steps[indices],
                    -np.log(asr_mean)[indices],
                    label=exp_name,
                    color=color,
                    marker="o",
                    alpha=alpha,
                    s=2,
                )
        if plot_std_err:
            ax.fill_between(
                steps,
                -np.log(np.maximum(asr_mean - std_scale_factor * asr_std, epsilon)),
                -np.log(np.maximum(asr_mean + std_scale_factor * asr_std, epsilon)),
                alpha=fill_alpha,
                color=color if not use_line else line.get_color(),
            )
        # ax.set_ylabel("-log(ASR)")
        if log_scale_x:
            ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        if use_line:
            if not scatter:
                (line,) = ax.plot(steps, asr_mean, **plot_kwargs, alpha=alpha, linestyle=linestyle)
            else:
                # limit number of points to plot to every 100
                scatter = ax.scatter(
                    steps[::100], asr_mean[::100], label=exp_name, color=color, marker="o", alpha=alpha, s=2
                )
        if plot_std_err:
            ax.fill_between(
                steps,
                np.maximum(asr_mean - std_scale_factor * asr_std, epsilon),
                np.maximum(asr_mean + std_scale_factor * asr_std, epsilon),
                alpha=fill_alpha,
                color=color if not use_line else line.get_color(),
            )
        ax.set_ylabel("ASR (%)")

    # ax.set_xlabel("N")
    ax.set_xlim(left=1)
    return None


def plot_asr_trajectory(
    ax, asr_traj, steps, log_scale_y=True, log_scale_x=True, epsilon=1e-5, exp_name="ASR", linewidth=1
):
    if log_scale_y:
        scatter = ax.scatter(steps, -np.log(asr_traj), label=exp_name)
        # ax.set_ylabel("-log(ASR)")
        if log_scale_x:
            ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        scatter = ax.scatter(steps, asr_traj, label=exp_name)
        ax.set_ylabel("ASR")
        if log_scale_x:
            ax.set_xscale("log")

    # ax.set_xlabel("N")
    ax.set_xlim(left=1)
    return scatter.get_facecolor()[0]


def calculate_fitted_asr(steps, params, exponential, log_space_fit):
    if len(params) == 2:
        return power_law_single_term_no_constant(steps, *params)
    elif len(params) == 3 and not exponential:
        return power_law(steps, *params)
    elif len(params) == 3 and exponential:
        return exp_decay_model(steps, *params)
    elif len(params) == 4 and not exponential:
        return power_law_two_terms(steps, *params)
    elif len(params) == 4 and exponential:
        return functional_form(steps, *params)
    elif len(params) == 5 and not exponential:
        return power_law_two_terms_plus_e(steps, *params)
    elif len(params) == 5 and exponential:
        return exp_poly_model(steps, *params)
    else:
        raise ValueError(f"Unexpected number of params: {len(params)}")


def plot_fitted_asr(
    ax,
    steps,
    params,
    color,
    log_scale_y=True,
    log_scale_x=True,
    exp_name="",
    exponential=False,
    log_space_fit=False,
    scale_factor=1,
    extrapolate=False,
    linewidth=1,
    linestyle=None,
):
    if len(params) == 2:
        fitted_asr = (
            power_law_single_term_no_constant(steps, *params)
            if log_scale_y
            else exp_power_law_single_term_no_constant(steps, *params)
        )
        linestyle = "--" if linestyle is None else linestyle
        label = f"{exp_name}: fitted"
    elif len(params) == 3 and not exponential:
        fitted_asr = power_law(steps, *params) if log_scale_y else exp_power_law(steps, *params)
        linestyle = "--" if linestyle is None else linestyle
        label = f"{exp_name}: fitted"
    elif len(params) == 3 and exponential:
        fitted_asr = exp_decay_model(steps, *params) if log_scale_y else exp_exp_decay_model(steps, *params)
        linestyle = ":" if linestyle is None else linestyle
        label = f"{exp_name}: fitted"
    elif len(params) == 4 and not exponential:
        fitted_asr = power_law_two_terms(steps, *params) if log_scale_y else exp_power_law_two_terms(steps, *params)
        linestyle = ":" if linestyle is None else linestyle
        label = f"{exp_name}: fitted"
    elif len(params) == 4 and exponential:
        fitted_asr = functional_form(steps, *params) if log_scale_y else exp_functional_form(steps, *params)
        linestyle = ":" if linestyle is None else linestyle
        label = f"{exp_name}: fitted"
    elif len(params) == 5 and not exponential:
        fitted_asr = (
            power_law_two_terms_plus_e(steps, *params)
            if log_scale_y
            else exp_power_law_two_terms_plus_e(steps, *params)
        )
        linestyle = "-." if linestyle is None else linestyle
        label = f"{exp_name}: fitted"
    elif len(params) == 5 and exponential:
        fitted_asr = exp_poly_model(steps, *params) if log_scale_y else exp_exp_poly_model(steps, *params)
        linestyle = "-." if linestyle is None else linestyle
        label = f"{exp_name}: fitted"
    else:
        raise ValueError(f"Unexpected number of params: {len(params)}")

    if extrapolate:
        max_steps = max(steps)
        while min(fitted_asr) > 0.01:  # Extrapolate until ASR reaches 1.0 (100%)
            print(min(fitted_asr))
            max_steps *= 2
            steps = np.arange(1, max_steps + 1)
            fitted_asr = calculate_fitted_asr(steps, params, exponential, log_space_fit)
    ax.plot(steps, fitted_asr * scale_factor, linestyle=linestyle, color=color, label=label, linewidth=linewidth)

    if log_scale_x:
        ax.set_xscale("log")
    if log_scale_y:
        ax.set_yscale("log")

    return steps, fitted_asr


# Function to create lighter shades
def create_lighter_shade(color, factor=0.3):
    rgb = to_rgb(color)
    hsv = rgb_to_hsv(rgb)
    hsv[1] *= 1 - factor  # Reduce saturation
    hsv[2] = 1 - (1 - hsv[2]) * (1 - factor)  # Increase value
    return hsv_to_rgb(hsv)


def adjust_color(color, factor=0.3):
    """
    Adjusts a color by the given factor - lightens or darkens

    Parameters:
    color: RGB tuple with values in range [0,1]
    factor (float): Amount to adjust, between -1 and 1
                   negative = darker, 0 = no change, positive = lighter

    Returns:
    tuple: Adjusted RGB tuple in same [0,1] format
    """
    if factor > 0:
        # Lighten color
        return tuple(min(1.0, c + (1.0 - c) * factor) for c in color)
    else:
        # Darken color
        return tuple(max(0.0, c + (c * factor)) for c in color)