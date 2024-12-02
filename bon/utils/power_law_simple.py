import numpy as np
from scipy.optimize import curve_fit


def power_law_with_constant(x, a, b, c):
    return a * np.power(x, -b) + c


def exp_power_law_with_constant(x, a, b, c):
    return np.exp(-power_law_with_constant(x, a, b, c))


def linear_model(x_log, a_log, b):
    return a_log - b * x_log


def power_law_no_constant(x, a, b):
    return a * np.power(x, -b)


def exp_power_law_no_constant(x, a, b):
    return np.exp(-power_law_no_constant(x, a, b))


def fit_power_law(
    x: np.ndarray,
    y: np.ndarray,
    fit_type: str = "linear_log_spacing",
    epsilon: float = 1e-5,
    with_constant: bool = False,
    allow_negative_c: bool = False,
    loss: str = "huber",
    skip_first_points: int = 0,
):
    # only pass loss to curve_fit if it's not linear fit
    kwargs = {} if fit_type in ["linear", "linear_log_spacing"] else {"loss": loss}

    if fit_type == "log":
        # fit ax^-b (+c) with mean of trajectories after taking -log(ASR)
        y = -np.log(y + epsilon)
        y = y.mean(axis=0) if y.ndim > 1 else y
        function = power_law_with_constant if with_constant else power_law_no_constant
    elif fit_type == "log_mean_first":
        # same but take mean of trajectories before taking -log(ASR)
        y = y.mean(axis=0) if y.ndim > 1 else y
        y = -np.log(y + epsilon)
        function = power_law_with_constant if with_constant else power_law_no_constant
    elif fit_type == "raw":
        # fit exp(ax^-b (+c)) directly in ASR space
        y = y.mean(axis=0) if y.ndim > 1 else y
        function = exp_power_law_with_constant if with_constant else exp_power_law_no_constant
    elif fit_type == "linear" or fit_type == "linear_log_spacing":
        # fit linear model in log-log space (after taking -log(ASR))
        assert not with_constant, "Linear model does not support a constant term"
        y = -np.log(y + epsilon)
        y = y.mean(axis=0) if y.ndim > 1 else y
        y = np.log(y)
        x = np.log(x)
        function = linear_model
        if fit_type == "linear_log_spacing":
            indices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]) - 1  # fmt: skip
            # cut indicies down if we don't have enough points
            index_too_large = len(indices)
            for i in range(len(indices)):
                if indices[i] > len(x):
                    index_too_large = i
                    break
            indices = indices[:index_too_large]
            # Filter x and y to only include these indices
            x = x[indices]
            y = y[indices]
    else:
        raise ValueError(f"Invalid fit type: {fit_type}")

    if skip_first_points > 0:
        x = x[skip_first_points:]
        y = y[skip_first_points:]

    if with_constant:
        # 3 params: a, b, c
        p0 = [3, 0.3, 0.1]
        # normally we constrain c to be >= 0 so ASR cannot go over 100%
        if allow_negative_c:
            bounds = [(0, 0, -np.inf), (np.inf, np.inf, np.inf)]
        else:
            bounds = [(0, 0, 0), (np.inf, np.inf, np.inf)]
    else:
        # 2 params: a, b
        if fit_type == "linear" or fit_type == "linear_log_spacing":
            p0 = [np.log(3), np.log(0.3)]
            bounds = (-np.inf, np.inf)
        else:
            p0 = [3, 0.3]
            bounds = [(0, 0), (np.inf, np.inf)]
    # remove infs
    mask = ~np.isinf(y) & ~np.isinf(x) & ~np.isnan(y) & ~np.isnan(x)
    if mask.any():
        print(f"Warning: Removed {len(x) - mask.sum()} infs from fit")
        x = x[mask]
        y = y[mask]
    params, _ = curve_fit(function, x, y, p0=p0, bounds=bounds, maxfev=10000, **kwargs)
    if fit_type == "linear" or fit_type == "linear_log_spacing":
        params = [np.exp(params[0]), params[1]]
    return params


def fit_power_law_all_trajectories(
    x: np.ndarray,
    y: np.ndarray,
    fit_type: str = "linear_log_spacing",
    epsilon: float = 1e-5,
    skip_first_points: int = 0,
):
    x = np.log(x)
    if fit_type == "linear_log_spacing":
        indices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]) - 1  # fmt: skip
        # cut indicies down if we don't have enough points
        index_too_large = len(indices)
        for i in range(len(indices)):
            if indices[i] > len(x):
                index_too_large = i
                break
        indices = indices[:index_too_large]
        # Filter x and y to only include these indices
        x = x[indices]
    if skip_first_points > 0:
        x = x[skip_first_points:]

    y = -np.log(y + epsilon)
    all_params = []
    for i in range(len(y)):
        y_i = y[i]
        y_i = np.log(y_i)
        y_i = y_i[indices]

        if skip_first_points > 0:
            y_i = y_i[skip_first_points:]

        p0 = [np.log(3), np.log(0.3)]
        bounds = (-np.inf, np.inf)
        # # remove infs
        # mask = ~np.isinf(y) & ~np.isinf(x) & ~np.isnan(y) & ~np.isnan(x)
        # if mask.any():
        #     print(f"Warning: Removed {len(x) - mask.sum()} infs from fit")
        #     x = x[mask]
        #     y_i = y_i[mask]
        params, _ = curve_fit(linear_model, x, y_i, p0=p0, bounds=bounds, maxfev=10000)
        params = [np.exp(params[0]), params[1]]
        all_params.append(params)
    return all_params
