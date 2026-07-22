import numpy as np
import torch
import torch.nn as nn
import time
import pandas as pd
import metric as m
import process_geolife as data_processing
from privkit.utils import constants
import config


@torch.no_grad()
def evaluate_model(model, test_loader, mean_x, std_x, mean_y, std_y, single_point=True, T=50):
    model.eval()               # keep modules in eval (no teacher forcing)
    if T > 1:
        enable_mc_dropout(model)  # but turn on dropout masks

    rows = []
    all_step_errors = []   # list of lists, one per step
    exec_start = time.time()

    device = next(model.parameters()).device

    for batch_idx, (inputs, targets, input_dts, target_dts) in enumerate(test_loader):
        inputs   = inputs.to(device)
        targets  = targets.to(device)
        input_dts_np  = np.array(input_dts)   # shape [B, input_len]
        target_dts_np = np.array(target_dts)  # shape [B, target_len]

        # ---- MC passes ----
        outs = []
        for _ in range(max(1, T)):
            out = model(inputs, targets, teacher_forcing_ratio=0.0)  # [B, K, 2]
            outs.append(out.unsqueeze(0))
        Y = torch.cat(outs, dim=0)   # [T, B, K, 2]  (if T=1 -> shape [1,B,K,2])

        mean = Y.mean(dim=0)                         # [B, K, 2]
        var  = Y.var(dim=0, unbiased=True)           # [B, K, 2] in *normalized* units

        # ---- denormalize mean and variance ----
        # x_norm = (x - mean_x)/std_x => var_x_actual = var_x_norm * std_x^2
        mean_np = mean.cpu().numpy()
        var_np  = var.cpu().numpy()

        pred_x = mean_np[:, :, 0] * std_x + mean_x
        pred_y = mean_np[:, :, 1] * std_y + mean_y

        variance_x = var_np[:, :, 0] * (std_x ** 2)    # variance in original x units
        variance_y = var_np[:, :, 1] * (std_y ** 2)

        # ---- ground-truth in original units ----
        targets_np = targets.cpu().numpy()
        gt_x = targets_np[:, :, 0] * std_x + mean_x
        gt_y = targets_np[:, :, 1] * std_y + mean_y

        # per-step distance error using predictive *mean*
        dist_err = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)  # [B, K]

        # accumulate per-step errors
        K = dist_err.shape[1]
        if not all_step_errors:
            all_step_errors = [[] for _ in range(K)]
        for k in range(K):
            all_step_errors[k].extend(dist_err[:, k].tolist())

        # ---- build rows ----
        inputs_np = inputs.cpu().numpy()
        in_x = inputs_np[:, :, 0] * std_x + mean_x
        in_y = inputs_np[:, :, 1] * std_y + mean_y

        for i in range(inputs.size(0)):
            # input sequence rows
            input_cartesian = np.stack((in_x[i], in_y[i]), axis=-1)
            input_geo = [(lat, lon) for lon, lat in
                         [data_processing.cartesian_to_geographic(x, y) for x, y in input_cartesian]]

            for step, ((lat, lon), datetime) in enumerate(zip(input_geo, input_dts_np[i])):
                rows.append({
                    constants.TID: f"{batch_idx}_{i}",
                    constants.UID: f"user_{i}",
                    constants.DATETIME: datetime,
                    constants.LATITUDE: lat,
                    constants.LONGITUDE: lon,
                    "pred_lat": None,
                    "pred_lon": None,
                    "step": step,
                    "type": "input",
                    "predictability": None
                })

            # predicted sequence rows (use predictive mean)
            pred_cartesian = np.stack((pred_x[i], pred_y[i]), axis=-1)
            pred_geo = [(lat, lon) for lon, lat in
                        [data_processing.cartesian_to_geographic(x, y) for x, y in pred_cartesian]]
            actual_cartesian = np.stack((gt_x[i], gt_y[i]), axis=-1)
            actual_geo = [(lat, lon) for lon, lat in
                          [data_processing.cartesian_to_geographic(x, y) for x, y in actual_cartesian]]

            offset = len(input_geo)
            predictability_seq = m.get_predictability_score_per_point(
                    variance_x[i:i+1], variance_y[i:i+1]
                )[0]  # shape (steps,)
            for j, ((plat, plon), datetime) in enumerate(zip(pred_geo, target_dts_np[i])):
                tlat, tlon = actual_geo[j]

                # predictability proxy from MC spread, mapped to an adaptive epsilon
                predictability = predictability_seq[j]
                epsilon = m.improved_adaptive_epsilon_formula_pred(predictability)

                rows.append({
                    constants.TID: f"{batch_idx}_{i}",
                    constants.UID: f"user_{i}",
                    constants.DATETIME: datetime,
                    constants.LATITUDE: tlat,
                    constants.LONGITUDE: tlon,
                    "pred_lat": plat,
                    "pred_lon": plon,
                    "step": offset + j,
                    "type": "predicted",
                    "predictability": predictability,
                    "epsilon": epsilon
                })

    exec_time = time.time() - exec_start
    df = pd.DataFrame(rows)
    return df, exec_time, all_step_errors


def enable_mc_dropout(model: nn.Module):
    """
    Turn on dropout *modules* (and LSTM internal dropout) while leaving the
    top-level modules in eval() so teacher forcing stays off.
    """
    model.eval()  # keeps teacher forcing off (it is gated on self.training)

    # Re-enable every stochastic module so the MC passes match the dropout the
    # network was trained with: the explicit nn.Dropout layers AND the LSTM's
    # internal (inter-layer) dropout.
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.LSTM)):
            module.train()


def evaluate_model_composition(model, test_loader, mean_x, std_x, mean_y, std_y):
    model.eval()
    rows = []
    all_distance_errors = []

    with torch.no_grad():
        for batch_idx, (inputs, targets, input_dts, target_dts) in enumerate(test_loader):
            inputs, targets = inputs.to(next(model.parameters()).device), targets.to(next(model.parameters()).device)
            outputs = model(inputs, targets)

            inputs_np = inputs.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            input_dts_np  = np.array(input_dts)   # shape [B, input_len]
            target_dts_np = np.array(target_dts)  # shape [B, target_len]

            # Denormalize
            input_x = inputs_np[:, :, 0] * std_x + mean_x
            input_y = inputs_np[:, :, 1] * std_y + mean_y
            predicted_x = outputs_np[:, :, 0] * std_x + mean_x
            predicted_y = outputs_np[:, :, 1] * std_y + mean_y
            actual_x = targets_np[:, :, 0] * std_x + mean_x
            actual_y = targets_np[:, :, 1] * std_y + mean_y

            # Compute distance error in meters
            distance_error_meters = np.sqrt((predicted_x - actual_x) ** 2 + (predicted_y - actual_y) ** 2)
            all_distance_errors.extend(distance_error_meters.tolist())
            for i in range(inputs.size(0)):
                input_cartesian = np.stack((input_x[i], input_y[i]), axis=-1)
                pred_cartesian = np.stack((predicted_x[i], predicted_y[i]), axis=-1)
                actual_cartesian = np.stack((actual_x[i], actual_y[i]), axis=-1)

                input_geo = [(lat, lon) for lon, lat in [data_processing.cartesian_to_geographic(x, y) for x, y in input_cartesian]]
                pred_geo = [(lat, lon) for lon, lat in [data_processing.cartesian_to_geographic(x, y) for x, y in pred_cartesian]]
                actual_geo = [(lat, lon) for lon, lat in [data_processing.cartesian_to_geographic(x, y) for x, y in actual_cartesian]]

                # Add input sequence to rows
                for step, ((lat, lon), datetime) in enumerate(zip(input_geo, input_dts_np[i])):
                    rows.append({
                        constants.TID: f"{batch_idx}_{i}",
                        constants.UID: f"user_{i}",
                        constants.DATETIME: datetime,
                        constants.LATITUDE: lat,
                        constants.LONGITUDE: lon,
                        "pred_lat": None,
                        "pred_lon": None,
                        "step": step,
                        "type": "input",
                        "predictability": None
                    })

                # Add predicted sequence to rows (steps continue after input)
                offset = len(input_geo)
                actual_geo = np.array(actual_geo)
                rows = process_predicted_value_composition(actual_geo, pred_geo, batch_idx, i, offset, rows, target_dts_np[i])

    return pd.DataFrame(rows), all_distance_errors


def process_predicted_value_composition(actual_geo, pred_geo, batch_idx, i, offset, rows, target_dts=None):
    distance_in_traj = m.compute_proximity_between_points(pred_geo)

    if distance_in_traj < config.max_mean_distance:
        centroid_lat, centroid_lon = mean_centroid(
            [lat for lat, lon in pred_geo],
            [lon for lat, lon in pred_geo]
        )
        epsilon = config.epsilon
        predictability = -1
        for j, (true_lat, true_lon) in enumerate(actual_geo):
            datetime = target_dts[j] if target_dts is not None else 0
            rows.append({
                constants.TID: f"{batch_idx}_{i}",
                constants.UID: f"user_{i}",
                constants.DATETIME: datetime,
                constants.LATITUDE: true_lat,
                constants.LONGITUDE: true_lon,
                "pred_lat": centroid_lat,
                "pred_lon": centroid_lon,
                "step": offset + j,
                "type": "close_points",
                "predictability": predictability,
                "epsilon": epsilon
            })
    else:
        for j, (lat, lon) in enumerate(pred_geo):
            pred_lat, pred_lon = lat, lon
            true_lat, true_lon = actual_geo[j]

            epsilon = config.epsilon

            rows.append({
                constants.TID: f"{batch_idx}_{i}",
                constants.UID: f"user_{i}",
                constants.DATETIME: target_dts[j] if target_dts is not None else 0,
                constants.LATITUDE: true_lat,
                constants.LONGITUDE: true_lon,
                "pred_lat": pred_lat,
                "pred_lon": pred_lon,
                "step": offset + j,
                "type": "predicted",
                "predictability": -1,
                "epsilon": epsilon
            })
    return rows


def mean_centroid(latitudes, longitudes):
    """
    Computes the centroid of a set of latitude and longitude points using arithmetic mean.
    This is accurate for small areas (e.g., city blocks).
    """
    return np.mean(latitudes), np.mean(longitudes)
