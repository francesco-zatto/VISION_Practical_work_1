import numpy as np

def end_point_error(w_r, w_e):
    epe = np.sqrt(np.sum((w_r - w_e) ** 2, axis=2))
    return np.mean(epe), np.std(epe)

def angular_error(w_r, w_e):
    norm_r = np.linalg.norm(w_r, axis=2)
    norm_e = np.linalg.norm(w_e, axis=2)
    ang_error = np.arccos(
        (np.sum(w_r * w_e, axis=2)) / (norm_r * norm_e)
    )
    return np.mean(ang_error), np.std(ang_error)

def norm_error(w_r, w_e):
    norm_r = np.linalg.norm(w_r, axis=2)
    norm_e = np.linalg.norm(w_e, axis=2)
    norm_error = np.abs(norm_r - norm_e)
    return np.mean(norm_error), np.std(norm_error)

def relative_norm_error(w_r, w_e):
    norm_r = np.linalg.norm(w_r, axis=2)
    norm_e = np.linalg.norm(w_e, axis=2)
    rel_norm_error = np.abs(norm_r - norm_e) / (norm_r)
    return np.mean(rel_norm_error), np.std(rel_norm_error)