import os
import pandas as pd
import platform
import yaml

# Optional: use torch if installed to detect GPU
try:
    import torch
    if torch.cuda.is_available():
        device_model = torch.cuda.get_device_name(0)
    else:
        device_model = "cpu"
except ImportError:
    device_model = "cpu"

def write_partial(col_1 , col_2 , headers, row, filename="layer_times.csv"):
    # Add machine name + device model
    machine_name = platform.node()
    headers = [col_1, col_2] + headers
    row = [machine_name, device_model] + row

    if not os.path.exists(filename):
        df = pd.DataFrame([row], columns=headers)
        df.to_csv(filename, index=False)
    else:
        df = pd.read_csv(filename)
        # ensure headers are aligned
        if list(df.columns) != headers:
            for h in headers:
                if h not in df.columns:
                    df[h] = ""
            df.to_csv(filename, index=False)

        # append row
        new_row = pd.DataFrame([row], columns=headers)
        new_row.to_csv(filename, mode="a", header=False, index=False)

    print(f"[CSV] Saved row -> {filename}")


def get_output_sizes(cfg_path, img_size=(640, 640)):
    """
    Estimate output tensor sizes (MB) for each layer defined in YOLOv8-style YAML.
    - Only uses config (no torch).
    - Assumes float32 (4 bytes).
    - Detect layer returns 0.0 placeholder unless anchors/nc are specified.
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    dtype_size = 4
    H, W = img_size
    C = 3  # RGB input
    sizes = []

    # Keep track of outputs for skip/concat
    saved = {}

    # process backbone + head
    for idx, layer in enumerate(cfg["backbone"] + cfg["head"]):
        from_idx, repeats, module, args = layer
        module = str(module)

        if module == "Conv":
            C = args[0]
            H //= args[2]  # stride
            W //= args[2]
        elif module == "C2f":
            C = args[0]
        elif module == "SPPF":
            C = args[0]
        elif "Upsample" in module:
            H *= args[1]
            W *= args[1]
        elif module == "Concat":
            # Sum channels of sources
            total_C = 0
            for src in from_idx:
                if src == -1:
                    total_C += saved[idx - 1][0]
                elif isinstance(src, int):
                    total_C += saved[src][0]
                else:
                    raise ValueError(f"Concat source not handled: {src}")
            C = total_C
        elif module == "Detect":
            # Hard to estimate from YAML only
            sizes.append(0.0)
            saved[idx] = (C, H, W)
            continue
        else:
            # fallback (if unknown, just keep channels)
            pass

        size_mb = round(C * H * W * dtype_size / (1024 * 1024), 2)
        sizes.append(size_mb)
        saved[idx] = (C, H, W)

    return sizes

