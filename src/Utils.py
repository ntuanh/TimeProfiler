import os
import pandas as pd
import platform

# Optional: use torch if installed to detect GPU
try:
    import torch
    if torch.cuda.is_available():
        device_model = torch.cuda.get_device_name(0)
    else:
        device_model = "cpu"
except ImportError:
    device_model = "cpu"

def write_partial(headers, row, filename="layer_times.csv"):
    # Add machine name + device model
    machine_name = platform.node()
    headers = ["machine_name", "device_model"] + headers
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
