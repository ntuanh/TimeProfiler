import os
import pandas as pd
import socket
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

def write_partial( col_1 , col_2 , headers, row, data_1 ="None" , data_2 = "None", filename="layer_times.csv" ):
    if data_1 == "None" :
        data_1 = socket.gethostname()
    if data_2 == "None":
        data_2 = device_model
    headers = [col_1, col_2] + headers
    row = [data_1 , data_2] + row

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

def get_layer_output(cut_point , yaml_file = "cfg/yolo11n.yaml"):
    """
    Parse YOLO config YAML and return list of 'from' indices for all layers.
    """
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    res = [cut_point - 1]
    from_idx = []
    # backbone and head sections
    for section in ["backbone", "head"]:
        for i, layer in enumerate(config.get(section, [])):
            from_idx.append(layer[0])

    # return from_idx

    for i in range(cut_point , len(from_idx)):
        # print(type(from_idx[0]))
        if isinstance(from_idx[i] , list):
            for j in from_idx[i]:
                if j != -1 and j not in res and j < cut_point :
                    res.append(j)
    res.sort()
    return tuple((cut_point , res))

def get_output_sizes(cfg_path, img_size=(640, 640)):
    """
    Estimate output tensor sizes (MB) for each layer defined in YOLOv11 YAML.
    - Works without torch.
    - Assumes float32 (4 bytes).
    - Detect layer returns 0.0 placeholder unless anchors/nc are specified.
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    dtype_size = 4
    H, W = img_size
    C = 3  # RGB input
    sizes = []
    saved = {}

    # YOLOv11: layers are in backbone + head
    for idx, layer in enumerate(cfg["backbone"] + cfg["head"]):
        from_idx, repeats, module, args = layer
        module = str(module)

        if module in ("Conv", "ConvBnAct"):
            C = args[0]
            stride = args[2] if len(args) > 2 else 1
            H //= stride
            W //= stride
        elif module in ("C2f", "C3", "C3k", "C3k2", "C3TR", "C2PSA"):
            C = args[0]
        elif module == "SPPF":
            C = args[0]
        elif "Upsample" in module:
            scale = args[1] if len(args) > 1 else 2
            H *= scale
            W *= scale
        elif module == "Concat":
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
            sizes.append(0.0)  # can't infer from YAML
            saved[idx] = (C, H, W)
            continue
        else:
            # unknown block → keep previous channels
            pass

        size_mb = round(C * H * W * dtype_size / (1024 * 1024), 2)
        sizes.append(size_mb)
        saved[idx] = (C, H, W)

    for i in range(0 ,len(sizes)):
        from_idx = get_layer_output(i)[1]
        if len(from_idx) == 1 :
            continue
        temp = 0
        for j in range(len(from_idx)):
            temp = temp + sizes[from_idx[j] + 1]
        sizes[i] = temp
    return sizes



