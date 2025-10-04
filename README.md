# TimeProfiler

This project provides tools for profiling split inference tasks, including:

1. **Measuring communication time** between two machines during split inference.
2. **Measuring per-layer execution time** on a device, with automatic detection of CPU or GPU usage.
3. **Controller mode** for managing and coordinating split inference experiments.

---

## 1. Measure Communication Time (Split Inference)

**Requirements:**
- Two machines (or two terminals) with network access.
- RabbitMQ server running and accessible (configure in `config.yaml`).

**Steps:**

1. **Edit `config.yaml`**  
   Set RabbitMQ connection details and other parameters as needed.

2. **Start the Receiver**  
   On the first machine/terminal, run:
   ```sh
   python main.py --role receiver 
   ```

3. **Start the Sender**  
   On the second machine/terminal, run:
   ```sh
   python main.py --role sender 
   ```

4. **Results:**  
   Communication times will be saved to `comm_names.csv`.

---

## 2. Measure Per-Layer Execution Time

**Steps:**

1. **Edit `config.yaml`**  
   Set `mode: layer_times` and specify the model path and input shape if needed.

2. **Run the Profiler**  
   On your device, run:
   ```sh
   python main.py
   ```

3. **Results:**  
   Per-layer times and device info will be saved to `layer_times.csv`.

---

## 3. Controller Mode

The controller coordinates split inference experiments, manages configuration, and can trigger sender/receiver processes.

**Steps:**

1. **Edit `config.yaml`**  
   Ensure all devices and experiment parameters are set.

2. **Run the Controller**  
   On the controller machine, run:
   ```sh
   python main.py --role controller
   ```

3. **Results:**  
   The controller will manage the workflow and aggregate results as needed.

---

## Example `config.yaml`

```yaml
mode: communication_times  # or 'layer_times'
model: yolo11n.pt
time_layer:
  num_round: 500
  input_shape: [1, 3, 640, 640]
rabbit:
  queue_device_1: "queue1"
  queue_device_2: "queue2"
  address: "localhost"
  username: "guest"
  password: "guest"
  virtual_host: "/"
controller:
  devices: ["device1", "device2"]
  experiment_name: "split_inference_test"
```

---

## Notes

- The code will automatically detect if it is running on a GPU or CPU.
- For more details, see `main.py` and the modules in the `src` directory.