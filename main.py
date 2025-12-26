import argparse
import yaml
from src.sender import MessageSender
from src.receiver import MessageReceiver
from src.time_layers import LayerProfiler
from src.controller import Controller
from src.handle_data import Data
from src.dijkstra import Dijkstra
from src.Utils import get_output_sizes

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def start_running():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["sender", "receiver", "controller"], required=True)
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.role == "sender":
        app = MessageSender(config)
        app.run()
    elif args.role == "receiver":
        app = MessageReceiver(config)
        app.run()
    else:
        data = Controller(config).run()
        layer_times = data["layer_times"]
        comm_times = data["comm_times"]
        print("Layer times [0]: " , layer_times[0])
        print("Layer times [1]: ", layer_times[1])
        print("Comm times : " , comm_times)
        cost = Data(layer_times , comm_times ).run()
        dijkstra_app = Dijkstra(cost , data["name_devices"])
        dijkstra_app.run()
        print(f"size of layer times 1  {len(layer_times[0])}")
        print(f"size of layer times 2 {len(layer_times[1])}")
        print(f"size of comm times {len(comm_times)}")

if __name__ == "__main__":
    start_running()
    # config = load_config(path="config.yaml")
    # print(get_output_sizes("cfg/yolo11n.yaml"))
