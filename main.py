import argparse
import yaml
from src.sender import MessageSender
from src.receiver import MessageReceiver
from src.time_layers import LayerProfiler


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def comm_layers():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["sender", "receiver"], required=True)
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.role == "sender":
        app = MessageSender(config)
    else:
        app = MessageReceiver(config)

    app.run()


if __name__ == "__main__":
    config = load_config("config.yaml")
    if config["mode"] == "communication_times":
        comm_layers()
    elif config["mode"] == "layer_times":
        layer_times_app = LayerProfiler(
            model_path= config["model"] ,
            num_runs=config["time_layer"]["num_round"] ,
            input_shape= config["time_layer"]["input_shape"]
        )
        layer_times_app.run()
    else :
        print("Wrong model's name !")

