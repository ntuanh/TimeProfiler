import argparse
import yaml
from sender import MessageSender
from receiver import MessageReceiver


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def time_layers():
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
    if config["mode"] == "time_layer":
        time_layers()
    # else :
    #     print(f"Mode : {config["mode"]}")
