import argparse
import yaml
from src.sender import MessageSender
from src.receiver import MessageReceiver
from src.time_layers import LayerProfiler
from src.controller import Controller

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
    elif args.role == "receiver":
        app = MessageReceiver(config)
    else:
        app = Controller(config)
    app.run()

if __name__ == "__main__":
    start_running()