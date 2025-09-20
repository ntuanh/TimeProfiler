import os
import json
import time
import yaml
import pika
import pickle
import socket
from src.Utils import write_partial
# device_name = socket.gethostname()


from src.Utils import get_output_sizes

MAX_SIZE_QUEUE = 16777216


class MessageSender:
    def __init__(self, config):
        self.config = config
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=config["rabbit"]["address"],
                credentials=pika.PlainCredentials(
                    config["rabbit"]["username"],
                    config["rabbit"]["password"]
                ),
                virtual_host=config["rabbit"]["virtual_host"]
            )
        )
        self.queue_device_1 = config["rabbit"]["queue_device_1"]
        self.queue_device_2 = config["rabbit"]["queue_device_2"]
        self.channel = self.connection.channel()
        self.queue = "time_layers"
        self.channel.queue_declare(queue=self.queue_device_1 , durable= True)
        self.channel.queue_declare(queue=self.queue_device_2 , durable= True)


        # compute dynamically from yaml
        cfg_path = os.path.join(os.path.dirname(__file__), "cfg", "yolov8.yaml")
        self.size_data = get_output_sizes(cfg_path)

        self.start_time = time.time()
        self.num_round = self.config["time_layer"]["num_round"]

    def send_message(self , messages, notify='yes'):
        message_with_timestamp = {
            "message": messages,
            "notify": notify
        }
        self.channel.basic_publish(exchange='',
                              routing_key=self.queue_device_2,
                              body=pickle.dumps(message_with_timestamp)
                              )

    def run(self):
        num_layer_output = 1
        sender_name = socket.gethostname()
        receiver_name = ""
        header = []
        row = []
        for size in self.size_data:
            num_layer_output += 1
            header.append(f"output_L{num_layer_output}")
            size_bytes = int(size * 1e6)
            if size_bytes >= MAX_SIZE_QUEUE :
                print("Chubby size ")
                row.append(-1)
                continue
            message = '1' * size_bytes
            avg_time = 0.0
            for _ in range(self.num_round):
                time_old = time.time_ns()
                self.send_message(message, 'yes')
                while True:
                    method_frame, header_frame, body = self.channel.basic_get(queue=self.queue_device_1, auto_ack=True)
                    if method_frame and body:
                        time_new = time.time_ns()
                        t = time_new - time_old
                        avg_time += t / 2
                        receiver_name = body.get("receiver_name")
                        break
                    else:
                        continue
            avg_time = avg_time / self.num_round
            time_ms = avg_time / 1e6
            row.append(f"{time_ms:.3f} ms")
            print(f"Layer {num_layer_output}: {time_ms:.3f} ms")
        self.send_message('', 'no')
        write_partial(sender_name , receiver_name , header , row , "comm_names.csv")


def time_layers(config):
    sender = MessageSender(config)
    sender.run()
