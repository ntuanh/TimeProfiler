import pika
import time
import pickle
from tqdm import tqdm


class MessageSender:
    def __init__(self, config: dict):
        self.rounds = config["time_layer"]["num_round"]
        self.queue_device_1 = config["rabbit"]["queue_device_1"]
        self.queue_device_2 = config["rabbit"]["queue_device_2"]

        credentials = pika.PlainCredentials(
            config["rabbit"]["username"], config["rabbit"]["password"]
        )
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                config["rabbit"]["address"],
                5672,
                config["rabbit"]["virtual_host"],
                credentials,
            )
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue_device_1, durable=True)
        self.channel.queue_declare(queue=self.queue_device_2, durable=True)

        self.size_data = [i * 10**6 for i in range(1, 10)]
        self.layer_comm = []

    def send_message(self, messages, notify="yes"):
        message_with_timestamp = {"message": messages, "notify": notify}
        self.channel.basic_publish(
            exchange="",
            routing_key=self.queue_device_2,
            body=pickle.dumps(message_with_timestamp),
        )

    def run(self):
        print("Sender running...")
        for size in self.size_data:
            message = "1" * size
            avg_time = 0.0

            for _ in tqdm(range(self.rounds), desc=f"Size {size}"):
                time_old = time.time_ns()
                self.send_message(message, "yes")

                while True:
                    method_frame, _, body = self.channel.basic_get(
                        queue=self.queue_device_1, auto_ack=True
                    )
                    if method_frame and body:
                        time_new = time.time_ns()
                        avg_time += (time_new - time_old) / 2
                        break

            avg_time = avg_time / self.rounds
            self.layer_comm.append(avg_time)

        self.send_message("", "no")
        self.channel.queue_delete(queue=self.queue_device_1)
        self.connection.close()
        print("Sender finished:", self.layer_comm)
