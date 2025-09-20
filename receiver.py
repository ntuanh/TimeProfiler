import pika
import pickle
import socket


class MessageReceiver:
    def __init__(self, config: dict):
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

    def send_message(self, messages):
        message_with_timestamp = {
            "message": messages,
            "receiver_name" : socket.gethostname()
        }
        print( "receiver_name : " , socket.gethostname())
        self.channel.basic_publish(
            exchange="",
            routing_key=self.queue_device_1,
            body=pickle.dumps(message_with_timestamp),
        )

    def run(self):
        print("Receiver waiting...")

        while True:
            method_frame, _, body = self.channel.basic_get(
                queue=self.queue_device_2, auto_ack=True
            )
            if method_frame and body:
                body = pickle.loads(body)
                if body.get("notify") == "yes":
                    self.send_message(body)
                else:
                    break

        self.channel.queue_delete(queue=self.queue_device_2)
        self.connection.close()
        print("Receiver finished")
