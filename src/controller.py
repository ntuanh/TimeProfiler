import os
import json
import time
import yaml
import pika
import pickle
import socket


class Controller :
    def __init__(self, config: dict):
        print("Start Controller ...")
        self.config = config
        self.queue_device_1 = config["rabbit"]["queue_device_1"]
        self.queue_device_2 = config["rabbit"]["queue_device_2"]
        self.queue_device_3 = "host_queue"

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
        self.channel.queue_declare(queue=self.queue_device_3, durable=True)

        self.name_devices = []
        self.time_layers = []
        self.num_clients = 2

        self.data = {}

    def send_message(self , message_dict):
        device_queue = self.queue_device_1
        self.channel.basic_publish(exchange='',
                                   routing_key= device_queue ,
                                   body=pickle.dumps(message_dict , )
                                   )

    def listening(self , queue_num = 2):
        device_queue = self.queue_device_2
        if queue_num == 1 :
            device_queue = self.queue_device_1
        elif queue_num == 3:
            device_queue = self.queue_device_3
        method_frame, header_frame, body = self.channel.basic_get(queue=device_queue, auto_ack=True)
        if method_frame and body:
            data = pickle.loads(body)
            return data
        else :
            return None

    def get_num_msg_queue(self , queue_name):
        if queue_name == 2 :
            queue_name = self.queue_device_2
        else :
            queue_name = self.queue_device_1

        queue_info = self.channel.queue_declare(queue=queue_name, passive=True)
        return queue_info.method.message_count


    def run(self):
        count_clients = []
        sent_start = False
        count_layer_times = []
        count_devices = []

        while True :
            data = self.listening()
            if data is not None and data["message"] not in count_clients:
                if data["signal"] == "REQUEST":
                    count_clients.append(data["message"])

            if len(count_clients) == self.num_clients:
                for i in range(self.num_clients) :
                    self.send_message(
                        message_dict= {
                            "signal" : "START",
                            "message" : "."
                        },
                    )

                while True :
                    data = self.listening()
                    if data is not None :
                        signal = data["signal"]
                        if "time_layer" in signal :
                            signal = signal.split(" ")[1]
                            count_devices.append(signal)
                            count_layer_times.append(data["message"])
                        if len(count_devices) == 2 :
                            # print("Layer times ")
                            # print(count_devices)
                            # print(count_layer_times)
                            self.data["name_devices"] = count_devices 
                            self.data["layer_times"] = count_layer_times
                            break
                while True :
                    data = self.listening(queue_num=3)
                    if data is not None :
                        if data["signal"] == "comm_times":
                            print("Get communication times successfully !")
                            # print(data["message"])
                            self.data["comm_times"] = data["message"]
                            break
                break

        # print(self.name_devices)
        print(self.data)
        time.sleep(1)
        self.clean()

    def clean(self):
        """Clear both queues and close channel/connection"""
        try:
            # purge (clear) messages
            self.channel.queue_purge(queue=self.queue_device_1)
            self.channel.queue_purge(queue=self.queue_device_2)
            self.channel.queue_purge(queue=self.queue_device_3)
            print(f"Cleared queues: {self.queue_device_1}, {self.queue_device_2} , {self.queue_device_3}")

            # close channel & connection
            if self.channel.is_open:
                self.channel.close()
            if self.connection.is_open:
                self.connection.close()
            print("Cleaned up RabbitMQ connection.")
        except Exception as e:
            print(f"Error during clean-up: {e}")