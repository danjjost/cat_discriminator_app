import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from socket import socket
import uuid
import torch.nn.functional as F

import torchvision.transforms as transforms

import os

from PIL import Image

from src.augmentation.data_augmenter import DataAugmenter
from src.cat_discriminator_neural_net import CatDiscriminatorNeuralNet
import torch

from src.cat_discriminator_neural_net import CatDiscriminatorNeuralNet
import socket

from src.models.cat_evaluation_result import CatEvaluationResult


# 🔧 CONFIG
saved_model_path = "trained_networks/cat_discriminator.pth"
uncertain_image_directory = os.getcwd() + "\\data\\uncertain-images"
is_confident_threshold = 0.7
image_size=512

# Connect to an external server to get the LAN IP (doesn't actually send data)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
hostName = s.getsockname()[0]
s.close()
serverPort = 3005


class CustomHTTPServer(HTTPServer):    
    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        print("Initializing server...")

        if not os.path.exists(uncertain_image_directory):
            os.makedirs(uncertain_image_directory)

class EvaluationServer(BaseHTTPRequestHandler):
    def __init__(self, *args, server=None, **kwargs):
        self.initialize_neural_net()
        
        self.server: CustomHTTPServer = server
        BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

    def initialize_neural_net(self):
        self.net = CatDiscriminatorNeuralNet(saved_model_path)
        self.net.cuda()
    
    def load_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            DataAugmenter(target_image_size=image_size, augment_images=False),
            transforms.ToTensor(), # converts numpy to tensor
        ])
        
        image = transform(image)

        return image

    def image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        return encoded_string
    
    def base64_to_image(self, encoded_string) -> bytes:
        return base64.b64decode(encoded_string)
            
    def do_POST(self):        
        content_length = int(self.headers['Content-Length'])
        image_base64 = self.rfile.read(content_length)
        
        image_path = self.save_image(image_base64)

        image = self.load_image(image_path)
        image = image.unsqueeze(0).cuda()

        evaluation_result = self.net.evaluate_single_image(image)

        self.send_evaluation_result(evaluation_result)
        
        if(self.is_confident(evaluation_result)):
            self.delete_file(image_path)

    def send_evaluation_result(self, evaluation_result):
        result_json = evaluation_result.to_json()
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*") 
        self.end_headers()
        self.wfile.write(result_json.encode('utf-8'))

    def save_image(self, image_base64):
        image_id = uuid.uuid4()
        image_path = uncertain_image_directory + "\\" + str(image_id) + ".jpg"
        image_data = self.base64_to_image(image_base64)        

        self.delete_file(image_path)

        with open(image_path, "wb") as image_file:
            image_file.write(image_data)

        return image_path

    def is_confident(self, evaluation_result: CatEvaluationResult) -> bool:
        return evaluation_result.bathrooom_cat_percent > is_confident_threshold or evaluation_result.captain_percent > is_confident_threshold or evaluation_result.control_percent > is_confident_threshold

    def delete_file(self, image_path):
        if os.path.isfile(image_path):
            os.remove(image_path)


if __name__ == "__main__":        
    webServer = CustomHTTPServer((hostName, serverPort), EvaluationServer)

    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
