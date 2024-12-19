import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from socket import socket
from socketserver import BaseServer
import ssl
import time
import uuid
import torch.nn.functional as F

import torchvision.transforms as transforms

import os

from PIL import Image

from cat_discriminator_neural_net import CatDiscriminatorNeuralNet
from src.cats_dataset import CatsDataset
import torch

from cat_discriminator_neural_net import CatDiscriminatorNeuralNet
import socket


# 🔧 CONFIG
trained_model_path = "trained_networks/cat_discriminator_512x512.pth"
working_directory = os.getcwd() + "\\temp"
is_confident_threshold = 0.7

# Connect to an external server to get the LAN IP (doesn't actually send data)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
hostName = s.getsockname()[0]
s.close()
serverPort = 3005

class EvaluationResult:
    def __init__(self) -> None:
        self.control = 0.0
        self.captain = 0.0
        self.bathroom_cat = 0.0

    def to_json(self) -> str:
        return json.dumps({
            'control': self.control,
            'captain': self.captain,
            'bathroom_cat': self.bathroom_cat
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'EvaluationResult':
        data = json.loads(json_str)
        result = cls()
        result.control = data.get('control', 0.0)
        result.captain = data.get('captain', 0.0)
        result.bathroom_cat = data.get('bathroom_cat', 0.0)
        return result

class CustomHTTPServer(HTTPServer):    
    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        print("Initializing server...")

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(((0.5), (0.5), 0.5), ((0.5), (0.5), 0.5))
        ])

        if not os.path.exists(working_directory):
            os.makedirs(working_directory)
            
        self.net = CatDiscriminatorNeuralNet()
        self.net.train(False)

        if os.path.isfile(trained_model_path):
            self.net.load_state_dict(torch.load(trained_model_path, weights_only=True))
        else:
            print("No trained model found at: ", trained_model_path)

        self.net.cuda()

class EvaluationServer(BaseHTTPRequestHandler):
    def __init__(self, *args, server=None, **kwargs):
        self.server: CustomHTTPServer = server
        BaseHTTPRequestHandler.__init__(self, *args, **kwargs)
    
    
    def load_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path)
        image = self.server.transform(image)
        image = image.unsqueeze(0) # makes the image a batch, rather than a single image
        return image
    
    def evaluate_image(self, image: torch.Tensor) -> EvaluationResult:
        outputs = self.server.net(image.to('cuda'))  # Get raw logits
        probabilities = F.softmax(outputs, dim=1)  # Convert logits to probabilities
        
        output_probababilities = probabilities[0]
        result = EvaluationResult()
        result.bathroom_cat = output_probababilities[0].item()
        result.captain = output_probababilities[1].item()
        result.control = output_probababilities[2].item()

        return result

    def image_to_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    
    def base64_to_image(self, encoded_string) -> bytes:
        return base64.b64decode(encoded_string)
            
    def do_POST(self):        
        content_length = int(self.headers['Content-Length'])
        image_base64 = self.rfile.read(content_length)
        
        image_id = uuid.uuid4()
        image_path = working_directory + "\\" + str(image_id) + ".jpg"
        image_data = self.base64_to_image(image_base64)
        print(image_path)

        self.clear_file(image_path)

        with open(image_path, "wb") as image_file:
            image_file.write(image_data)

        image = self.load_image(image_path)

        evaluation_result = self.evaluate_image(image)

        result_json = evaluation_result.to_json()

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*") 
        self.end_headers()
        self.wfile.write(result_json.encode('utf-8'))
        
        if(self.is_confident(evaluation_result)):
            self.clear_file(image_path)

    def is_confident(self, evaluation_result: EvaluationResult) -> bool:
        return evaluation_result.bathroom_cat > is_confident_threshold or evaluation_result.captain > is_confident_threshold or evaluation_result.control > is_confident_threshold

    def clear_file(self, image_path):
        if os.path.isfile(image_path):
            os.remove(image_path)


if __name__ == "__main__":        
    webServer = CustomHTTPServer((hostName, serverPort), EvaluationServer)

    #context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    #context.load_cert_chain(certfile='cert.pem', keyfile='key.pem')

    #webServer.socket = context.wrap_socket(webServer.socket, server_side=True)

    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
