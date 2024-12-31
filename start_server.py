from socket import socket

from evaluation_server import CustomHTTPServer, EvaluationServer, EvaluationServerConfig

import socket



# 🔧 CONFIG
evaluation_server_config = EvaluationServerConfig()
serverPort = 3005

def get_host_name():
    # Connect to an external server to get the LAN IP (doesn't actually send data)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    hostName = s.getsockname()[0]
    s.close()
    return hostName

if __name__ == "__main__":        
    host_name = get_host_name()
    web_server = CustomHTTPServer(evaluation_server_config, (host_name, serverPort), EvaluationServer)

    print("Server started http://%s:%s" % (host_name, serverPort))

    try:
        web_server.serve_forever()
    except KeyboardInterrupt:
        pass

    web_server.server_close()
    print("Server stopped.")
