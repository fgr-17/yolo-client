import socket

def get_ip_address(domain: str):
    return socket.gethostbyname(domain)

class remote_data:
    def __init__(self, hostname, port) -> None:
        self.hostname = hostname
        self.ip = get_ip_address(hostname)
        self.port = port
        pass

    def get_ip(self):
        return self.ip
    
    def get_port(self):
        return self.port
