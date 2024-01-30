from twisted.web import server
from twisted.web.resource import Resource
from twisted.internet import reactor
from twisted.internet import endpoints
from abc import ABC, abstractmethod
from augmatrix.block_service.data_context import encode, decode 
import bson

def class_to_dict(obj):
    # Get all attributes of the object
    obj_attributes = [attr for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__")]

    # Create a dictionary to store the attributes and their values
    obj_dict = {}
    for attr in obj_attributes:
        obj_dict[attr] = getattr(obj, attr)

    return obj_dict

class ServiceRunner(Resource, ABC):
    isLeaf = True

    def __init__(self, inputs_dataclass, outputs_dataclass, func_args_dataclass):
        self.inputs_dataclass = inputs_dataclass
        self.outputs_dataclass = outputs_dataclass
        self.func_args_dataclass = func_args_dataclass

    def render(self, request):
        # Read binary data to MessagePack
        d_data = request.content.read()
        data_msgpack = bson.loads(d_data)
        func_args_data = data_msgpack["func_args"]
        inputs_data = data_msgpack["inputs"]

        # Deserialize func_args and inputs the structure
        func_args = decode(func_args_data, self.func_args_dataclass)
        inputs = decode(inputs_data, self.inputs_dataclass)

        # Get various data required to run the program
        self.properties = func_args.properties
        self.credentials = func_args.credentials

        outputs = self.run(**inputs.to_dict())
        outputs_data = encode(outputs)

        # Write the byte data to the response
        request.write(outputs_data)

    @abstractmethod
    def run(self, request):
        assert False, "Method not implemented"

class ServerManager:
    def __init__(self, service_runner):
        self.service_runner = service_runner

    def start(self, host="0.0.0.0", port=80):
        site = server.Site(self.service_runner)
        endpoint_spec = f"tcp:port={port}:interface={host}"

        server_endpoint = endpoints.serverFromString(reactor, endpoint_spec)
        server_endpoint.listen(site)
        reactor.run()
 
        server_endpoint = endpoints.serverFromString(reactor, endpoint_spec)
        server_endpoint.listen(site)
        reactor.run()
