from twisted.web import server
from twisted.web.resource import Resource
from twisted.internet import reactor
from twisted.internet import endpoints
from abc import ABC, abstractmethod
from augmatrix.block_service.data_context import encode, decode
from augmatrix.datasets import variable_def_to_dataclass
import bson
import json

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

    def __init__(self):
        with open("structure.json", "r") as fr:
            structure = json.loads(fr.read())
            self.func_args_dataclass = variable_def_to_dataclass(structure['func_args_schema'], 'FunctionArguments')
            self.inputs_dataclass = variable_def_to_dataclass(structure['inputs_schema'], 'Inputs')
            self.outputs_dataclass = variable_def_to_dataclass(structure['outputs_schema'], 'Outputs')
            self.block_algo_type = structure.get("algoType", "Map")
        print(self.func_args_dataclass)
        print(self.inputs_dataclass)
        print(self.outputs_dataclass)
        print(self.block_algo_type)

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
        outputs = self.run(inputs, json.loads(func_args.properties), json.loads(func_args.credentials))
        outputs_data = []
        if self.block_algo_type == "Splitter":
            print(outputs)
            outputs_data = [encode(output) for output in outputs]
        else:
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
