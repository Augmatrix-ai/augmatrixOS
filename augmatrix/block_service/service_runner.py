from twisted.web import server
from twisted.web.resource import Resource
from twisted.internet import reactor
from twisted.internet import endpoints
from abc import ABC, abstractmethod
from .utils import encode, decode, decode_to_object
from augmatrix.datasets import variable_def_to_dataclass
import json

class ServiceRunner(Resource, ABC):
    isLeaf = True

    def __init__(self, structure_json_path):
        with open(structure_json_path, "r") as fr:
            structure = json.loads(fr.read())
            self.func_args_dataclass = variable_def_to_dataclass(structure['func_args_schema'], 'FunctionArguments')
            self.inputs_dataclass = variable_def_to_dataclass(structure['inputs_schema'], 'Inputs')
            self.outputs_dataclass = variable_def_to_dataclass(structure['outputs_schema'], 'Outputs')
            self.block_algo_type = structure.get("algoType", "Map")

    def render(self, request):

        print("recieved ...")
        # Read binary data to MessagePack
        d_data = request.content.read()
        data_msgpack = decode(d_data)
        func_args_data = data_msgpack["func_args"]
        inputs_data = data_msgpack["inputs"]

        # Deserialize func_args and inputs the structure
        func_args = decode_to_object(func_args_data, self.func_args_dataclass)
        inputs = decode_to_object(inputs_data, self.inputs_dataclass)

        # Get various data required to run the program
        outputs = self.run(inputs, json.loads(func_args.properties), json.loads(func_args.credentials))
        outputs_data = []
        if self.block_algo_type == "Splitter":
            outputs_data = [encode(output) for output in outputs]
        else:
            outputs_data = encode(outputs)

        # Write the byte data to the response
        request.write(encode(outputs_data))
        print("responsed ...")

    @abstractmethod
    def run(self, request):
        assert False, "Method not implemented"

class ServerManager:
    def __init__(self, service_runner):
        self.service_runner = service_runner

    def start(self, host="0.0.0.0", port=80):
        print(f"Started service {host}:{port}")
        endpoint_spec = f"tcp:port={port}:interface={host}"

        server_endpoint = endpoints.serverFromString(reactor, endpoint_spec)
        server_endpoint.listen(server.Site(self.service_runner))
        reactor.run()
