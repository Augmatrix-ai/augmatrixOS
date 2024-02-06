import httpx
from augmatrix.block_service.data_context import encode, decode, decode_to_object
from augmatrix.datasets import variable_def_to_dataclass
import json

class ClientRunner:
    def __init__(self, url):
        self.url = url

    def call_function(self, structure_path, func_args, inputs, credentials={}):
        with open(structure_path, "r") as fr_struct:
            structure = json.loads(fr_struct.read())

            func_args_dataclass = variable_def_to_dataclass(structure['func_args_schema'], 'FunctionArguments')
            inputs_dataclass = variable_def_to_dataclass(structure['inputs_schema'], 'Inputs')
            outputs_dataclass = variable_def_to_dataclass(structure['outputs_schema'], 'Outputs')

            func_arguments = func_args_dataclass(properties=json.dumps(func_args), credentials=json.dumps(credentials))
            inputs_instance = inputs_dataclass(**inputs)

            func_args_data = encode(func_arguments)
            inputs_data = encode(inputs_instance)
            data_dict = {'func_args': func_args_data, 'inputs': inputs_data}

            b_data = encode(data_dict)
            response_data = decode(self._send_request(self.url, b_data))

            outputs = {}
            if structure['algoType'] == 'Splitter':
                outputs = [decode_to_object(output, outputs_dataclass) for output in response_data]
            else:
                outputs = decode_to_object(response_data, outputs_dataclass)

            return outputs

    def _send_request(self, url, data, method='POST', content_type='application/msgpack'):
        headers = {'content-type': content_type}
        with httpx.Client(http2=True) as client:
            if method.upper() == 'POST':
                response = client.post(url, content=data, headers=headers, timeout=600)
            else:
                raise ValueError("Unsupported method")
            return response.content