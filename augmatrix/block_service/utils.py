from typing import Dict, List, Union
from dataclasses import dataclass
import msgpack
import zlib

class AugmatrixDataType:
    def to_dict(self):
        # Get all attributes of the object
        obj_attributes = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]

        # Create a dictionary to store the attributes and their values
        obj_dict = {}
        for attr in obj_attributes:
            obj_dict[attr] = getattr(self, attr)

        return obj_dict

def encode(obj):
    # Serialize the object to MessagePack
    msgpack_data = msgpack.packb(obj, default=lambda x: x.__dict__, use_bin_type=True)
    
    # Compress the MessagePack data using zlib
    compressed_data = zlib.compress(msgpack_data)

    return compressed_data

def decode_to_object(data, cls):
    # Decompress the compressed data using zlib
    decompressed_data = zlib.decompress(data)

    # Deserialize the decompressed data from MessagePack
    if isinstance(decompressed_data, bytes):
        decompressed_data = msgpack.unpackb(decompressed_data, raw=False)
    
    # Get the dictionary keys from the class's attributes
    cls_attributes = cls.__annotations__.keys()

    # Create a dictionary containing only the relevant keys and values
    obj_data = {key: decompressed_data[key] for key in cls_attributes if key in decompressed_data}

    # Instantiate the class with the extracted data
    return cls(**obj_data)

def decode(data):
    decompressed_data = zlib.decompress(data)
    if isinstance(decompressed_data, bytes):
        decompressed_data = msgpack.unpackb(decompressed_data, raw=False)
    return decompressed_data
