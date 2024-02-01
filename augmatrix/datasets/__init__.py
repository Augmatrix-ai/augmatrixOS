from .dataset_loader import AugmatrixLoader
from typing import Dict, List, Union, Any
from dataclasses import dataclass
from augmatrix.block_service.data_context import AugmatrixDataType

DATATYPE_TO_PY_DATATYPE = {
    "Object": Any,
    "Image": bytes,
    "JSON": str,
    "Text": str,
    "Audio": bytes,
    "Video": bytes,
    "PDF": bytes
}

def load_dataset(dataset_names):
    return AugmatrixLoader(dataset_names).load_datasets()

def variable_def_to_dataclass(variables_struct, class_name):
    attributes = {}
    for var_id, var_info in variables_struct.items():
        attributes[var_id] = DATATYPE_TO_PY_DATATYPE[var_info['type_text']]

    kclass = type(class_name, (AugmatrixDataType,), attributes)
    kclass.__annotations__ = attributes
    
    return dataclass(kclass)

    