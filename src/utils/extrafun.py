from pathlib import Path
import os
import yaml
from ruamel.yaml import YAML

def read_config(configname):
    ruamelFile = YAML()
    path = Path(configname)
    if os.path.exists(path):
       with open(path, "r") as f:
            cfg = ruamelFile.load(f)
    return cfg

