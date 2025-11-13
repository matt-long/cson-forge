import os
from pathlib import Path
from datetime import datetime

USER = os.environ["USER"]
here = os.path.dirname(os.path.realpath(__file__))

# local 
logs = Path(f"{here}/logs")
blueprints = Path(f"{here}/blueprints")
config_defs = Path(f"{here}/model-configs")

# model configuration
model_config = config_defs / "roms-marbl-cson-default"

# data paths
source_data = Path(f"/Users/{USER}/data/source_data")
source_data.mkdir(exist_ok=True)

input_data = Path(f"/Users/{USER}/data/input_data")
input_data.mkdir(exist_ok=True)

scratch = Path(f"/Users/{USER}/scratch")
scratch.mkdir(exist_ok=True)
