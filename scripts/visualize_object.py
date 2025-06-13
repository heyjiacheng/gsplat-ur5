from pathlib import Path
from dataclasses import dataclass
import json
import tyro

from embodied_gaussians.scene_builders.domain import Body
from embodied_gaussians.scene_builders.simple_visualizer import visualize

@dataclass
class Params:
    path: tyro.conf.PositionalRequiredArgs[str]

def main(params: Params):
    path = Path(params.path)
    if not path.exists():
        print(f"File {path} does not exist")
        return

    with open(path, "r") as f:
        model_data = json.load(f)
    
    body = Body.model_validate(model_data)

    visualize(body)




if __name__ == "__main__":
    params =  tyro.cli(Params)
    main(params)