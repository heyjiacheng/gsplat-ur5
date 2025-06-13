import time
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import tyro

from ground_finder import GroundFinder, GroundFinderSettings
from embodied_gaussians.utils.utils import read_extrinsics
from utils import get_datapoints_from_live_cameras


@dataclass
class Params:
    save_path: tyro.conf.PositionalRequiredArgs[Path]
    extrinsics: Path
    max_depth: float = 2.0
    builder: GroundFinderSettings = field(
        default_factory=lambda: GroundFinderSettings()
    )
    visualize: bool = False


def main(params: Params):
    assert params.save_path.suffix == ".json"
    if params.save_path.exists():
        overwrite = input(f"File {params.save_path} already exists. Overwrite? (y/n)")
        if overwrite != "y":
            print("Aborting.")
            return
    else:
        params.save_path.parent.mkdir(parents=True, exist_ok=True)

    all_extrinsics = read_extrinsics(params.extrinsics)
    datapoints = get_datapoints_from_live_cameras(all_extrinsics)

    result = GroundFinder.find_ground(
        params.builder, datapoints, visualize=params.visualize
    )
    res = {"plane": result.plane.tolist()}

    with open(f"{params.save_path}", "w") as f:
        json.dump(res, f, indent=2)

    np.save(f"{params.save_path.with_suffix('.npy')}", result.points)
    print(f"Saved ground to {params.save_path}")


if __name__ == "__main__":
    params = tyro.cli(Params)
    assert params.save_path.suffix == ".json"
    main(params)

