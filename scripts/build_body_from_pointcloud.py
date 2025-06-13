from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

import tyro

from embodied_gaussians.scene_builders.domain import (
    save_posed_images,
    load_posed_images,
    GaussianLearningRates,
)
from embodied_gaussians.scene_builders.pointcloud_body_builder import (
    PointCloudBodyBuilder,
    PointCloudBodyBuilderSettings,
)
from embodied_gaussians.utils.utils import read_extrinsics
from utils import get_datapoints_from_live_cameras


@dataclass
class Params:
    save_path: tyro.conf.PositionalRequiredArgs[Path]
    extrinsics: Path
    points: Path
    """
    Path to a numpy file containing a Nx3 array of points
    """
    visualize: bool = False
    offline: bool = False
    save_posed_images: bool = True
    builder: PointCloudBodyBuilderSettings = field(
        default_factory=lambda: PointCloudBodyBuilderSettings(
            training_learning_rates=GaussianLearningRates(
                colors=0.1,
                means=0.0,
                scales=0.004,
                quats=0.0,
            ),
            min_scale=(0.0, 0.0, 0.0),
            max_scale=(0.03, 0.03, 0.03)
        )
    )


def main(params: Params):
    assert params.save_path.suffix == ".json"
    if params.save_path.exists():
        overwrite = input(f"File {params.save_path} already exists. Overwrite? (y/n)")
        if overwrite != "y":
            print("Aborting.")
            return
    else:
        params.save_path.parent.mkdir(parents=True, exist_ok=True)

    extrinsics = read_extrinsics(params.extrinsics)
    points = np.load(params.points)
    if not params.offline:
        datapoints = get_datapoints_from_live_cameras(extrinsics)
        if params.save_posed_images:
            save_posed_images(
                f"temp/posed_images/{params.save_path.stem}.npz", datapoints
            )
    else:
        try:
            datapoints = load_posed_images(
                f"temp/posed_images/{params.save_path.stem}.npz"
            )
        except FileNotFoundError:
            print("Posed images not found. Run with offline=False to generate them")
            return

    result = PointCloudBodyBuilder.build(
        name=params.save_path.stem,
        points=points,
        settings=params.builder,
        datapoints=datapoints,
        visualize=params.visualize,
    )

    if result is None:
        print("Body builder failed")
        return

    with open(params.save_path, "w") as f:
        f.write(result.model_dump_json(indent=4))


if __name__ == "__main__":
    params = tyro.cli(Params)
    main(params)
