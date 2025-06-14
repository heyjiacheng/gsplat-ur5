import typing
import time
from realsense import MultiRealsense
from embodied_gaussians.scene_builders.domain import MaskedPosedImageAndDepth
from embodied_gaussians.utils.utils import ExtrinsicsData


def get_datapoints_from_live_cameras(
    extrinsics: dict[str, ExtrinsicsData],
    segmentor: typing.Literal["sam", "quick"] = "quick",
) -> list[MaskedPosedImageAndDepth]:
    if segmentor == "quick":
        from quick_segmentor import QuickSegmentor

        segmentor = QuickSegmentor()  # type: ignore
    elif segmentor == "sam":
        from sam_segmentor import SamSegmentor

        segmentor = SamSegmentor()  # type: ignore
    else:
        raise ValueError(f"Unknown segmentor {segmentor}")

    datapoints = []
    serial_numbers = list(extrinsics.keys())
    print("serial_numbers", serial_numbers)
    with MultiRealsense(serial_numbers=serial_numbers, enable_depth=True) as realsenses:
        # realsenses.set_exposure(177, 70)
        # realsenses.set_white_balance(4600)
        # time.sleep(1)  # Give some time for color to adjust
        all_camera_data = realsenses.get()
        all_intrinsics = realsenses.get_intrinsics()
        all_depth_scale = realsenses.get_depth_scale()

        for serial, camera_data in all_camera_data.items():
            if serial not in extrinsics:
                print(f"Camera {serial} is not known. Skipping.")
                continue
            K = all_intrinsics[serial]
            depth_scale = all_depth_scale[serial]
            color = camera_data["color"]
            mask = segmentor.segment_with_gui(color)
            datapoint = MaskedPosedImageAndDepth(
                K=K,
                X_WC=extrinsics[serial].X_WC,
                image=camera_data["color"],
                format="bgr",
                depth=camera_data["depth"],
                depth_scale=depth_scale,
                mask=mask,
            )
            datapoints.append(datapoint)
    return datapoints


# Create type variables for the argument and return types of the function
A = typing.TypeVar("A", bound=typing.Callable[..., typing.Any])
R = typing.TypeVar("R")


def static(**kwargs: typing.Any) -> typing.Callable[[A], A]:
    """A decorator that adds static variables to a function
    :param kwargs: list of static variables to add
    :return: decorated function

    Example:
        @static(x=0, y=0)
        def my_function():
            # static vars are stored as attributes of "my_function"
            # we use static as a more readable synonym.
            static = my_function

            static.x += 1
            static.y += 2
            print(f"{static.f.x}, {static.f.x}")

        invoking f three times would print 1, 2 then 2, 4, then 3, 6

    Static variables are similar to global variables, with the same shortcomings!
    Use them only in small scripts, not in production code!
    """

    def decorator(func: A) -> A:
        for key, value in kwargs.items():
            setattr(func, key, value)
        return func

    return decorator
