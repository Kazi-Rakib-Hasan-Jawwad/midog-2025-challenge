"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

from pathlib import Path
import json
import torch
from glob import glob
import SimpleITK
import numpy

from utils.inference_utils import DetectionAlgorithm



INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        ("histopathology-region-of-interest-cropout",): interf0_handler,
    }[interface_key]

    # Call the handler
    return handler()


def interf0_handler():
    # Read the input
    sitk_image = load_image_file(location=INPUT_PATH / "images/histopathology-roi-cropout")

    # Read addtional resources, these can be included in the resources folder 
    # Here, we load some config files 
    resource_dir = Path("/opt/app/resources")
    model_config = resource_dir / "model_config.yaml"
    inference_config = resource_dir / "inference_config.yaml"
    patch_config = resource_dir / "patch_config.yaml"

    # Load the algorithm 
    detection = DetectionAlgorithm(
        model_config=model_config,
        patch_config=patch_config,
        inference_config=inference_config
    )

    # Run inference steps 
    output = detection.predict(sitk_image)

    # Save your output
    write_json_file(
        location=OUTPUT_PATH / "mitotic-figures.json", content=output
    )

    return 0


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def load_image_file(*, location):
    # Use SimpleITK to read a file
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    return SimpleITK.ReadImage(input_files[0])




if __name__ == "__main__":
    raise SystemExit(run())
