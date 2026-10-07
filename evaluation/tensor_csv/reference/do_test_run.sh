#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="example-algorithm-track-1-object-detection-preliminary-evaluation-phase"

DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/do_build.sh"

cleanup() {
    echo "=+= Cleaning permissions ..."
    # Ensure permissions are set correctly on the output
    # This allows the host user (e.g. you) to access and handle these files
    docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "chmod -R -f o+rwX /output/* || true"

    # Ensure volume is removed
    docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null
}

# This allows for the Docker user to read
chmod -R -f o+rX "$INPUT_DIR" "${SCRIPT_DIR}/model"


if [ -d "${OUTPUT_DIR}/interf0" ]; then
  # This allows for the Docker user to write
  chmod -f o+rwX "${OUTPUT_DIR}/interf0"

  echo "=+= Cleaning up any earlier output"
  # Use the container itself to circumvent ownership problems
  docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "${OUTPUT_DIR}/interf0":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwX "${OUTPUT_DIR}/interf0"
fi


docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null

trap cleanup EXIT

run_docker_forward_pass() {
    local interface_dir="$1"

    echo "=+= Doing a forward pass on ${interface_dir}"

    ## Note the extra arguments that are passed here:
    # '--network none'
    #    entails there is no internet connection
    # 'gpus all'
    #    enables access to any GPUs present
    # '--volume <NAME>:/tmp'
    #   is added because on Grand Challenge this directory cannot be used to store permanent files
    # '-volume ../model:/opt/ml/model/":ro'
    #   is added to provide access to the (optional) tarball-upload locally
    # '--memory'
    #   was added to increase docker memory
    # '--shm-size'
    #   was added to increase shared memory when using multiple workers 
    docker run --rm \
        --platform=linux/amd64 \
        --network none \
        --memory=16g \
        --shm-size=8g \
        --gpus all \
        --volume "${INPUT_DIR}/${interface_dir}":/input:ro \
        --volume "${OUTPUT_DIR}/${interface_dir}":/output \
        --volume "$DOCKER_NOOP_VOLUME":/tmp \
        --volume "${SCRIPT_DIR}/model":/opt/ml/model:ro \
        "$DOCKER_IMAGE_TAG"

  echo "=+= Wrote results to ${OUTPUT_DIR}/${interface_dir}"
}


run_docker_forward_pass "interf0"


# Path to the output and expected files
OUTPUT_JSON="${OUTPUT_DIR}/interf0/mitotic-figures.json"
EXPECTED_JSON="${OUTPUT_DIR}/expected_output.json"

echo "=+= Comparing output with expected output..."

python3 -c "
import json
import sys

try:
    with open('$OUTPUT_JSON', 'r') as f:
        output_data = json.load(f)
    
    with open('$EXPECTED_JSON', 'r') as f:
        expected_data = json.load(f)
    
    if output_data == expected_data:
        print('=+= Output matches expected output!')
        sys.exit(0)
    else:
        print('ERROR: Output does not match expected output.')
        sys.exit(1)
        
except FileNotFoundError as e:
    print(f'ERROR: File not found: {e}')
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f'ERROR: Invalid JSON: {e}')
    sys.exit(1)
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
"


echo "=+= Save this image for uploading via ./do_save.sh"
