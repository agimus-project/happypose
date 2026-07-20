#!/bin/bash

happypose_docker() {
  echo "starting happypose docker"
  xhost +local:docker;
  docker run -it --rm -d \
    --name="happypose_dev" \
    --runtime=nvidia \
    --workdir $HOME/happypose \
    -v /etc/localtime:/etc/localtime:ro \
    -v /dev/input:/dev/input \
    --shm-size=12G \
    --net=host \
    --add-host happypose_dev:127.0.0.1 \
    --hostname=happypose_dev \
    --privileged=true \
    --env=QT_X11_NO_MITSHM=1 \
    --device=/dev/dri:/dev/dri \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /etc/localtime:/etc/localtime:ro \
    -v "./:$HOME/happypose" \
    -e HAPPYPOSE_DATA_DIR="$HOME/happypose/dataset" \
    happypose:latest

    happypose_docker_attach;
}

happypose_docker_attach() {
  docker exec -it -e "COLUMNS=$COLUMNS" -e "LINES=$LINES" happypose_dev /bin/bash -c "source /happypose/.venv/bin/activate && bash"
}
