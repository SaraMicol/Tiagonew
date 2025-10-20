#!/bin/bash

# Variables required for logging as a user with the same id as the user running this script
export LOCAL_USER_ID=`id -u $USER`
export LOCAL_GROUP_ID=`id -g $USER`
export LOCAL_GROUP_NAME=`id -gn $USER`
DOCKER_USER_ARGS="--env LOCAL_USER_ID --env LOCAL_GROUP_ID --env LOCAL_GROUP_NAME"

# Variables for forwarding ssh agent into docker container
SSH_AUTH_ARGS=""
if [ ! -z $SSH_AUTH_SOCK ]; then
    DOCKER_SSH_AUTH_ARGS="-v $(dirname $SSH_AUTH_SOCK):$(dirname $SSH_AUTH_SOCK) -e SSH_AUTH_SOCK=$SSH_AUTH_SOCK"
fi

DOCKER_NETWORK_ARGS="--net host"
if [[ "$@" == *"--net "* ]]; then
    DOCKER_NETWORK_ARGS=""
fi

xhost +

echo "
docker run \
  --env LOCAL_USER_ID \
  --env LOCAL_GROUP_ID \
  --env LOCAL_GROUP_NAME \
  --env DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /run/user/1000/keyring:/run/user/1000/keyring \
  -e SSH_AUTH_SOCK=/run/user/1000/keyring/ssh \
  --net host \
  --privileged \
  --rm \
  -v /dev/snd:/dev/snd \
  -v /var/run/docker.sock:/var/run/docker.sock \
  "$@""

docker run \
  --env LOCAL_USER_ID \
  --env LOCAL_GROUP_ID \
  --env LOCAL_GROUP_NAME \
  --env DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /run/user/1000/keyring:/run/user/1000/keyring \
  -e SSH_AUTH_SOCK=/run/user/1000/keyring/ssh \
  --net host \
  --privileged \
  --rm \
  -v /dev/snd:/dev/snd \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v ./manipulation_challenge_2025/:/tiago_public_ws/src/manipulation_challenge \
  -it irim