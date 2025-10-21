#!/bin/bash

# Variables required for logging as a user with the same id as the user running this script
export LOCAL_USER_ID=`id -u $USER`
export LOCAL_GROUP_ID=`id -g $USER`
export LOCAL_GROUP_NAME=`id -gn $USER`
DOCKER_USER_ARGS="--env LOCAL_USER_ID --env LOCAL_GROUP_ID --env LOCAL_GROUP_NAME"

# Variables for forwarding ssh agent into docker container
SSH_AUTH_ARGS=""
if [ ! -z $SSH_AUTH_SOCK ]; then
    # Monta la directory del socket SSH e imposta la variabile d'ambiente
    DOCKER_SSH_AUTH_ARGS="-v $(dirname $SSH_AUTH_SOCK):$(dirname $SSH_AUTH_SOCK) -e SSH_AUTH_SOCK=$SSH_AUTH_SOCK"
fi

# Settings required for X11 forwarding inside the docker (per GUI)
DOCKER_X11_ARGS="--env DISPLAY --env QT_X11_NO_MITSHM=1 --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw"

# Base Docker command for running the container (senza NVIDIA)
DOCKER_COMMAND="docker run"

# Determine network arguments
DOCKER_NETWORK_ARGS="--net host"
if [[ "$@" == *"--net "* ]]; then
    # Se l'utente fornisce il proprio argomento --net, non aggiungere --net host
    DOCKER_NETWORK_ARGS=""
fi

# Arguments from your specific command that are NOT related to NVIDIA or the network argument above
# Includes: --privileged, --rm, -v /dev/snd, -v /var/run/docker.sock
DOCKER_MANDATORY_ARGS="--runtime=runc --privileged --rm -v /dev/snd:/dev/snd -v /var/run/docker.sock:/var/run/docker.sock"

xhost +

# Stampa il comando che verrà eseguito
echo "
$DOCKER_COMMAND \
$DOCKER_USER_ARGS \
$DOCKER_X11_ARGS \
$DOCKER_SSH_AUTH_ARGS \
$DOCKER_NETWORK_ARGS \
$DOCKER_MANDATORY_ARGS \
"$@""

# Esegue il comando
$DOCKER_COMMAND \
$DOCKER_USER_ARGS \
$DOCKER_X11_ARGS \
$DOCKER_SSH_AUTH_ARGS \
$DOCKER_NETWORK_ARGS \
$DOCKER_MANDATORY_ARGS \
"$@"