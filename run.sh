SHARED_FOLDER=$(dirname $(readlink -f $0) | rev | cut -d'/' -f1- | rev)

docker run -it --runtime=nvidia --rm --network host --ipc=host \
    --privileged \
    --name test \
    --user=$( id -u $USER ):$( id -g $USER ) \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
    -v ${SHARED_FOLDER}:/shared-folder \
    -e DISPLAY=$DISPLAY \
    --env QT_X11_NO_MITSHM=1 \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --mount src="$(pwd)",target=/home/mamba/stable-baselines3,type=bind stablebaselines/stable-baselines3 bash