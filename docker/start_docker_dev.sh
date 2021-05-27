docker run --gpus all -it -p 8888:8888  \
  --shm-size=8gb --env="DISPLAY"\
  -v="/home/madr/pycharm-community-2021.1.1:/pycharm" \
  -v="$PWD/pers_files:/pers_files" \
  -v="$PWD/TI_pyscripts:/pyscripts" \
 --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --name=detectron2 detectron2:v0
