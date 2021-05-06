docker run --gpus all -it -p 8888:8888  \
  --shm-size=8gb --env="DISPLAY"\
  -v="/home/madsraad/pycharm-community:/pycharm" \
  -v="/home/madsraad/ml/detectron2/pers_files:/pers_files" \
 --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --name=detectron2 detectron2:v0
