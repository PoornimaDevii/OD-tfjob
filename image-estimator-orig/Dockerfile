FROM tensorflow/tensorflow:2.3.0-gpu
#FROM tensorflow/tensorflow:2.1.0-gpu-py3
#FROM tensorflow/tensorflow:2.2.1-gpu
RUN apt update && apt install curl unzip wget libsm6 libxext6 libxrender-dev libglib2.0-0 -y
RUN python3 -m pip install --upgrade pip
RUN pip install opencv-python==4.1.1.26 lxml tqdm
COPY yolov3-tf2/ /opt/yolov3-tf2
ADD main.sh /opt/
RUN chmod +x /opt/main.sh
ENTRYPOINT ["/opt/main.sh"]

