FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

COPY VERSION /

ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

RUN apt-get update

RUN pip install --upgrade pip --no-cache-dir

RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs

COPY src ${EXEC_DIR}/
WORKDIR ${EXEC_DIR}

RUN apt-get -y install libgomp1 gfortran libglib2.0.0 libgles2-mesa-dev

RUN  pip install  -r ${EXEC_DIR}/requirements.txt --no-cache-dir

ENTRYPOINT ["python3", "main.py"]