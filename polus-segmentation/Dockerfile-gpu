
FROM   labshare/polus-bfio-util:1.3.4-tensorflow

COPY VERSION /

ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"



RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir ${DATA_DIR}/outputs


COPY src ${EXEC_DIR}/
WORKDIR ${EXEC_DIR}

RUN pip3 uninstall -y  tensorflow-gpu

RUN apt-get update

#RUN  apt-get install build-essential



RUN apt-get -y install libgomp1 gfortran libglib2.0.0 libgles2-mesa-dev

RUN pip3 install -r ${EXEC_DIR}/requirements-gpu.txt --no-cache-dir

#ENTRYPOINT ["python3", "main.py"]


