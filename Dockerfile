FROM python:3.9

# install MPI
RUN apt-get update
RUN apt-get install -y libopenmpi-dev openmpi-bin openmpi-doc

RUN pip install --upgrade pip
# Avoid cache miss after change in mount directory and no changes to reqirements.txt
COPY requirements.txt /var
RUN pip install -r /var/requirements.txt
RUN pip install patchelf
# Install mujoco
WORKDIR /root/.mujoco
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
RUN tar -xf mujoco210-linux-x86_64.tar.gz
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
ENV MUJOCO_PY_MUJOCO_PATH /root/.mujoco/mujoco210
RUN apt-get update
RUN apt-get install libosmesa6-dev -y
# Trigger mujoco_py cython compilation so it is cached on image build instead of image run
RUN python -c "import mujoco_py"

ADD . /mnt
WORKDIR /mnt

ENTRYPOINT ["python", "./test.py"]
