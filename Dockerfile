FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.9 python3.9-dev python3-pip libgl1-mesa-glx libglib2.0-0 && apt-get clean
WORKDIR /workspace
COPY requirements.txt .
RUN python3.9 -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python3.9 -m pip install --no-deps .
ENTRYPOINT ["python3.9", "-m", "chaw_op.commands.train"]
