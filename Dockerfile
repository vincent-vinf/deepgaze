FROM tensorflow/tensorflow:1.15.5
WORKDIR /workspace

# RUN python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple --upgrade pip && \
#    pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip3 install --user -i https://mirrors.aliyun.com/pypi/simple opencv-python==3.4.5.20 Flask

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update -qqy && \
    apt-get install -qqy --no-install-recommends \
        libsm6 \
        libxrender1 \
        libxext6 && \
    apt-get clean -qqy && \
    rm -rf /var/cache/apt

COPY . .

RUN python3 setup.py install
