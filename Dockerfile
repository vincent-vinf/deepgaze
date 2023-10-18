FROM tensorflow/tensorflow:1.15.5
WORKDIR /workspace

# RUN python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple --upgrade pip && \
#    pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip3 install --user -i https://mirrors.aliyun.com/pypi/simple opencv-python==3.4.5.20

COPY . .
