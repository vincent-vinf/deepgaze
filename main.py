#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import cv2
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator

from flask import Flask, request, jsonify
import os
from datetime import datetime
import random
import string

sess = tf.Session()  # Launch the graph in a session.
my_head_pose_estimator = CnnHeadPoseEstimator(sess)
my_head_pose_estimator.load_yaw_variables(os.path.realpath("etc/tensorflow/head_pose/yaw/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_roll_variables(os.path.realpath("etc/tensorflow/head_pose/roll/cnn_cccdd_30k.tf"))
my_head_pose_estimator.load_pitch_variables(os.path.realpath("etc/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"))


def head_pose(in_path):
    print("Processing image ..... " + in_path)
    # file_name = "1.jpg"
    image = cv2.imread(in_path)
    roll_degree = my_head_pose_estimator.return_roll(image, radians=False)  # Evaluate the roll angle using a CNN
    pitch_degree = my_head_pose_estimator.return_pitch(image, radians=False)  # Evaluate the pitch angle using a CNN
    yaw_degree = my_head_pose_estimator.return_yaw(image, radians=False)  # Evaluate the yaw angle using a CNN
    print("Estimated [roll, pitch, yaw] (degrees) ..... [" + str(roll_degree[0, 0, 0]) + "," + str(
        pitch_degree[0, 0, 0]) + "," + str(yaw_degree[0, 0, 0]) + "]")

    return roll_degree[0, 0, 0], pitch_degree[0, 0, 0], yaw_degree[0, 0, 0]


app = Flask(__name__)

# 设置保存上传文件的目录
UPLOAD_DIR = 'uploads'


@app.route('/head-pose', methods=['POST'])
def upload_file():
    # 检查是否有文件被上传
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # 检查文件名是否存在
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # 生成唯一的文件名
    filename = generate_filename()

    # 保存上传的图片到本地
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)
    roll, pitch, yaw = head_pose(save_path)

    return jsonify({'filename': filename,
                    "roll": float(roll),
                    "pitch": float(pitch),
                    "yaw": float(yaw)
                    }), 200


def generate_filename():
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    random_letters = ''.join(random.choices(string.ascii_lowercase, k=4))
    filename = f"{current_time}_{random_letters}.jpg"
    return filename


if __name__ == '__main__':
    # 创建保存文件的目录（如果不存在）
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # 启动 Flask 服务器
    app.run(host="0.0.0.0", port=6443)
