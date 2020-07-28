import os
import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from keras.layers import Input
from keras.models import Sequential, load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import datetime
import re
from timeit import default_timer as timer
import cv2

upload_folder = "uploads"
output_folder = "static\outputs"
ALLOWED_EXTENSIONS = set(['mp4', 'mov'])
output_filename = "result_" + ''.join(re.split('[- :.]',str(datetime.datetime.now()),6)[0:5]) + ".mp4"


app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_video(yolo, video_path, output_path=""):
    
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    frame_count=vid.get(cv2.CAP_PROP_FRAME_COUNT)
    c=0
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        if isOutput:
            out.write(result)
        c += 1


        progress=100*c/frame_count
        print("{:.1f}%,{}".format(progress,return_value))
        
        if c==frame_count:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

graph = tf.get_default_graph()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global graph
    with graph.as_default():
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('ファイルがありません')
                return redirect("index.html")
            file = request.files['file']
            if file.filename == '':
                flash('ファイルがありません')
                return redirect("index.html")
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(upload_folder, filename))
                filepath = os.path.join(upload_folder, filename)
                outputpath = os.path.join(output_folder,output_filename)

                detect_video(YOLO(), filepath, outputpath)

                return render_template("index.html",textmassage="識別結果",video_path=outputpath)

        return render_template("index.html",textmassage="サンプル動画",video_path='static\outputs\demo.mp4')


if __name__ == "__main__":
    # app.run()
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)