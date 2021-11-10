import io
import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template,make_response
from werkzeug.utils import secure_filename
from datetime import timedelta
from inference import prepare,predict_one_img
import time

app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])

model = None
config = None
img_data = None
checkpoint_path="./static/model/checkpoint.pth.tar"

def load_model():
    global model
    global config
    model, config = prepare(checkpoint_path=checkpoint_path, topk=1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    global img_data
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径

        f.save(upload_path)
        img_data = Image.open(upload_path).convert('RGB')
        img_data.save(os.path.join(basepath, 'static/images', 'test.jpg'))
        print("load img")
        # 使用Opencv转换一下图片格式和名称
        # img = cv2.imread(upload_path)
        # cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)
        result=predict()
        return render_template('upload_ok.html', val1=time.time(),predictres=result)

    return render_template('upload.html')

def predict():
    global img_data
    data = predict_one_img(model=model,device="cpu",config=config,img_data=img_data,k=1)

    # return the data dictionary as a JSON response
    return data

if __name__ == "__main__":
    print(("* Loading pytorch model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run(debug=True)