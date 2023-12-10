import time

from flask import Flask, render_template, request, jsonify
from PIL import Image
from io import BytesIO
import base64
import os

app = Flask(__name__)
def process_image(image_data):
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    img = img.convert('L') 
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@app.route('/')
def index():
    return render_template('index.html')


upload_path = 'D:/wangfuming/Desktop/Visual/file/upload/'
processed_path = 'D:/wangfuming/Desktop/Visual/file/processed/'

@app.route('/upload', methods=['POST'])
def upload():
    print(0)
    if 'images[]' not in request.files:
        return jsonify({'error': 'No images found'})
    print(1)
    images = request.files.getlist('images[]')
    upload_file = []
    # # 图片保存
    # for img in images:
    #     file_name = img.filename
    #     upload_file.append(os.path.join(upload_path, file_name))
    #     img.save(os.path.join(upload_path, file_name))
    time_list = []
    psnr = []
    processed_images = []
    ans = []
    for img in images:
        img_data = base64.b64encode(img.read()).decode("utf-8")
        processed_img = process_image(img_data)
        # img_data = base64.b64decode(processed_img)
        # image = Image.open(BytesIO(img_data))
        # image.save(os.path.join(processed_path, img.filename))
        # processed_images.append(processed_path+img.filename)
        processed_images.append(processed_img)
        time_list.append(time.time())
        psnr.append(time.time()-1)
        ans.append({
            'processed_image':processed_img,
            'low_image':img_data,
            'time':0.75,
            'psnr':26
        })

    return jsonify({'processed_images': ans})

    # print(processed_images)
    # print(2)
    # return jsonify({'processed_images': processed_images,
    #                 'low_images': processed_images,
    #                 'time_list': time_list,
    #                 'psnr': psnr})

if __name__ == '__main__':
    app.run(debug=True)
