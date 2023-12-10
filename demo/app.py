# from flask import Flask, render_template, request

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     files = request.files.getlist('files')
#     for file in files:
#         file.save(file.filename)
#     return '文件上传成功！'

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, render_template, request

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     files = request.files.getlist('files')
#     filenames = []
#     for file in files:
#         filename = file.filename
#         file.save(filename)
#         filenames.append(filename)
#     return render_template('uploaded.html', filenames=filenames)

# if __name__ == '__main__':
#     app.run(debug=True)

# import os
# from flask import Flask, render_template, request, jsonify

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/process_images', methods=['POST'])
# def process_images():
#     images = []
    
#     if 'images' in request.files:
#         uploaded_files = request.files.getlist('images')
        
#         for file in uploaded_files:
#             # 处理图片并保存到服务器
#             # 这里可以根据具体需求进行图片处理操作
#             # processed_image_path = process_image(file)
            
#             # 将处理后的图片路径添加到返回结果中
#             images.append(file)
    
#     return jsonify({'images': images})

# # def process_image(file):
# #     # 在这里进行图片处理操作，这里只是简单地保存到服务器
# #     filename = secure_filename(file.filename)
# #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
# #     return os.path.join(app.config['UPLOAD_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run()



from flask import Flask, render_template, request, jsonify
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

def process_image(image_data):
    # 在这里添加你的图片处理逻辑
    # 这个例子中只是简单地将图片转换为灰度图
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    img = img.convert('L')  # 转换为灰度图
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'images[]' not in request.files:
        return jsonify({'error': 'No images found'})
    images = request.files.getlist('images[]')
    processed_images = []
    for img in images:
        img_data = base64.b64encode(img.read()).decode("utf-8")
        processed_img = process_image(img_data)
        processed_images.append(processed_img)
    return jsonify({'processed_images': processed_images})

if __name__ == '__main__':
    app.run(debug=True)
