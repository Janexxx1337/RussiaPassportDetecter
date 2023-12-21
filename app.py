from flask import Flask, request, render_template
import cv2
import numpy as np
import io
import main as pasp_read  # Импорт модуля main

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'Файл не загружен', 400
        file = request.files['file']
        if file.filename == '':
            return 'Файл не выбран', 400

        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        in_memory_file.seek(0)  # Перемещение указателя в начало файла
        data = np.frombuffer(in_memory_file.read(), dtype=np.uint8)  # Использование np.frombuffer
        color_image_flag = 1
        image = cv2.imdecode(data, color_image_flag)

        photo = pasp_read.resize(image)  # Использование функции resize из модуля main
        enhanced_photo = pasp_read.apply_clahe(photo)  # Предполагая, что функция apply_clahe также в main
        result = pasp_read.pasp_read(enhanced_photo)  # Использование функции pasp_read из модуля main

        return render_template('result.html', result=result)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
