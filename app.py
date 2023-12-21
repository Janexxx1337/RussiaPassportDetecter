from flask import Flask, request, render_template
import numpy as np
import cv2
import io
import main
import logging

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)

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
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            return 'Ошибка декодирования изображения', 400

        try:
            result = main.catching(image)
            if result is not None:
                # Передаем данные в HTML-шаблон
                return render_template('result.html', result=result)
            else:
                return 'Ошибка обработки изображения', 500
        except Exception as e:
            app.logger.error(f'Ошибка обработки: {e}')
            return 'Ошибка обработки изображения', 500

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
