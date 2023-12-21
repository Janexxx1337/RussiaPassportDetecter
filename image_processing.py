import cv2


def preprocess_image(image_path):
    # Открытие изображения
    image = cv2.imread(image_path)

    # Применение фильтра для улучшения контрастности
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Применение порогового фильтра
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresholded
