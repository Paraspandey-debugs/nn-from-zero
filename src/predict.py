def predict_external_image(image_path, dense1, activation1, dense2, activation2, dense3):
    import cv2
    import numpy as np

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))

    if np.mean(image) >= 127.5:
        image = 255 - image

    image_data = (image.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

    dense1.forward(image_data)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)

    return np.argmax(dense3.output, axis=1)[0]