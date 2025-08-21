import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 모델 로드
model_path = 'converted_savedmodel/model.savedmodel'
model_layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')

# 클래스 레이블
with open('converted_savedmodel/labels.txt', 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f.readlines()]


def classify_image(image_path):
    # 이미지 전처리
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 예측
    result = model_layer(img_array)
    if isinstance(result, dict):
        predictions = list(result.values())[0].numpy()
    else:
        predictions = result.numpy()

    # 가장 높은 확률의 클래스
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    print(f"이미지: {image_path}")
    print(f"예측 결과: {class_names[predicted_class]}")
    print(f"신뢰도: {confidence:.2%}")

    # 상위 3개
    print(f"\n상위 3개 예측:")
    top3_indices = np.argsort(predictions[0])[-3:][::-1]
    for i, idx in enumerate(top3_indices, 1):
        print(f"  {i}. {class_names[idx]}: {predictions[0][idx]:.2%}")

    return predicted_class, confidence


# 현재 폴더의 이미지 파일
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
for file in os.listdir('.'):
    if any(file.lower().endswith(ext) for ext in image_extensions):
        print(f"{file}")

# 테스트
print("\n" + "=" * 50)
classify_image('PetImages00/Dog/510.jpg')
