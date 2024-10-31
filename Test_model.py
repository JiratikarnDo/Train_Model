# นำเข้าไลบรารีที่จำเป็น
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model_path = 'C:/Users/wmmyo/Downloads/model_finlove.h5'
model = load_model(model_path)
print("Loaded trained model.")

# เตรียมข้อมูลสำหรับการทดสอบ
# กำหนด path ของโฟลเดอร์ที่มีรูปภาพทดสอบ
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    'C:/Users/wmmyo/Downloads/060151ab-9604-428a-ab11-f04e401880d0.jpg',  # แก้ไข path ให้เป็น path ของข้อมูลที่ต้องการใช้ทดสอบ
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# ประเมินผลโมเดลกับชุดข้อมูลทดสอบ
loss, accuracy = model.evaluate(validation_generator)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# ทำนายผลจากภาพเดี่ยว
# กำหนด path ของภาพที่ต้องการทดสอบ
img_path = 'C:/Users/wmmyo/Downloads/060151ab-9604-428a-ab11-f04e401880d0.jpg'  # แก้ไข path ให้เป็น path ของภาพที่ต้องการทดสอบ
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

# ทำนายผล
predictions = model.predict(x)
class_indices = validation_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}
predicted_class = class_labels[np.argmax(predictions)]

print(f'Predicted Class: {predicted_class}')
