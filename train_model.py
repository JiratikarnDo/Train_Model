# นำเข้าไลบรารีที่จำเป็น
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# เตรียมข้อมูลสำหรับการฝึก
train_datagen = ImageDataGenerator(
    rescale=1./255,          # แปลงค่าพิกเซลให้มีค่าอยู่ในช่วง 0-1
    rotation_range=20,       # หมุนภาพสุ่มในช่วง 20 องศา
    width_shift_range=0.2,   # การเลื่อนแนวนอนสุ่ม
    height_shift_range=0.2,  # การเลื่อนแนวตั้งสุ่ม
    zoom_range=0.2,          # ซูมภาพสุ่ม
    horizontal_flip=True,    # สะท้อนภาพในแนวนอน
    fill_mode='nearest'
)

# โหลดข้อมูลภาพจากโฟลเดอร์ในเครื่อง
train_generator = train_datagen.flow_from_directory(
    'C:/Users/wmmyo/Downloads/DataSet',  # แก้ไข path ให้เป็น path ของข้อมูลที่ต้องการใช้ฝึก
    target_size=(224, 224),  # ขนาดภาพที่จะใช้ในการฝึก
    batch_size=128,  # ปรับ batch_size เป็น 64
    class_mode='categorical'
)

# สร้างโมเดล MobileNetV2 หรือโหลดโมเดลที่ฝึกไปแล้ว
model_path = 'C:/Users/wmmyo/Downloads/model_finlove.h5'
if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
    model = load_model(model_path)  # ลองโหลดโมเดลที่บันทึกไว้
    print("Loaded existing model.")
else:
    # สร้างโมเดลใหม่หากไม่มีโมเดลที่บันทึกไว้
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    print("Created new model.")

# Compile โมเดล
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# ฝึกโมเดล
model.fit(train_generator, epochs=6)

# บันทึกโมเดลหลังจากฝึกเสร็จ
model.save('C:/Users/wmmyo/Downloads/model_finlove.h5')
