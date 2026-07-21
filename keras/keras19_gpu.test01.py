
import tensorflow as tf
print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices("GPU")
print(gpus)

if(gpus):
    print("GPU 돈다~")      #[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
else:
    print("GPU 없다~")      