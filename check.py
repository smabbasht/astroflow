import tensorflow as tf

# print(tf.sysconfig.get_build_info())
# physical_devices = tf.config.list_physical_devices('GPU')
# print(physical_devices)
# physical_devices = tf.config.experimental.list_physical_devices('/device=GPU:0')
device = tf.device("/gpu:0" if tf.config.experimental.list_physical_devices("GPU") else "/cpu:0") 
print(device)
print(tf.config.experimental.list_physical_devices("GPU"))


# if physical_devices:
#   tf.config.experimental.set_memory_growth(physical_devices[0], True)
# # check if the code us running on gpu 
# # device = tensorflow.test.gpu_device_name()
# #
# # print(tensorflow.test.is_built_with_cuda())
#
# from tensorflow.python.client import device_lib
#
# def get_available_devices():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos]
#
# print(get_available_devices())
