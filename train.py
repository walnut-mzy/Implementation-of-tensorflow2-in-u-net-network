import tensorflow as tf
from model import U_net
from  utils import get_train,get_test
import setting
import datetime
import utils
#tf.config.experimental_run_functions_eagerly(True)
begin=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print(begin)
model = U_net()
model.build(input_shape=setting.input_shape)
model.summary()
# #
# # tf.keras.metrics.MeanIoU(num_classes=34) # 根据独热编码进行计算
# # 我们是顺序编码 需要更改类
# class MeanIou(tf.keras.metrics.MeanIoU): # 继承这个类
#     def __call__(self,y_true,y_pred,sample_weight=None):
#         print(y_pred)
#         y_pred = tf.argmax(y_pred,axis=-1)
#         return super().__call__(y_true,y_pred,sample_weight=sample_weight)
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"]
             )

dataset_val=get_test()
dataset=get_train()
model.fit(dataset,
          epochs=60,
        steps_per_epoch=utils.step_per_epoch,
        validation_steps=utils.validation_steps,
          validation_data=dataset_val
          )
if setting.model_save_path:
    model.save(str(begin)+".h5")
end=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
print("开始时间为:{}\n结束时间为:{}\n总用时:\n{}".format(begin,end,end-begin))