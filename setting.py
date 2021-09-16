import os

BUFFER_SIZE=100

BATCH_SIZE=32

repeat=10

input_shape=(None,224,224,3)


train_path="C:/Users/mzy/Desktop/cityscapes/leftImg8bit/test/berlin/*.png"

train_label_path="C:/Users/mzy/Desktop/cityscapes/gtFine/test/berlin/*_gtFine_labelIds.png"

#注：笔者为了方便两个设为一致的
test_path="C:/Users/mzy/Desktop/cityscapes/leftImg8bit/test/berlin/*.png"

test_label_path="C:/Users/mzy/Desktop/cityscapes/gtFine/test/berlin/*_gtFine_labelIds.png"

#如果为none则不保存模型，如果加上路径则保存模型
model_save_path="./model"

#这部分为方便程序运行勿动
list_path=[train_path[:4],train_label_path[:4],test_path[:4],test_label_path[:4],model_save_path]

for path in list_path:
    if path:
        if not os.path.exists(path):
            os.mkdir(path)