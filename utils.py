import glob
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import setting
step_per_epoch=0
validation_steps=0
def read_png_label(path,color=None):
    img = tf.io.read_file(path)
    if color:
        img = tf.image.decode_png(img,channels=1)
    else:
        img = tf.image.decode_png(img, channels=3)
    # #显示图片
    # print(img)
    # img1=tf.squeeze(img,axis=-1)
    # img1=img1.numpy()
    # plt.imshow(img1)
    # plt.show()
    return img
def show_pic(path,islabel=None):
    img = tf.io.read_file(path)
    if islabel:
        img = tf.image.decode_png(img, channels=1)
        img = tf.squeeze(img, axis=-1)
    else:
        img = tf.image.decode_png(img, channels=3)
    img1 = img.numpy()
    plt.imshow(img1)
    plt.show()
#数据增强函数
# 数据增强
def crop_img(img,mask,istrain=True):
    if istrain:
        concat_img = tf.concat([img,mask],axis=-1) # 把image和label合并在一起  axis = -1，表示最后一个维度  （这样使label和image随机裁剪的位置一致）
        concat_img = tf.image.resize(concat_img,(256,256), # 修改大小为256，256
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)#使用最近邻插值调整images为size
        crop_img = tf.image.random_crop(concat_img,[224,224,4]) # 随机裁剪
       # print("train",crop_img)
        return crop_img[ :, :, :3],crop_img[ :, :, 3:] # 高维切片(第一，第二维度全要，第三个维度的前3是image，最后一个维度就是label)
    else:
        concat_img = tf.concat([img, mask], axis=-1)  # 把image和label合并在一起  axis = -1，表示最后一个维度  （这样使label和image随机裁剪的位置一致）
        concat_img = tf.image.resize(concat_img, (224, 224),  # 修改大小为224,224
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # 使用最近邻插值调整images为size
       # print("train", concat_img)
        return concat_img[:, :, :3], concat_img[:, :, 3:]
def load_pic(path,path_label):
    img=read_png_label(path)
    path_label=read_png_label(path_label,color=True)
    img,mask=crop_img(img,path_label)
    if tf.random.uniform(()) > 0.5:  # 从均匀分布中返回随机值 如果大于0.5就执行下面的随机翻转
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    img = tf.cast(img, tf.float32) / 127.5 - 1
    mask = tf.cast(mask, tf.int32)
    return img,mask
def load_pic_test(path,path_label):
    img=read_png_label(path)
    path_label=read_png_label(path_label,color=True)
    img,mask=crop_img(img,path_label,istrain=None)
    img = tf.cast(img, tf.float32) / 127.5 - 1
    mask = tf.cast(mask, tf.int32)
    return img,mask
def get_train():
    img = glob.glob(setting.train_path)
    img_label=glob.glob(setting.train_label_path)
    global step_per_epoch
    step_per_epoch=len(img_label)/setting.BATCH_SIZE
    #read_png_label(img_label[0])
    # dict_img=[]
    # # for i,j in zip(img,img_label):
    # #     dict_img.append((i,j))
    # # print(dict_img)
    #也不知道这句话啥意思先写上估计有用
    auto=tf.data.experimental.AUTOTUNE # 根据cpu使用情况自动规划线程读取图片

    dataset_train = tf.data.Dataset.from_tensor_slices((img,img_label))

    dataset_train=dataset_train.map(
        load_pic,
        num_parallel_calls=auto
    )
    dataset_train=dataset_train.repeat(setting.repeat).shuffle(setting.BUFFER_SIZE).batch(setting.BATCH_SIZE).prefetch(auto)
    return dataset_train
def get_test():
    img = glob.glob(setting.test_path)
    img_label = glob.glob(setting.train_label_path)
    img=img[:70]
    img_label=img_label[:70]
    global validation_steps
    validation_steps=len(img)/setting.BATCH_SIZE
    #read_png_label(img_label[0])
    # dict_img=[]
    # # for i,j in zip(img,img_label):
    # #     dict_img.append((i,j))
    # # print(dict_img)
    # 也不知道这句话啥意思先写上估计有用
    auto = tf.data.experimental.AUTOTUNE  # 根据cpu使用情况自动规划线程读取图片

    dataset_train = tf.data.Dataset.from_tensor_slices((img, img_label))

    dataset_train=dataset_train.map(
        load_pic_test,
        num_parallel_calls=auto
    )
    dataset_train=dataset_train.shuffle(setting.BUFFER_SIZE).batch(setting.BATCH_SIZE).prefetch(auto)
    return dataset_train
#get_train()
#read_png_label("C:/Users/mzy/Desktop/cityscapes/gtFine/test/berlin/berlin_000000_000019_gtFine_labelIds.png",color=True)