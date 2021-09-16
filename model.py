import tensorflow as tf
import  setting
class U_net(tf.keras.Model):
    def __init__(self):
        super(U_net, self).__init__()
        self.conv2d1=tf.keras.layers.Conv2D(filters=64,kernel_size=3,strides=1,padding="same")
        self.relu1=tf.keras.layers.ReLU()
        self.conv2d2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        self.relu2 = tf.keras.layers.ReLU()
        self.conv2d3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        self.relu3 = tf.keras.layers.ReLU()
        self.max2d1=tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding="valid",strides=None)
        self.conv2d4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")
        self.relu4 = tf.keras.layers.ReLU()
        self.conv2d5 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")
        self.relu5 = tf.keras.layers.ReLU()
        self.max2d2=tf.keras.layers.MaxPooling2D(pool_size=2,strides=None,padding="valid")
        self.conv2d6 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")
        self.relu6 = tf.keras.layers.ReLU()
        self.conv2d7 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")
        self.relu7 = tf.keras.layers.ReLU()
        self.max2d3=tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding="valid",strides=None)
        self.conv2d8 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same")
        self.relu8 = tf.keras.layers.ReLU()
        self.conv2d9 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same")
        self.relu9 = tf.keras.layers.ReLU()
        self.max2d4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid", strides=None)
        self.conv2d10 = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, strides=1, padding="same")
        self.relu10 = tf.keras.layers.ReLU()
        self.conv2d11 = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, strides=1, padding="same")
        self.relu11 = tf.keras.layers.ReLU()
        #self.max2d5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid", strides=None)
        self.conv2d1tran=tf.keras.layers.Conv2DTranspose(filters=512,strides=2 ,kernel_size=1,padding="valid")
        self.conv2d12 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same")
        self.relu12 = tf.keras.layers.ReLU()
        self.conv2d13 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding="same")
        self.relu13 = tf.keras.layers.ReLU()
        self.conv2d1tran1 = tf.keras.layers.Conv2DTranspose(filters=256, strides=2, kernel_size=1, padding="valid")
        self.conv2d14 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")
        self.relu14 = tf.keras.layers.ReLU()
        self.conv2d15 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")
        self.relu15 = tf.keras.layers.ReLU()
        self.conv2d1tran2 = tf.keras.layers.Conv2DTranspose(filters=128, strides=2, kernel_size=1, padding="valid")
        self.conv2d16 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")
        self.relu16 = tf.keras.layers.ReLU()
        self.conv2d17 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same")
        self.relu17 = tf.keras.layers.ReLU()
        self.conv2d1tran3 = tf.keras.layers.Conv2DTranspose(filters=64, strides=2, kernel_size=1, padding="valid")
        self.conv2d18 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        self.relu18 = tf.keras.layers.ReLU()
        self.conv2d19 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        self.relu19 = tf.keras.layers.ReLU()
        self.conv2d20=tf.keras.layers.Conv2D(filters=34,kernel_size=1,strides=1,padding="same",activation="softmax")

        #以下全是bn层
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.bn6 = tf.keras.layers.BatchNormalization()
        self.bn7 = tf.keras.layers.BatchNormalization()
        self.bn8 = tf.keras.layers.BatchNormalization()
        self.bn9 = tf.keras.layers.BatchNormalization()
        self.bn10 = tf.keras.layers.BatchNormalization()
        self.bn11= tf.keras.layers.BatchNormalization()
        self.bn12 = tf.keras.layers.BatchNormalization()
        self.bn13 = tf.keras.layers.BatchNormalization()
        self.bn14 = tf.keras.layers.BatchNormalization()
        self.bn15 = tf.keras.layers.BatchNormalization()
        self.bn16 = tf.keras.layers.BatchNormalization()
        self.bn17 = tf.keras.layers.BatchNormalization()
        self.bn18 = tf.keras.layers.BatchNormalization()
        self.bn19 = tf.keras.layers.BatchNormalization()
        self.bn20 = tf.keras.layers.BatchNormalization()
        self.bn21 = tf.keras.layers.BatchNormalization()
        self.bn22 = tf.keras.layers.BatchNormalization()
        self.bn23 = tf.keras.layers.BatchNormalization()
    def call(self, inputs, training=None, mask=None):
        #第一层
        x=self.conv2d1(inputs)
        x=self.relu1(x)
        x = self.bn1(x)
        x=self.conv2d2(x)
        x=self.relu2(x)
        x = self.bn2(x)
        x=self.conv2d3(x)
        x=self.relu3(x)
        x = self.bn3(x)
        x1=x
        # print(x)

        #向下取样
        x=self.max2d1(x)
        # print(x)

        #第二层
        x=self.conv2d4(x)
        x=self.relu4(x)
        x = self.bn4(x)
        x=self.conv2d5(x)
        x=self.relu5(x)
        x = self.bn5(x)
        x2=x
        #print(x)

        #向下取样
        x=self.max2d2(x)
        #print(x)

        #第三层
        x=self.conv2d6(x)
        x=self.relu6(x)
        x = self.bn6(x)
        x=self.conv2d7(x)
        x=self.relu7(x)
        x = self.bn7(x)
        x3=x
        #print(x)

        #向下取样
        x=self.max2d3(x)
        #print(x)

        #第四层
        x=self.conv2d8(x)
        x=self.relu8(x)
        x = self.bn8(x)
        x=self.conv2d9(x)
        x=self.relu9(x)
        x = self.bn9(x)
        x4=x
        #print("x4",x)

        #向下取样
        x=self.max2d4(x)
        #print(x)

        #第五层
        x=self.conv2d10(x)
        x=self.relu10(x)
        x = self.bn10(x)
        x=self.conv2d11(x)
        x=self.relu11(x)
        x = self.bn11(x)
        x5=x
        #print("x5",x)

        #反卷积第一层
        x=self.conv2d1tran(x)
        x = self.bn12(x)
        #print("x1",x)

        #向上取样第一层
        x=tf.concat([x,x4],axis=-1)
        x=self.conv2d12(x)
        x=self.relu12(x)
        x = self.bn13(x)
        x=self.conv2d13(x)
        x=self.relu13(x)
        x = self.bn14(x)
        #print(x)

        #反卷积第二层
        x=self.conv2d1tran1(x)
        x = self.bn15(x)
        #print("x2",x)

        #向上取样第二层
        x=tf.concat([x,x3],axis=-1)
        x=self.conv2d14(x)
        x=self.relu14(x)
        x = self.bn16(x)
        x=self.conv2d15(x)
        x=self.relu15(x)
        x = self.bn17(x)
        #print(x)

        #反卷积第三层
        x=self.conv2d1tran2(x)
        x = self.bn18(x)
        #print("x3",x)

        #向上取样第三层
        x=tf.concat([x,x2],axis=-1)
        x=self.conv2d16(x)
        x=self.relu16(x)
        x = self.bn19(x)
        x=self.conv2d17(x)
        x=self.relu17(x)
        x = self.bn20(x)

        #反卷积低四层
        x=self.conv2d1tran3(x)
        x = self.bn21(x)
        #print("X4",x)

        #向上取样第四层
        x=tf.concat([x,x1],axis=-1)
        x=self.conv2d18(x)
        x=self.relu18(x)
        x = self.bn22(x)
        x=self.conv2d19(x)
        x=self.relu19(x)
        x = self.bn23(x)
        #print(x)

        #输出层
        x=self.conv2d20(x)

       # print(x)

        return x

