# 导入random模块和os模块
import random
import os

# 定义一个列表，存放生成的10000个文件的名称
files = ["merge-" + str(i+1) + ".wav" for i in range(10000)]

# 打乱文件的顺序
random.shuffle(files)

# 将前7000个文件移动到train文件夹
for i in range(7000):
    os.rename(os.path.join("samples/Merge", files[i]), os.path.join("samples/train", files[i]))

# 将后3000个文件移动到valid文件夹
for i in range(7000, 10000):
    os.rename(os.path.join("samples/Merge", files[i]), os.path.join("samples/valid", files[i]))