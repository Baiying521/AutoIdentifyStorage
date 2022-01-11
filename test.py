import os

path = "./test/"
for file in os.listdir(path):
    print(file)
    print(path + file)
