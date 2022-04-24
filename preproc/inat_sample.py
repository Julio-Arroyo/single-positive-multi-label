import os, random
import matplotlib.pyplot as plt


random.seed(2022)


NUM_IMAGES = 1000
iNat_path = "/media/julioarroyo/aspen/single-positive-multi-label/data/iNat21/train_mini"
if __name__ == '__main__':
    list_images = []
    for i in range(NUM_IMAGES):
        folder_path = random.choice(os.listdir(iNat_path))
        list_images.append(iNat_path + '/' + folder_path + '/' + random.choice(os.listdir(iNat_path + '/' + folder_path)))

    for img_path in list_images:
        im = plt.imread(img_path)
        plt.imshow(im)
        plt.show()
