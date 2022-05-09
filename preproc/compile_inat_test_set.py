import random
import shutil
import json
import os


RANDOM_SEED = 2022
OMITTED_SUPERCATEGORIES = set(['Animalia', 'Fungi', 'Mollusks', 'Plants'])


def make_dict(D):
    supercat_2_cat_2_imgs = {}
    cat_2_supercat = {}

    for category in D['categories']:
        curr_supercat = category['supercategory']
        curr_cat = category['id']

        cat_2_supercat[curr_cat] = curr_supercat

        if curr_supercat in OMITTED_SUPERCATEGORIES:
            continue

        if curr_supercat not in supercat_2_cat_2_imgs:
            supercat_2_cat_2_imgs[curr_supercat] = {}

        # No reason to see same category twice while iterating over all categories
        assert curr_cat not in supercat_2_cat_2_imgs[curr_supercat]

        supercat_2_cat_2_imgs[curr_supercat][curr_cat] = []
    
    return supercat_2_cat_2_imgs, cat_2_supercat


def classify_images_by_cat(D, cat_2_supercat, supercat_2_cat_2_imgs):
    for annotation in D['annotations']:
        curr_cat = annotation['category_id']
        curr_supercat = cat_2_supercat[curr_cat]

        if curr_supercat in OMITTED_SUPERCATEGORIES:
            continue

        assert curr_supercat in supercat_2_cat_2_imgs
        assert curr_cat in supercat_2_cat_2_imgs[curr_supercat], f'Current supercategory: {curr_supercat}'

        supercat_2_cat_2_imgs[curr_supercat][curr_cat].append(annotation['image_id'])


def get_img_id_2_img_name(D):
    img_id_2_img_name = {}
    for image in D['images']:
        img_id_2_img_name[image['id']] = image['file_name']
    return img_id_2_img_name


def get_test_image_ids(supercat_2_cat_2_imgs):
    test_images = []

    for supercat in supercat_2_cat_2_imgs:
        categories = random.sample(supercat_2_cat_2_imgs[supercat].keys(), k=NUM_CATEGORIES)
        for category in categories:
            assert NUM_IMGS_PER_CATEGORY <= len(supercat_2_cat_2_imgs[supercat][category])
            new_images = random.sample(supercat_2_cat_2_imgs[supercat][category], k = NUM_IMGS_PER_CATEGORY)
            test_images += new_images
    return test_images


if __name__ == '__main__':
    NUM_CATEGORIES = 15
    NUM_IMGS_PER_CATEGORY = 10

    random.seed(RANDOM_SEED)

    f = open('/media/julioarroyo/aspen/iNat21/annotations/val.json')
    D = json.load(f)

    (supercat_2_cat_2_imgs, cat_2_supercat) = make_dict(D)
    classify_images_by_cat(D, cat_2_supercat, supercat_2_cat_2_imgs)

    img_id_2_img_name = get_img_id_2_img_name(D)

    test_image_ids = get_test_image_ids(supercat_2_cat_2_imgs)
    test_image_filenames = [img_id_2_img_name[curr_img_id] for curr_img_id in test_image_ids]

    dest_path = '/media/julioarroyo/aspen/relabel_iNat_images/'
    src_path = '/media/julioarroyo/aspen/iNat21/'

    for img_file in test_image_filenames:
        # get path without filename
        dir_path = os.path.dirname(dest_path + img_file)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)  # create directory

        shutil.copyfile(src_path + img_file, dest_path + img_file)
