import numpy as np


if __name__ == '__main__':
    threshold = 0.9
    ds = 'coco'
    teacher_preds = np.load(f'../data/{ds}/teacher_preds.npy')

    # assert values
    for i in range(teacher_preds.shape[0]):
        for j in range(teacher_preds.shape[1]):
            if teacher_preds[i, j] < 0 or teacher_preds[i, j] > 1:
                assert False, 'Found invalid prediction'
    
    teacher_preds[teacher_preds < threshold] = 0
    teacher_preds[teacher_preds >= threshold] = 1

    np.save(f'../data/{ds}/teacher_labels.npy', teacher_preds)
    