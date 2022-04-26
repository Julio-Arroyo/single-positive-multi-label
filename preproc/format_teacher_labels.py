import numpy as np


if __name__ == '__main__':
    threshold = 0.65
    print(f'Threshold: {threshold}')
    ds = 'pascal'
    teacher_preds = np.load(f'../data/{ds}/teacher_preds.npy')
    print(f'Dimensions: {teacher_preds.shape}')

    print(f'Average number of labels per image: {np.sum(teacher_preds) / teacher_preds.shape[0]}')

    # assert values
    for i in range(teacher_preds.shape[0]):
        for j in range(teacher_preds.shape[1]):
            if teacher_preds[i, j] < 0 or teacher_preds[i, j] > 1:
                assert False, 'Found invalid prediction'
    
    teacher_preds[teacher_preds < threshold] = 0
    teacher_preds[teacher_preds >= threshold] = 1

    # get average number of labels per image
    print(f'Average number of labels per image: {np.sum(teacher_preds) / teacher_preds.shape[0]}')

    np.save(f'../data/{ds}/teacher_labels_t{threshold*100}.npy', teacher_preds)
    