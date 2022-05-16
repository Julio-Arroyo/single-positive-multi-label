#!/bin/bash

python3 teacher_preds.py
cd preproc
python3 format_teacher_labels.py --threshold 0.05
python3 format_teacher_labels.py --threshold 0.15
python3 format_teacher_labels.py --threshold 0.25
python3 format_teacher_labels.py --threshold 0.35
python3 format_teacher_labels.py --threshold 0.45
python3 format_teacher_labels.py --threshold 0.55
python3 format_teacher_labels.py --threshold 0.65
python3 format_teacher_labels.py --threshold 0.75
python3 format_teacher_labels.py --threshold 0.85
python3 format_teacher_labels.py --threshold 0.95
cd ..
python3 train.py 'coco' 'bce_ls' 'uniform' 1 > logs/pseudo_multi_label/coco/student_t75.txt