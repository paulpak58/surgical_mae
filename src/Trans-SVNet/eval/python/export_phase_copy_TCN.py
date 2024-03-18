import pickle
import shutil
import os
import argparse

with open('./cholec80.pkl', 'rb') as f:
    test_paths_labels = pickle.load(f)

parser = argparse.ArgumentParser(description='lstm testing')
parser.add_argument('-s', '--seq', default=10, type=int, help='sequence length, default 10')
parser.add_argument('-n', '--name', type=str, help='name of pred')

args = parser.parse_args()
# sequence_length = args.seq
# pred_name = args.name
model_name = args.name

sequence_length = 1
if model_name == 'Trans_SV':
    pred_name = 'Trans_SV_new_weights.pkl'
elif model_name == 'TeCNO':
    pred_name = 'TeCNO_test_length30_new_weights.pkl'
else:
    raise Exception('model name error')

with open(pred_name, 'rb') as f:
    ori_preds = pickle.load(f)
'''
# Validation + Test results in overall lower accuracy
with open(pred_val_name, 'rb') as f:
    ori_val_preds = pickle.load(f)
'''

num_video = 32
num_labels = 0
for i in range(48,80):
    print(f'Video {i} len {len(test_paths_labels[i])}')
    num_labels += len(test_paths_labels[i])

num_preds = len(ori_preds)

print('num of labels  : {:6d}'.format(num_labels))
print("num ori preds  : {:6d}".format(num_preds))
print("labels example : ", test_paths_labels[0][0][1])
# print("labels example : ", test_paths_labels[0])
print("preds example  : ", ori_preds[0])
print("sequence length : ", sequence_length)

# raise Exception('stop')

########################
'''
phase_dict_key = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
            'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']
preds_all = []
label_all = []
count = 0
for i in range(48,80):
    if pred_name.split('_')[0] == 'TeCNO':
        filename = '/home/ppak/surgical_adventure/src/Trans-SVNet/eval/tcn_result/phase/video' + str(1 + i) + '-phase.txt'
        gt_filename = '/home/ppak/surgical_adventure/src/Trans-SVNet/eval/tcn_result/gt-phase/video' + str(1 + i) + '-phase.txt'
    elif pred_name.split('_')[0] == 'Trans':
        filename = '/home/ppak/surgical_adventure/src/Trans-SVNet/eval/trans_sv_result/phase/video' + str(1 + i) + '-phase.txt'
        gt_filename = '/home/ppak/surgical_adventure/src/Trans-SVNet/eval/trans_sv_result/gt-phase/video' + str(1 + i) + '-phase.txt'
    else:
        raise Exception("Model error")
    f = open(filename, 'w')
    f2 = open(gt_filename, 'w')
    preds_each = []
    for j in range(count, count + len(test_paths_labels[i])):
        preds_each.append(ori_preds[j])
        preds_all.append(ori_preds[j])
    for k in range(len(preds_each)):
        f.write(str(25 * k))
        f.write('\t')
        f.write(str(int(preds_each[k])))
        f.write('\n')
        
        f2.write(str(25 * k))
        f2.write('\t')
        f2.write(str(int(test_paths_labels[i][k][1])))
        label_all.append(test_paths_labels[i][k][1])
        f2.write('\n')
    print(f"Video {i} complete")

    f.close()
    f2.close()
    count += len(test_paths_labels[i])
test_corrects = 0
print('num of labels       : {:6d}'.format(len(label_all)))
print('rsult of all preds  : {:6d}'.format(len(preds_all)))
print('global num labels   : {:6d}'.format(num_labels))
for i in range(num_labels):
    if label_all[i] == preds_all[i]:
        test_corrects += 1
print('right number preds  : {:6d}'.format(test_corrects))
print('test accuracy       : {:.4f}'.format(test_corrects / num_labels))
'''

if num_labels == (num_preds + (sequence_length - 1) * num_video):

    phase_dict_key = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
                  'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']
    preds_all = []
    label_all = []
    count = 0
    # for i in range(40,80):
    for i in range(48,80):
        if model_name == 'TeCNO':
            filename = '/home/ppak/surgical_adventure/src/Trans-SVNet/eval/tcn_result/phase/video' + str(1 + i) + '-phase.txt'
            gt_filename = '/home/ppak/surgical_adventure/src/Trans-SVNet/eval/tcn_result/gt-phase/video' + str(1 + i) + '-phase.txt'
        elif model_name == 'Trans_SV':
            filename = '/home/ppak/surgical_adventure/src/Trans-SVNet/eval/trans_sv_result/phase/video' + str(1 + i) + '-phase.txt'
            gt_filename = '/home/ppak/surgical_adventure/src/Trans-SVNet/eval/trans_sv_result/gt-phase/video' + str(1 + i) + '-phase.txt'
        else:
            raise Exception("Model error")
        f = open(filename, 'w')
        #f.write('Frame Phase')
        #f.write('\n')

        f2 = open(gt_filename, 'w')
        #f2.write('Frame Phase')
        #f2.write('\n')

        preds_each = []
        for j in range(count, count + len(test_paths_labels[i]) - (sequence_length - 1)):
            if j == count:
                # TODO 单个视频的初始几个帧的predict设置
                for k in range(sequence_length - 1):
                    preds_each.append(0)
                    preds_all.append(0)
            preds_each.append(ori_preds[j])
            preds_all.append(ori_preds[j])
        for k in range(len(preds_each)):
            f.write(str(25 * k))
            f.write('\t')
            f.write(str(int(preds_each[k])))
            f.write('\n')
            
            f2.write(str(25 * k))
            f2.write('\t')
            f2.write(str(int(test_paths_labels[i][k][1])))
            label_all.append(test_paths_labels[i][k][1])
            f2.write('\n')
        print(f"Video {i} complete")

        f.close()
        f2.close()
        count += len(test_paths_labels[i]) - (sequence_length - 1)
    test_corrects = 0

    print('num of labels       : {:6d}'.format(len(label_all)))
    print('rsult of all preds  : {:6d}'.format(len(preds_all)))

    for i in range(num_labels):
        if int(label_all[i]) == int(preds_all[i]):
            test_corrects += 1

    print('right number preds  : {:6d}'.format(test_corrects))
    print('test accuracy       : {:.4f}'.format(test_corrects / num_labels))
else:
    print('number error, please check')