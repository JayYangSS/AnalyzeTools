
# -*- coding: utf-8 -*-

"""This code used for show the vanish bbox and save the result"""
#import cv2
import numpy as np
import os

label_name_list=['vehicle', 'human', 'Cyclist', 'dont care']
# def cv_imread(image_path):
#     cv_img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
#     return cv_img

# def cv_imwrite(write_path, img):
#     cv2.imencode('.png', img,)[1].tofile(write_path)


# def show_FP_FN_bbox(img,FP_bboxes,FN_bboxes):
#     for FP_bbox in FP_bboxes:
#         cv2.rectangle(img, (int(FP_bbox[0]), int(FP_bbox[1])),(int(FP_bbox[2]), int(FP_bbox[3])), (0, 0, 255), 2)
#         cv2.putText(img, FP_bbox[5] + ':' + '%.2f' % FP_bbox[4], (int(FP_bbox[0]), int(FP_bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#
#     for FN_bbox in FN_bboxes:
#         cv2.rectangle(img, (int(FN_bbox[0]), int(FN_bbox[1])), (int(FN_bbox[2]), int(FN_bbox[3])), (0, 255, 0), 2)
#         cv2.putText(img, FN_bbox[5] + ':' + '%.2f' % FN_bbox[4], (int(FN_bbox[0]), int(FN_bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#     return img

class boxWithIou:
    def __init__(self):
        self.box=[]
        self.iou=0 #GT与detection的IOU




"""box:[x1,y1,x2,y2]"""#box2默认传入GT box
def IOU(box1, box2):
    if (box1[2] < box2[0]) or (box2[2] < box1[0]):
        return 0,0,0,0
    if (box1[3] < box2[1]) or (box2[3] < box1[1]):
        return 0,0,0,0
    intersect_w = min(box1[2], box2[2]) - max(box1[0], box2[0]) + 1
    intersect_h = min(box1[3], box2[3]) - max(box1[1], box2[1]) + 1
    union_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1) + (box2[2] - box2[0] + 1) * (    box2[3] - box2[1] + 1) - intersect_w * intersect_h
    iou = float(intersect_h * intersect_w) / float(union_area)

    #calculate the localization error
    box1_w=box1[2]-box1[0]+1
    box1_h=box1[3]-box1[1]+1
    box2_w = box2[2] - box2[0] + 1
    box2_h = box2[3] - box2[1] + 1

    eps=0.00000001
    width_error = abs(box2_w-box1_w)/(box2_w+eps)
    height_error = abs(box2_h - box1_h) / (box2_h + eps)
    bottom_error = abs(box2[3]-box1[3])/(box2_h + eps)
    return iou,width_error,height_error,bottom_error



def getDetBoxes(video_result_path):
    video_result_file = open(video_result_path, 'r')
    result_all_dict = {}
    result_lines = video_result_file.readlines()
    video_result_file.close()
    line_id = 0
    video_result_dict = {}
    while line_id < len(result_lines):
        line = result_lines[line_id].strip()
        split_strs = line.split(' ')
        video_frame_id = split_strs[1]
        box_num = int(split_strs[0])
        video_result_dict[video_frame_id]=[]
        #if line_id==0:
        #    line_id = line_id + 1
        for i in range(box_num):
            line_id = line_id + 1
            coord_line = result_lines[line_id]
            coord_split_strs = coord_line.split(' ')
            coord = []

            coord.append(int(coord_split_strs[0]))  #x1
            coord.append(int(coord_split_strs[1])) # y1
            coord.append(int(coord_split_strs[2])) # x2
            coord.append(int(coord_split_strs[3]))  # y2
            video_result_dict[video_frame_id].append(coord)
        line_id = line_id + 1
    return video_result_dict



def getGTBoxes(video_GT_path,class_id):
    video_gt_file=open(video_GT_path,'r')
    gt_label_lines = video_gt_file.readlines()
    video_gt_file.close()

    video_gt_dict={}
    for line_id,line in  enumerate(gt_label_lines):
        strs = line.strip().split(' ')
        obj_num = int(strs[0])
        video_gt_dict[str(4*line_id)]=[]

        for obj_id in range(obj_num):
            class_label=int(strs[5*obj_id+5])
            if class_id!=class_label:
                continue
            x1 = int(strs[5*obj_id+1])
            y1 = int(strs[5*obj_id+2])
            x2 = int(strs[5*obj_id+3])
            y2 = int(strs[5*obj_id+4])
            video_gt_dict[str(4 * line_id)].append([x1,y1,x2,y2])
    return video_gt_dict


"""bbox:[x1,y1,x2,y2,label_name,prob]"""
def getTP_FP_FN(gt_boxes,det_boxes,iou_thresh=0.4):
    TP,FP,FN=0,0,0
    FP_bboxes=[]#误检框
    FN_bboxes=[]#漏检框
    gt_with_iou_bboxes=[]#带iou信息的真值框
    det_assign=np.zeros(len(det_boxes))
    gt_assign = np.zeros(len(gt_boxes))

    best_match_detId=0
    acc_width_error=0
    acc_height_error=0
    acc_bottom_error=0
    for gt_id,gt_box in enumerate(gt_boxes):

        gt_with_iou=boxWithIou()#带有IOU信息的gt目标存储
        gt_with_iou.box=gt_box

        iou_max = 0.0
        best_match_width_error=0
        best_match_height_error=0
        best_match_bottom_error=0
        for det_id,det_box in enumerate(det_boxes):
            iou, width_error, height_error, bottom_error =IOU(det_box,gt_box)

            if iou >= iou_max:
                iou_max=iou
                best_match_detId=det_id
                best_match_width_error = width_error
                best_match_height_error = height_error
                best_match_bottom_error = bottom_error

        if iou_max>iou_thresh:
            TP+=1
            acc_width_error += best_match_width_error
            acc_height_error += best_match_height_error
            acc_bottom_error += best_match_bottom_error
            det_assign[best_match_detId]=1
        else:
            FN+=1
            FN_bboxes.append(gt_box)

        gt_with_iou.iou=iou_max
        gt_with_iou_bboxes.append(gt_with_iou)

    #calculate FP
    for id in range(len(det_assign)):
        if det_assign[id]==0:
            FP+=1
            FP_bboxes.append(det_boxes[id])

    return (TP,FP,FN,FP_bboxes,FN_bboxes,gt_with_iou_bboxes,(acc_width_error,acc_height_error,acc_bottom_error))



def video_localization_error(video_gt_labels,video_det_results,iou_thresh=0.4):
    assert len(video_det_results)==len(video_gt_labels),'GT img num do not match detect img num!!!'
    frame_num=len(video_gt_labels)
    video_acc_width_error=0
    video_acc_height_error = 0
    video_acc_bottom_error = 0
    for id in range(frame_num):
        frame_id=str(4*id)
        gt_boxes=video_gt_labels[frame_id]
        det_boxes=video_det_results[frame_id]
        TP, FP, FN, FP_bboxes, FN_bboxes, gt_with_iou_bboxes,localization_error=getTP_FP_FN(gt_boxes, det_boxes, iou_thresh=iou_thresh)
        acc_width_error, acc_height_error, acc_bottom_error=localization_error
        video_acc_width_error +=acc_width_error
        video_acc_height_error +=acc_height_error
        video_acc_bottom_error +=acc_bottom_error
    return video_acc_width_error,video_acc_height_error,video_acc_bottom_error,TP,gt_with_iou_bboxes





video_result_path=u'test/detection.txt'
video_GT_path=u'test/GT.txt'

video_det_results=getDetBoxes(video_result_path)
video_gt_labels=getGTBoxes(video_GT_path,class_id=1)
video_acc_width_error,video_acc_height_error,video_acc_bottom_error,TP,gt_with_iou_bboxes = video_localization_error(video_gt_labels,video_det_results)

pass


