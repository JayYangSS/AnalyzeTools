#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: jayn
# date: 2020/5/19

import logging
import random
import collections
import cv2
import numpy as np
from core.data import dataset, data_reader
import os


__all__ = ['roi_yolo_dataset']


def roi_yolo_dataset(worker_num, **kwargs):
    data_set = RoiYoloDataset(**kwargs)
    fp16 = kwargs.get('fp16', False)
    batch_size = kwargs['batch_size']
    train_loader = data_reader.DataReader(data_set, batch_size, thread_num=worker_num, fp16=fp16)
    return train_loader


class SampleList(object):

    def __init__(self):
        self.cursor = 0
        self.sample_list = []


class Roi(object):
    def __init__(self):
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0
        self.label = 0.0


class ImageDatabase(object):

    def __init__(self):
        self.name = ''
        self.cache = None
        self.gt_boxes = []


class RoiYoloDataset(dataset.Dataset):

    def __init__(self, root_folder, source, roidb_file, train=False, **kwargs):

        self.root_folder = root_folder
        self.source = source
        self.roidb_file = roidb_file
        self.lines_id = 0

        # 是否训练
        self.train = train

        # 训练模式输出标签
        self.output_labels = True # self.train

        # 是否缓存图片数据
        self.cache_images = kwargs.pop('cache_images', False)

        # 是否做样本均衡
        self.balanced_sampling = kwargs.pop('balanced_sampling', False)

        # 图片扩展名
        self.image_extension = kwargs.pop('image_extension', '')

        # shuffle
        self.shuffle = kwargs.pop('shuffle', False)

        # mirror
        self.mirror = kwargs.pop('mirror', False)

        # is color
        self.is_color = kwargs.pop('is_color', True)

        # batch_size
        self.batch_size = kwargs.pop('batch_size', 1)

        self.class_list = []
        self.imdb = []
        if self.balanced_sampling:
            self.class_list.append(SampleList())

        # scales
        self.scales = kwargs.pop('scales', [])

        #crop mode
        self.crop_mode=kwargs.pop('crop_mode',False)

        # resize scales
        self.resize_scales = kwargs.pop('resize_scales', [])

        # do_hsv_jitter
        self.do_hsv_jitter = kwargs.pop('do_hsv_jitter', False)

        # read labels
        self._read_labels()

        # shuffle
        if self.shuffle:
            logging.info('Shuffling image list')
            if self.balanced_sampling:
                # shuffle sample list of each class
                for i in range(len(self.class_list)):
                    random.shuffle(self.class_list[i].sample_list)

                # shuffle the class list
                random.shuffle(self.class_list)
            else:
                # shuffle imagedb
                random.shuffle(self.imdb)

        # configure scales
        if len(self.scales) == 0:
            self.scales.append(600)

        # configure resize scales added by zhaoxian
        if len(self.resize_scales) == 0:
            self.resize_scales.append(-1.0)

        # configure mean value
        mean_value = kwargs.pop('mean_value', None)
        if mean_value is None:
            raise ValueError('mean value in required')

        self.mean_value = []
        if isinstance(mean_value, (int, float)):
            self.mean_value.append(mean_value)
        elif isinstance(mean_value, collections.Iterable):
            for v in mean_value:
                self.mean_value.append(v)
        else:
            raise TypeError('mean value invalid')

        num_channels = 3 if self.is_color else 1
        if not (len(self.mean_value) == 1 or len(self.mean_value) == num_channels):
            raise ValueError('Specify either 1 mean_value or as many as channels')
        if num_channels > 1 and len(self.mean_value) == 1:
            for _ in range(num_channels - 1):
                self.mean_value.append(self.mean_value[0])

        # transform scale
        self.transform_scale = kwargs.pop('scale', 1.0)

        # max size
        self.max_size = kwargs.pop('max_size', 0.0)

        # coff_c
        self.coff_c = kwargs.pop('coff_c', 0.1)

        # coff_min_r
        self.coff_min_r = kwargs.pop('coff_min_r', 0.67)

        # coff_max_r
        self.coff_max_r = kwargs.pop('coff_max_r', 1.5)

        # coff_hue
        self.coff_hue = kwargs.pop('coff_hue', 0.05)

        # scale_w_h_ratio
        self.scale_w_h_ratio = kwargs.pop('scale_w_h_ratio', 1.0)

        # max_rois
        self.max_rois = kwargs.pop('max_rois')

        super(RoiYoloDataset, self).__init__(input_dimension=self.multi_scale())

    def _read_labels(self):
        """
        read labels from source and roidb
        :return: none
        """
        with open(self.source, 'r', encoding='utf-8') as source_file:
            lines = source_file.readlines()

        for line in lines:
            file_name = line.strip()
            file_data = None
            if self.cache_images:
                if self.image_extension != '':
                    file_path = self.root_folder + file_name + '.' + self.image_extension
                else:
                    file_path = self.root_folder + file_name
                with open(file_path, 'rb', encoding='utf-8') as image_file:
                    file_data = image_file.read()

            entry = ImageDatabase()
            entry.name = file_name
            entry.cache = file_data

            self.imdb.append(entry)
        logging.info('Number of images:%d' % len(self.imdb))

        # load roidb
        if self.output_labels:
            with open(self.roidb_file, 'r', encoding='utf-8') as roidb_file:
                num_images = roidb_file.readline()
                num_images = int(num_images)

                if num_images != len(self.imdb):
                    raise ValueError('Numbers of images and annotations mismatch')

                for i in range(num_images):
                    line = roidb_file.readline().strip()
                    line_splits = line.split(' ')
                    num_rois = int(line_splits[0])

                    if num_rois == 0:
                        if self.balanced_sampling:
                            self.class_list[0].sample_list.append(i)
                    else:
                        line_splits = line_splits[1:]
                        for j in range(num_rois):
                            roi = Roi()
                            roi.x1 = int(line_splits[j * 5])
                            roi.y1 = int(line_splits[j * 5 + 1])
                            roi.x2 = int(line_splits[j * 5 + 2])
                            roi.y2 = int(line_splits[j * 5 + 3])
                            roi.label = int(line_splits[j * 5 + 4])

                            self.imdb[i].gt_boxes.append(roi)

                            if self.balanced_sampling:
                                while roi.label + 1 > len(self.class_list):
                                    self.class_list.append(SampleList())
                                self.class_list[int(roi.label)].sample_list.append(i)

            if self.balanced_sampling:
                # 删除空的classes
                for i in range(len(self.class_list)-1, -1, -1):
                    if len(self.class_list[i].sample_list):
                        self.class_list.pop(i)
                logging.info('Number of valid classes:%d' % len(self.class_list))

            logging.info('Load roidb file completed')

    def __len__(self):
        return len(self.imdb)

    def multi_scale(self):
        if self.train and len(self.scales) > 1:
            cur_scale = self.scales[random.randint(1, 1000) % len(self.scales)]
        else:
            cur_scale = self.scales[0]
        # cur_scale = self.scales[0]
        return cur_scale, cur_scale

    @dataset.Dataset.resize_getitem
    def __getitem__(self, item):
        num_channels = 3 if self.is_color else 1

        im_scale = 1.0

        '''
        if self.train and len(self.scales) > 1:
            scale = self.scales[random.randint(1, 1000) % len(self.scales)]
        else:
            scale = self.scales[0]
        '''
        scale = self.input_dim[0]

        # 1.prepare image data
        # 1.1 read image
        db_id = item # self.lines_id
        if self.balanced_sampling:
            cursor = self.class_list[self.lines_id].cursor
            db_id = int(self.class_list[self.lines_id].sample_list[cursor])

        flag = cv2.IMREAD_COLOR if self.is_color else cv2.IMREAD_GRAYSCALE
        if self.cache_images:
            cv_img = cv2.imdecode(self.imdb[db_id].cache, flag)
        else:
            if self.image_extension != '':
                image_path = self.root_folder + self.imdb[db_id].name + '.' + self.image_extension
            else:
                image_path = self.root_folder + self.imdb[db_id].name
            # print('image_path:%s' % image_path)
            cv_img = cv2.imread(image_path, flag)
            if cv_img is None:
                raise ValueError('image_path:%s not exits' % image_path)

            #image crop
            if self.crop_mode:
                crop_region=[180,160,960,240]#tl_x,tl_y,w,h
                cv_img=cv_img[crop_region[0]:crop_region[0]+crop_region[2],crop_region[1]:crop_region[1]+crop_region[3]]

                #roi label process



        # 1.2 resize image
        height = cv_img.shape[0]
        width = cv_img.shape[1]
        im_size_min = min(width, height)
        im_size_max = max(width, height)

        # compute scale
        if self.train:
            if self.batch_size == 1:
                im_scale = float(scale) / im_size_min
                if int(im_scale * im_size_max) > self.max_size:
                    im_scale = float(self.max_size) / im_size_max

                # modified by zhaoxian 2016/09/01
                if self.resize_scales[0] > 0:
                    if self.train and len(self.resize_scales) > 1:
                        im_scale = self.resize_scales[random.randint() % len(self.resize_scales)]
                    else:
                        im_scale = self.resize_scales[0]

                cv_img = cv2.resize(cv_img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
            else:
                # set scale size
                scale_w = 0
                scale_h = 0
                if self.batch_size > 1:
                    scale_w = scale
                    scale_h = int(self.scale_w_h_ratio * scale)

                cv_img = cv2.resize(cv_img, (scale_w, scale_h), cv2.INTER_LINEAR)
        else:
            scale_w = scale
            scale_h = int(self.scale_w_h_ratio * scale)
            cv_img = cv2.resize(cv_img, (scale_w, scale_h), cv2.INTER_LINEAR)

        # 1.3 flip image
        do_mirror = self.mirror
        if self.train:
            do_mirror = self.mirror and (random.randint(1, 1000) % 2)

        # 1.4 HSV颜色空间随机调整对比度和色度，进行数据扩增处理, add by lalei 161109
        if self.do_hsv_jitter and num_channels == 3:
            cv_img = self._hsv_color_jitter(cv_img)

        # 1.5 transform image
        cv_img = self._transform(cv_img, do_mirror)

        new_width = cv_img.shape[1]
        new_height = cv_img.shape[0]

        # 2. prepare gt_boxes
        if self.output_labels:


            #set the valid ground truth
            valid_gt_boxes=[]
            for roi in self.imdb[db_id].gt_boxes:
                # 2.0 crop roi
                if self.crop_mode:
                    pre_roi_w=roi.x2-roi.x1+1
                    pre_roi_h=roi.y2-roi.y1+1


                    roi.x1 = min(crop_region[0]+crop_region[2] - 1,max(crop_region[0],roi.x1))-crop_region[0]
                    roi.y1 = min(crop_region[1]+crop_region[3] - 1, max(crop_region[1], roi.y1))-crop_region[1]
                    roi.x2 = min(crop_region[0]+crop_region[2] - 1, max(crop_region[0], roi.x2))-crop_region[0]
                    roi.y2 = min(crop_region[1]+crop_region[3] - 1, max(crop_region[1], roi.y2))-crop_region[1]

                    post_roi_w=roi.x2-roi.x1+1
                    post_roi_h=roi.y2-roi.y1+1

                    #截断过多或不在crop区域内的GT_BOX过滤掉
                    if post_roi_h/pre_roi_h<0.5 or pre_roi_w/post_roi_w<0.5:
                        continue


                # 2.1 flip roi
                if do_mirror:
                    width_minus_1 = width - 1
                    x1 = roi.x1
                    roi.x1 = width_minus_1 - roi.x2
                    roi.x2 = width_minus_1 - x1

                # 2.2 scale roi
                if self.batch_size == 1 and self.train:
                    roi.x1 *= im_scale
                    roi.x2 *= im_scale
                    roi.y1 *= im_scale
                    roi.y2 *= im_scale
                else:
                    roi.x1 *= float(scale_w) / width
                    roi.y1 *= float(scale_h) / height
                    roi.x2 *= float(scale_w) / width
                    roi.y2 *= float(scale_h) / height
                    new_width = scale_w
                    new_height = scale_h

                valid_gt_boxes.append(roi)

        '''
        # 2.4 set label data
        out_label = np.zeros((self.max_rois + 1, 5), np.float32)
        out_label[0, 0] = new_width
        out_label[0, 1] = new_height
        if not self.train:
            im_scale = 1.0
        out_label[0, 2] = im_scale

        if self.output_labels:
            for i, roi in enumerate(self.imdb[db_id].gt_boxes):
                out_label[i + 1, 0] = (roi.x1 + roi.x2) / 2 / new_width
                out_label[i + 1, 1] = (roi.y1 + roi.y2) / 2 / new_height
                out_label[i + 1, 2] = (roi.x2 - roi.x1 + 1) / new_width
                out_label[i + 1, 3] = (roi.y2 - roi.y1 + 1) / new_height
                out_label[i + 1, 4] = roi.label - 1  # 兼容frcnn标签

                if i >= self.max_rois - 1:
                    break
        '''


        # 2.4 set label data
        out_label = np.zeros((self.max_rois, 6), np.float32)
        if self.output_labels:
            #for i, roi in enumerate(self.imdb[db_id].gt_boxes):

            for i, roi in enumerate(valid_gt_boxes):
                out_label[i, 0] = 1.0
                out_label[i, 1] = roi.x1 / new_width
                out_label[i, 2] = roi.y1 / new_height
                out_label[i, 3] = roi.x2 / new_width
                out_label[i, 4] = roi.y2 / new_height
                out_label[i, 5] = roi.label - 1  # 兼容frcnn标签

                if i >= self.max_rois - 1:
                    break

        # view the bbox on the images
        show_gt_box_img = self._inversetranform(cv_img)
        color_list=[(255,0,0),(0,255,0),(0,0,255),(0,255,255),(255,0,255),(255,255,0),
                    (128,128,255),(128,255,128),(255,128,128),(0,128,128),(128,0,128),(128,128,0)]
        for i, roi in enumerate(valid_gt_boxes):
            cv2.rectangle(show_gt_box_img,(roi.x1,roi.x2),(roi.x2,roi.y2),color_list[roi.label-1],thickness=1)
            if i >= self.max_rois - 1:
                break

        if not os.path.exists('./label_data_view'):
            os.mkdir('label_data_view')
        cv2.imwrite(os.path.join('label_data_view',self.imdb[db_id].name + '.' + self.image_extension),show_gt_box_img)
        cv2.waitKey(1)





        # go to the next iter
        if self.balanced_sampling:
            self.class_list[self.lines_id].cursor += 1
            if self.class_list[self.lines_id].cursor + 1 > len(self.class_list[self.lines_id].sample_list):
                self.class_list[self.lines_id].cursor = 0
                if self.shuffle:
                    random.shuffle(self.class_list[self.lines_id].sample_list)

            self.lines_id += 1
            if self.lines_id + 1 > len(self.class_list):
                self.lines_id = 0
                if self.shuffle:
                    random.shuffle(self.class_list)
        else:
            self.lines_id += 1
            if self.lines_id + 1 > len(self.imdb):
                self.lines_id = 0
                if self.shuffle:
                    random.shuffle(self.imdb)

        # transpose to [c, h, w]
        out_image = cv_img.transpose([2, 0, 1])
        #print('out_image shape:', out_image.shape)
        #print('out_label shape:', out_label.shape)
        #np.save('./pytorch_image.npy', out_image)
        #exit(0)

        return out_image, out_label

    def _hsv_color_jitter(self, src_img):
        if not isinstance(src_img, np.ndarray):
            raise TypeError('img type invalid')

        img_w = src_img.shape[0]
        img_h = src_img.shape[1]

        # hsv空间对比度调整，sv进行幂次变化，h加[-0.05, 0.05]的均匀分布的随机扰动
        hsv_img = np.zeros(src_img.shape, np.float)
        cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV, hsv_img)

        (f_h_img, f_s_img, f_v_img) = cv2.split(hsv_img)
        f_hsv_img = np.zeros((img_w, img_h, 3), np.float)
        f_h_img /= 256
        f_s_img /= 256
        f_v_img /= 256

        # h加均匀扰动,如果coff_hue=0，不进行色度调整
        alpha = random.uniform(-self.coff_hue, self.coff_hue)
        f_h_img = f_h_img + alpha

        # sv进行幂变换
        r = random.uniform(self.coff_min_r, self.coff_max_r)
        for h in range(img_h):
            for w in range(img_w):
                f_s_img[h, w] = self.coff_c * pow(f_s_img[h, w], r)
                f_v_img[h, w] = self.coff_c * pow(f_v_img[h, w], r)

        cv2.merge((f_h_img, f_s_img, f_v_img), f_hsv_img)

        # 和matlab im2uint8一样，输入数据乘以255，小于0的数设置为0，大于255的数设置为255，其他的数四舍五入
        u_hsv_img = f_hsv_img * 255
        np.clip(u_hsv_img, 0, 255, u_hsv_img)
        u_hsv_img = np.uint8(u_hsv_img)

        u_dst_bgr = np.zeros(u_hsv_img.shape, np.uint8)
        cv2.cvtColor(u_hsv_img, cv2.COLOR_HSV2BGR, u_dst_bgr)

        return u_dst_bgr

    def _transform(self, src_img, mirror):
        # convert to float
        src_img = src_img.astype(dtype=np.float)

        # mean value
        src_img -= self.mean_value

        # mirror
        if mirror:
            src_img = cv2.flip(src_img, 1)

        # transfrom scale
        src_img *= self.transform_scale
        return src_img

    def _inversetranform(self,src_img):
        dst_img=src_img/self.transform_scale+self.mean_value
        dst_img=dst_img.astype(dtype=np.int8)
        return dst_img
