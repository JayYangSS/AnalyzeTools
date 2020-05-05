#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: jiyang5
# date: 2020/05/04

import xlsxwriter
import os
import glob


class XLSWritter(object):

    #param:
    #work_dir:保存result.txt的路径
    def __init__(self):
        self.__work_dirs=[]
        self.__xlsx_path =''
        self.fppi_recall_list=[]
        self.workbook=''
        self.worksheet=''

    @property
    def work_dirs(self):
        return self.__work_dirs

    @property
    def save_dir(self):
        return self.__save_dir



    @work_dirs.setter
    def work_dirs(self,work_dirs):
        #fppi_recall_list=glob.glob(work_dir+'/fppi.*-recall.*\.txt')
        for work_dir in work_dirs:
            tmp_fppi_recall_list = glob.glob(work_dir + '/fppi*')
            assert 0!=len(tmp_fppi_recall_list),work_dir+' does not exist fppi_result files!'

            fppi_recall_list=[work_dir + '/fppi(0.1)-recall_0_32.txt']
            fppi_recall_list.append(work_dir + '/fppi(0.1)-recall_32_160.txt')
            fppi_recall_list.append(work_dir + '/fppi(0.1)-recall_128_720.txt')
            fppi_recall_list.append(work_dir + '/fppi(0.1)-recall_0_720.txt')

            self.fppi_recall_list.append(fppi_recall_list)
        self.__work_dirs=work_dirs

    @save_dir.setter
    def save_dir(self,save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.__save_dir=save_dir



    #excel中写入结果并绘制图表
    def save_xlsx_file(self):

        height_range_strs=['small','normal','large','all']
        xlsx_path=os.path.join(self.save_dir,'result.xlsx')

        if os.path.exists(xlsx_path):
            os.remove(xlsx_path)

        # 创建一个excel
        self.workbook = xlsxwriter.Workbook(xlsx_path)

        # 创建一个sheet
        self.worksheet = self.workbook.add_worksheet('fppi_recall')
        self.worksheet.set_column('A:A',13)


        # 设置字体
        bold = self.workbook.add_format({
            'bold': True,  # 字体加粗
            'border': 1,  # 单元格边框宽度
            'align': 'left',  # 水平对齐方式
            'valign': 'vcenter',  # 垂直对齐方式
            'fg_color': '#F4B084',  # 单元格背景颜色
            'text_wrap': True,  # 是否自动换行
        })

        tabel_head_format = self.workbook.add_format({
            'bold': True,
            'border': 6,
            'align': 'center',
            'valign': 'vcenter',
            'fg_color': '#D7E4BC',
        })


        line_id=1
        for group_id,fppi_result_group in enumerate(self.fppi_recall_list):

            self.worksheet.merge_range('A%d:K%d'%(line_id,line_id+1),self.work_dirs[group_id], tabel_head_format)
            line_id+=2

            for file_id,fppi_result_file_name in enumerate(fppi_result_group):

                fppi_result_file=open(fppi_result_file_name)
                fppi_result_lines=fppi_result_file.readlines()

                class_name_list=fppi_result_lines[0].split()


                ap_list = [fppi_result_lines[1].split()[0]+'_'+height_range_strs[file_id]+str(group_id)]
                ap_list.extend(map(float,fppi_result_lines[1].split()[1:]))

                #recall_list = [fppi_result_lines[2].split()[0]]
                recall_list = ['recall_'+height_range_strs[file_id]+str(group_id)]
                recall_list.extend(map(float,fppi_result_lines[2].split()[1:]))

                self.worksheet.write_row('A%d'%(line_id),class_name_list,bold)
                line_id += 1
                self.worksheet.write_row('A%d' % (line_id), ap_list)
                line_id += 1
                self.worksheet.write_row('A%d' % (line_id), recall_list)
                line_id += 1

                fppi_result_file.close()

            line_id+=1 #一组实验结果处理完后隔一行进行下组操作

        #insert chart
        self.chart_process('small',{'x_offset': 0, 'y_offset': 0})
        self.chart_process('normal',{'x_offset': 800, 'y_offset': 0})
        self.chart_process('large',{'x_offset': 0, 'y_offset': 500})
        self.chart_process('all',{'x_offset': 800, 'y_offset': 500})




    #insert chart

    def chart_process(self,height_range='small',chart_offset={'x_offset': 0, 'y_offset': 0}):

        row_id=0
        if height_range=='small':
            row_id=4
        elif height_range=='normal':
            row_id=7
        elif height_range=='large':
            row_id=10
        elif height_range=='all':
            row_id=13


        chart=self.workbook.add_chart({'type': 'column'})
        ap_chart = self.workbook.add_chart({'type': 'line'})

        chart.set_title({'name':height_range+' result',
                         'name_font': {
                             'name': 'Calibri',
                             'color': 'blue'
                         }})

        chart.set_size({'width': 800, 'height': 500})

        chart.set_y_axis({
            'name': 'recall@FPPI0.1',
            'name_font': {
                'name': 'Arial',
                'color': '#920000'
            },

        })

        for chart_id in range(len(self.work_dirs)):

            chart.add_series({'name':'fppi_recall!A%d'%(chart_id*15+row_id+1),
                              'categories': '=fppi_recall!B3:K3',
                              'values':'=fppi_recall!B%d:K%d'%(chart_id*15+row_id+1,chart_id*15+row_id+1),
                              #'data_labels': {'value': True,
                              #                'num_format': '#,##0.000'}
                              })

            #添加AP图表
            ap_chart.add_series({
                'name': '=fppi_recall!A%d'%(chart_id*15+row_id),
                'categories': '=fppi_recall!B3:K3',
                'values': '=fppi_recall!B%d:K%d'%(chart_id*15+row_id,chart_id*15+row_id),
            })

        chart.set_table()
        chart.combine(ap_chart)





        self.worksheet.insert_chart('M1',chart,chart_offset)
        #self.workbook.close()




if __name__=='__main__':
    xls_writter=XLSWritter()

    work_dirs=[u'/Users/jayn/Documents/工作文档/performance_eval/IntegratedDetectionEvalTool',
               u'/Users/jayn/Documents/工作文档/performance_eval/IntegratedDetectionEvalTool',
               u'/Users/jayn/Documents/工作文档/performance_eval/IntegratedDetectionEvalTool']
    xls_writter.work_dirs=work_dirs
    xls_writter.save_dir=u'/Users/jayn/Documents/工作文档/performance_eval/IntegratedDetectionEvalTool'
    xls_writter.save_xlsx_file()


    xls_writter.workbook.close()

    print('Generate xlsx file done!')


