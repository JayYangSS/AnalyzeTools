import xlwt
import collections



def generate_result_dict(file_path):
    result_file=open(file_path,'r')
    result_lines=result_file.readlines()

    class_result_list=[]
    class_name_list=[]
    for line_id,line in enumerate(result_lines):

        group_id=line_id//7
        if line_id%7==0:#class name
            class_result_list.append(collections.OrderedDict())
            class_name_list.append(line.split(':')[0])
        else:
            if line.startswith('='):
                continue
            property_name=line.split(':')[0].strip()
            property_val=line.split(':')[1].strip()
            class_result_list[group_id][property_name]=property_val

    return class_name_list,class_result_list

if __name__=='__main__':
    class_name_list,class_result_list=generate_result_dict(u'/Users/jayn/PycharmProjects/Tools/result.txt')



    workbook = xlwt.Workbook(encoding='utf-8')
    sheet = workbook.add_sheet(u'worksheet', cell_overwrite_ok=True)
    sheet.col(0).width = 256 * 15  # 设置第一列的宽度为15，宽度的基本单位为256.所以设置的时候一般用256 × 需要的列宽。

    # 设置行高为可以修改，并修改为 40，行高的基本单位为20，设置同行高。
    sheet.row(0).height_mismatch = True
    sheet.row(0).height = 20 * 20
    for id,class_name in enumerate(class_name_list):
        sheet.write(0,id+1,class_name)
        result_dict=class_result_list[id]

        line_id=1
        for k,v in result_dict.items():
            sheet.write(line_id,0,k)
            sheet.write(line_id,id+1,v)
            line_id+=1




    workbook.save('result.xls')
