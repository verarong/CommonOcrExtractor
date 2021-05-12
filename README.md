# ocr_extractor

# [票据识别主工程](https://github.com/verarong/invoice_ocr) 

# 处理流程

![](http://www.weikunt.cn:7788/selif/qfazrl0x.png)

# 实现功能

ocr通用后处理，可视化结构化数据提取器

1.支持labelme进行需要提取字段的可视化标注

2.支持定义各字段的合法范围，并自动生成mask，对ocr结果点乘遮罩

3.实现原理为编辑距离的相似度和各字段相对位置投票，使用状态机控制

4.自动文本框切分，有效解决相近字段文本框在文本定位时框在一起的情况

5.支持配置转行等特殊处理

6.支持配置输出字段的特定样式化


# 使用方式

作为子项目导入ocr主项目：from app.extractor.information_extraction import DataHandle

将ocr的所有文本框及识别结果传入DataHandle一键食用：

state, predict = DataHandle(ocr_score, box, score_, invoice_type, invoice_direction_filter,
                                                True).extract()
