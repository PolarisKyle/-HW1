# -HW1
李宏毅机器学习HW1-PM2.5预测
本次作业使用豐原站的觀測記錄，分成 train set 跟 test set，train set 是豐原站每個月的前20天所有資料，test set則是從豐原站剩下的資料中取樣出來。 
train.csv:每個月前20天每個小時的氣象資料(每小時有18種測資)。共12個月。 
test.csv:從剩下的資料當中取樣出連續的10小時為一筆，前九小時的所有觀測數據當作feature，第十小時的PM2.5當作answer。一共取出240筆不重複的 test data，請根據feauure預測這240筆的PM2.5。
ML.PM2.5.py为代码运行的脚本文件。testpp.py为测试去均值和方差归一化的测试脚本，对结果无影响。
