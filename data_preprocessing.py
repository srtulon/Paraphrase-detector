import os
import re
import csv
import json



datalist=[]


file_name="train"
datalist.append(["Q1","Q2","result"])

with open(os.path.join(os.path.dirname(__file__), "data\\"+file_name+".data"), "r") as myFile:
    
    
    for tline in myFile:
        tline=re.split(r" {2,}", tline)
        s=tline[0].split("\t")
        
        # if s[2] not in datalist:
        #     datalist.append(s[2])
        # if s[3] not in datalist:
        #     datalist.append(s[3])    
                     
        result= 1
        if s[4]=="(3, 2)" or s[4]=="(4, 1)" or s[4]=="(5, 0)" or s[4]=="5" or s[4]=="4":
            datalist.append([s[2],s[3],result])
        elif s[4]=="(1, 4)" or s[4]=="(0, 5)"  or s[4]=="0" or s[4]=="1" or s[4]=="2":
            result= 0
            datalist.append([s[2],s[3],result])
        else:
            pass
        print(s[2],end=" ")
        print(s[3],end=" ")
        print(s[4])

file_name="dev"

with open(os.path.join(os.path.dirname(__file__), "data\\"+file_name+".data"), "r") as myFile:
    
    
    for tline in myFile:
        tline=re.split(r" {2,}", tline)
        s=tline[0].split("\t")
        
        # if s[2] not in datalist:
        #     datalist.append(s[2])
        # if s[3] not in datalist:
        #     datalist.append(s[3])    
                     
        result= 1
        if s[4]=="(3, 2)" or s[4]=="(4, 1)" or s[4]=="(5, 0)" or s[4]=="5" or s[4]=="4":
            datalist.append([s[2],s[3],result])
        elif s[4]=="(1, 4)" or s[4]=="(0, 5)"  or s[4]=="0" or s[4]=="1" or s[4]=="2":
            result= 0
            datalist.append([s[2],s[3],result])
        else:
            pass
        print(s[2],end=" ")
        print(s[3],end=" ")
        print(s[4])
        

file_name="test"

with open(os.path.join(os.path.dirname(__file__), "data\\"+file_name+".data"), "r") as myFile:
    
    
    for tline in myFile:
        tline=re.split(r" {2,}", tline)
        s=tline[0].split("\t")
        
        # if s[2] not in datalist:
        #     datalist.append(s[2])
        # if s[3] not in datalist:
        #     datalist.append(s[3])    
                     
        result= 1
        if s[4]=="(3, 2)" or s[4]=="(4, 1)" or s[4]=="(5, 0)" or s[4]=="5" or s[4]=="4":
            datalist.append([s[2],s[3],result])
        elif s[4]=="(1, 4)" or s[4]=="(0, 5)"  or s[4]=="0" or s[4]=="1" or s[4]=="2":
            result= 0
            datalist.append([s[2],s[3],result])
        else:
            pass
        print(s[2],end=" ")
        print(s[3],end=" ")
        print(s[4])       
        
file_name="train_dev_test"
                
with open(os.path.join(os.path.dirname(__file__), "data_processed\\"+file_name+".csv"), 'a', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(datalist)     
    
# with open(os.path.join(os.path.dirname(__file__), "data_processed\\"+file_name+"2.txt"), 'w', encoding='UTF8', newline='') as f:    
#     f.write(json.dumps(datalist))
#     f.close()  
        