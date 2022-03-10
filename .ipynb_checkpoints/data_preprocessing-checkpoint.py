import os
import re
import csv
import json

file_name="test"

datalist=[]

with open(os.path.join(os.path.dirname(__file__), "data\\"+file_name+".data"), "r") as myFile:
    for tline in myFile:
        tline=re.split(r" {2,}", tline)
        s=tline[0].split("\t")
        
        # if s[2] not in datalist:
        #     datalist.append(s[2])
        # if s[3] not in datalist:
        #     datalist.append(s[3])    
                     
        result= "yes"
        if s[4]=="(3, 2)" or s[4]=="(4, 1)" or s[4]=="(5, 0)":
            datalist.append([s[2],s[3],result])
        elif s[4]=="(1, 4)" or s[4]=="(0, 5)":
            result="no"
            datalist.append([s[2],s[3],result])
        else:
            pass
        print(s[2],end=" ")
        print(s[3],end=" ")
        print(s[4])
        
with open(os.path.join(os.path.dirname(__file__), "data_processed\\"+file_name+".csv"), 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(datalist)     
    
# with open(os.path.join(os.path.dirname(__file__), "data_processed\\"+file_name+"2.txt"), 'w', encoding='UTF8', newline='') as f:    
#     f.write(json.dumps(datalist))
#     f.close()  
        