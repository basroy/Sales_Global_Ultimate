import re, os

basedir = os.path.abspath(os.path.dirname(__file__))
print('path returned is ' + basedir) 
remote = os.getcwd()=='/home/basroy/scripts/python'
local_path = 'C:/Work/'
remote_path = '/home/basroy/data'
path = remote_path if remote else local_path

logfile = 'sclass_refresh.txt'
regex = '(<property name="(.*?)">(.*?)<\/property>)'
match_list = []
read_line = True
 
with open(os.path.join(local_path,logfile), "r") as file:
    match_list = []
    if read_line == True:
        for line in file:
            for match in re.finditer(regex, line, re.S):
                match_text = match.group()
                match_list.append(match_text)
                #print(match_text)

            x = re.search("./AnaplanClient.sh", line)
            y = "0" if x is None else x
            #print(line)
  #https://www.dataquest.io/blog/regular-expressions-data-scientists/
  # 
  # https://github.com/glenjarvis/talk-yaml-json-xml-oh-my
  # 
  # https://pythonicways.wordpress.com/2016/12/20/log-file-parsing-in-python/
  # 
  # https://opensource.com/article/19/5/log-data-apache-spark
  # https://blog.red-badger.com/2013/11/08/getting-started-with-elasticsearch
  #           
            if y == x:
                print(line)
            model_name = r'('
            oper = re.search("The operation was successful", line)
            oper_status = "0" if oper is None else oper
            if oper_status == oper:
                print(line)        
            oper = re.search("No dump file is available", line)
            oper_status = "0" if oper is None else oper
            if oper_status == oper:
                print(line)  
            oper = re.search("The operation has failed", line)
            oper_status = "0" if oper is None else oper
            if oper_status == oper:
                print(line)        
                
    else:
        data = f.read()
        for match in re.finditer(regex, data, re.S):
            match_text = match.group()
            match_list.append(match_text)
file.close()
