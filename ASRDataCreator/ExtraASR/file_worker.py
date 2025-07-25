import json
import os
import shutil
import re
def write_json(file_path,dic):
    res = None
    try:
        res = json.dumps(obj=dic,ensure_ascii=False,indent=4)

        output = re.sub(r'": \[\s+', '": [', res)
        output = re.sub(r'",\s+', '", ', output)
        output = re.sub(r'"\s+\]', '"]', output)
        output = re.sub(r'"\s+\]', '"]', output)

        with open(file_path,"w+",encoding="UTF-8") as file:
            file.writelines(output) 
    except Exception as exp:
        print(f"Error with write json in {file_path}")
        print("Exception",exp)
        print("dict",res)
        # raise exp

def update_json(file_path,dic):
    old = None
    try:
        old = read_json(file_path)
        old.append(dic)
        write_json(file_path, old)

    except Exception as exp:
        print(f"Error with update json in {file_path}")
        print("Exception",exp)
        print("dict",old)
        # raise exp


def read_json(file_path):
    try:
        with open(file_path,"r",encoding="UTF-8") as file:
            lines = []
            for line in file:
                lines.append(line.rstrip("\n"))
            s = "".join(lines)
        return json.loads(s)
    except Exception as ex:
        print(f"File {file_path} has err {ex}")
        return None


def read_file(path):
    try:
        lines = []
        with open(path, "r+",encoding="UTF-8") as file:
            for line in file:
                lines.append(line.rstrip("\n"))
            return lines
    except:
        return []
    
def write_file(path,lines):
    try:
        with open(path, "w+",encoding="UTF-8") as file:
            for line in lines:
                file.write(f"{line}\n")
    except:
        print(f"Data could not be recorded  in {path}")

def add_line(path,line):
    try:
        with open(path,"a") as file:
            s = line+"\n"
            file.write(s)
    except Exception as ex:
        print(ex)
        print(f"The error occurred while adding a line {line} to the list in {path}")

def add_lines(path, lines):
    try:
        with open(path,"a") as file:
            for line in lines:
                s = line+"\n"
                file.write(s)
    except Exception as ex:
        print(ex)
        print(f"The error occurred while adding a line {line} to the list in {path}")



def clear_directory(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

def clear_empty_dir(path):
    for  dirpath, dirnames, filenames in os.walk(path):
        if len(dirnames) == 0:
            shutil.rmtree(path)
            print(f"directory {path} was deleted due to emptiness")
            return True
        return False
    
def create_or_pass_dir(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def find_extension(file_name):
    for i in range(len(file_name)):
        if file_name[i] == ".":
            return file_name[i+1:]
    return None


def save_annotation(file_path,data,sentences):
    
    with open(os.path.join(file_path,"annotation.txt"),"w+",encoding="UTF-8") as file:
        for i,sentence in sentences:
            file.write(f"{i} {sentence}\n")

def save_data(file_path,data, **kwargs):
    data.update(kwargs)
    write_json(os.path.join(file_path,"data.json"),data)

def save_metrics(file_path,metrics):
    write_json(os.path.join(file_path,"metrics.json"),metrics)


def copy_file(from_path,to_path):
    shutil.copyfile(from_path,to_path)

def get_name_files(files):
    name_audio, name_data, name_video = None,None,None
    for file in files:
        if file.startswith("audio"):
            name_audio = file
        elif file.startswith("data"):
            name_data = file
        elif  file.startswith("video"):
            name_video = file
    if name_audio is None:
        print("audio not found")
    if name_data is  None:
        print("data not found")
    if name_video is None:
        print("video not found")
    return name_audio, name_data, name_video

def get_arr(config_var): # noqa
    arr = read_file(config_var)
    if (len(arr)) != 0:
        return arr
    try:
        config_var = json.loads(config_var,object_hook=list)
        if type(config_var) == str:
            config_var = [config_var]
    except:
        config_var = [config_var]
    return config_var