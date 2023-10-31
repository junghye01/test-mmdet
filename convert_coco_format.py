import os
import cv2
import json
import base64
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import KFold
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, ElementTree

def convert_to_coco(xml_paths):
    
    res=defaultdict(list)
    
    categories={
        'Others':0,
        'Porosity':1,
        'Slag':2
    }
    
    n_id=0
    #image_id=0
    for xml_path in tqdm(xml_paths):
        with open(xml_path,'r') as xml:
            tree=ET.parse(xml)
            root=tree.getroot()
            
            file_name=root.find('filename').text[:-4]
            parts=file_name.split('-')
            prefix = ord(parts[0][0]) - ord('a') + 1
            
            suffix = parts[1]
            image_id = int(f"{prefix}{parts[0][1:]}{suffix}")
            
            
            
            res['images'].append({
                'id':image_id,
                'width':int(root.find('size').find('width').text),
                'height':int(root.find('size').find('height').text),
                'file_name':root.find('filename').text
                
                
            })
            # conda activate /home/public/yunvqa/anaconda3/envs/blip2/envs/openmmlab
            
            objects=root.findall('object')
            
            for obj in objects:
                defect_type=obj.find('name').text
                if defect_type=='Porotisy':
                    defect_type='Porosity'
                
                bndbox=obj.find('bndbox')
                x1,y1=int(bndbox.find("xmin").text),int(bndbox.find("ymin").text)
                x2,y2=int(bndbox.find("xmax").text),int(bndbox.find("ymax").text)
                
                w,h=x2-x1,y2-y1
                
                res['annotations'].append({
                    'id':n_id,
                    'image_id':image_id,
                    'category_id':categories[defect_type],
                    'area':w*h,
                    'bbox':[x1,y1,w,h],
                    'iscrowd':0,
                })
                n_id+=1
                
    for name,id in categories.items():
        res['categories'].append({
            'id':id,
            'name':name,
        })
        
    return res


if __name__=='__main__':
    
    os.makedirs('data_completed/annos/',exist_ok=True)
    xml_paths=np.array(glob(os.path.join('data_completed/origin/train/Label/','*.xml')))
    #xml_paths=np.array(glob(os.path.join('data_completed/origin/val/Label/','*.xml')))
    train_json=convert_to_coco(xml_paths)
    with open(f'./data_completed/annos/train_annotations_full.json','w',encoding='utf-8') as f:
        json.dump(train_json,f,ensure_ascii=True,indent=4)
        
    
    # train/valid 10 fold split(make total json file)
    n_splits=10
    kf=KFold(n_splits=n_splits,shuffle=True,random_state=42)
    xml_paths=np.array(glob(os.path.join('data_completed/origin/train/Label/','*.xml')))
    
    for fold,(trn_idx,val_idx) in enumerate(kf.split(xml_paths)):
        val_json=xml_paths[val_idx]
        train_json=xml_paths[trn_idx]
        
        train_json=convert_to_coco(train_json)
        val_json=convert_to_coco(val_json)
        
        with open(f'./data_completed/annos/train_annotations_{n_splits}split_{fold}fold.json', 'w', encoding='utf-8') as f:
            json.dump(train_json, f, ensure_ascii=True, indent=4)
        with open(f'./data_completed/annos/valid_annotations_{n_splits}split_{fold}fold.json', 'w', encoding='utf-8') as f:
            json.dump(val_json, f, ensure_ascii=True, indent=4)
        
        if fold==1:
            break
        
    
    
    
            
           
            
            
            
            
            
