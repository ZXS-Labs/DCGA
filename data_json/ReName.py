import json
 
def get_json_data():#获取json里面数据
    trains = {} # 用来存储数据
    with open('/home/xuyinuo/completionFormer/CompletionFormer/data_json/kitti_dc_1e4.json','r',encoding='utf8') as f:
        json_data = json.load(f)

        trains = json_data["train"]
        for train in trains:
            rgb = train["rgb"]
            rgb = rgb.split("/") # 这里使用随机数
            date = rgb[1]
            sync = rgb[2]
            rgb.insert(1,date)
            rgb.insert(2,sync)
            rgb = "/".join(rgb)
            train["rgb"] = rgb

        vals = json_data["val"]
        for val in vals:
            rgb = val["rgb"]
            rgb = rgb.split("/") # 这里使用随机数
            date = rgb[1]
            sync = rgb[2]
            rgb.insert(1,date)
            rgb.insert(2,sync)
            rgb = "/".join(rgb)
            val["rgb"] = rgb
            print(rgb)

        dicts = json_data # 将修改后的内容保存在dict中      
    return dicts
 
def write_json_data(dict):#写入json文件
    with open('/home/xuyinuo/completionFormer/CompletionFormer/data_json/kitti_dc_1e4_rePath.json','w') as r:
        json.dump(dict, r, indent=4, ensure_ascii=False)
 
file = get_json_data()
write_json_data(file)