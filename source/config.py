config = {
    "segment": {
        "raw_data": ["data/segment/pku_training.utf8"],
        "processed_data": "data/segment/training_data.pkl",
        "dicts":"data/segment/dicts.pkl",
        "model": "data/segment/model",
        "history": "data/segment/history",
        "input": "data/segment/input.txt",
        "output": "data/segment/output.txt",
        "stand": "data/segment/stand.txt",
        "transp": "data/segment/transp.json",
        "tags": ["S", "B", "M", "E"],
        "epochs":20,
        "sequence_length": 16,
        "word_dimension": 256,
        "hidden_units": 64
    },
    "label": {
        "raw_data": ["data/label/peoples_daily.txt"],
        "processed_data": "data/label/training_data.pkl",
        "model": "data/label/model",
        "history": "data/label/history",
        "dicts": "data/label/dicts.pkl",
        "features": "data/label/features.pkl",
        "input": "data/label/input.txt",
        "output": "data/label/output.txt",
        "stand": "data/label/stand.txt",
        "diff": "data/label/diff.txt",
        "transp": "data/label/transp.json",
        "tags": ["x","a","ad","ag","an","b","bg","c","d","dg","e","f","h","i","j","k","l","m","mg","n","na","ng","nr","ns","nt","nx","nz","o","p","q","r","rg","s","t","tg","u","v","vd","vg","vn","w","y","yg","z"],
        "tag_detail": {"a": "形容词","ad": "副形词","ag": "形语素","an": "名形词","b": "区别词","c": "连词","d": "副词","dg": "副语素","e": "叹词","f": "方位词","g": "语素","h": "前接成分","i": "成语","j": "简称略语","k": "后接成分","l": "习用语","m": "数词","n": "名词","ng": "名语素","nr": "人名","ns": "地名","nt": "机构团体","nx": "字母专名","nz": "其他专名","o": "拟声词","p": "介词","q": "量词","r": "代词","s": "处所词","t": "时间词","tg": "时语素","u": "助词","ud": "结构助词","ug": "时态助词","uj": "结构助词的","ul": "时态助词了","uv": "结构助词地","uz": "时态助词着","v": "动词","vd": "副动词","vg": "动语素","vn": "名动词","w": "标点符号","x": "非语素字","y": "语气词","z": "状态词",
        },
        "epochs": 20,
        "sequence_length": 16,
        "word_dimension": 256,
        "hidden_units": 64,
    },
    "setting": {
        "temp":"data/temp/",
        "img":"data/img/",
    },
}

