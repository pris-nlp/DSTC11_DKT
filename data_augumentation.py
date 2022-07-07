from typing import Dict, List, Callable, Any
import os
import json,csv
import random
from util import *


def back_translate(text: str,
                   schemas: Dict[str, List[List[str]]],
                   keywords: List[str] = None,
                   handle_func: Callable[[Dict[str, str]], Any] = lambda x: x) -> Any:
    """
    输入一句话，使用不同翻译平台、翻译模式（中间语言）进行数据增强，生成多个回复句子。

    参数：
    ------
    text: ``str``
        输入的句子
    schemas: ``Dict[str, List[List[str]]]``
        定义了翻译平台和翻译模式（中间语言），见``input/schemas.json``
    keywords: ``List[str]``, optional, default=``None``
        如果指定了keywords，则使用keyword mask的方法，否则不使用
    handle_func: ``Callable[[Dict[str, str]], Any]``, optional, default=``lambda x: x``
        对结果（res）进行处理，例如：
        过滤掉重复的生成结果、改变输出结构、限制最大生成个数、使用匹配模型进行过滤等
    """
    text_b=""
    res = {"origin": text}
    for platform, schema_list in schemas.items():
        trans_func = __import__(f"{platform}.main", fromlist=platform).back_translate
        for schema in schema_list:
            try:
                schema_key = "->".join(schema)
                res[f"{platform}    {schema_key}"] = trans_func(text, lang_list=schema)
                text_b = res[f"{platform}    {schema_key}"]
            except Exception:
                pass

            '''
            if keywords:  # 使用keyword mask
                keywords = list(set(keywords))  # 过滤重复keywords
                hit_keywords = [keyword for keyword in keywords if keyword in text]
                for selected_keyword in hit_keywords:
                    try:
                        replaced_text = text.replace(selected_keyword, "UNK")
                        back_translate_res = trans_func(replaced_text, lang_list=schema)
                        if "UNK" in back_translate_res or "unk" in back_translate_res:
                            back_translate_res = back_translate_res.replace("UNK", selected_keyword)
                            back_translate_res = back_translate_res.replace("unk", selected_keyword)
                            res[f"{platform}    {schema_key}    kw_mask{selected_keyword}"] = back_translate_res
                    except Exception:
                        pass
            '''
    #return handle_func(res)
    return text_b


def main(input_dir, output_dir, type):
    print("sdnsdnsdnsdn.......")
    if type == "back_translation":
        schemas = {"baidu": [["en", "zh-CN", "en"]]}
        with open('./data/file7.tsv', 'w') as f2:
            tsv_w = csv.writer(f2, delimiter='\t')
            tsv_w.writerow(['text', 'text_1', 'label'])  # 单行写入
            tsv_w.writerows([[1, 'Frank', 99], [2, 'John', 70]])
            #csv_writer.writerow(['text', 'text_2', 'label'])
            with open(os.path.join(input_dir, "train.tsv"), "r") as f:
                reader = csv.reader(f, delimiter="\t", quotechar=None)
                i = 0
                for line in reader:
                    if i<=6573:
                        i += 1
                        continue
                    text_1 = line[0]
                    text_2 = back_translate(line[0], schemas)
                    label = line[1]
                    print([text_1, text_2, label])
                    tsv_w.writerow([text_1, text_2, label])
                    i += 1
            print(i)
            print("finished!!!")
            f.close()
            f2.close()




def test():
    def handle_res(res):
        no_repeat = list(set(res.values()))
        for item in no_repeat:
            print(item)
        return no_repeat

    schemas = {"baidu": [["en", "zh-CN", "en"]]}
    #schemas = json.load(open("./input/schemas.json", "r"))
    #keywords = [line.strip() for line in open("./input/keywords.txt", "r")]
    result = back_translate("i need to get some assistance figuring out how to rollover my 401k, please", schemas)
    result = json.dumps(result, indent=4, ensure_ascii=False)
    print(result)
    #print(json.dumps(result, indent=4, ensure_ascii=False))
    #print(json.dumps(handle_res(result), indent=4, ensure_ascii=False))


if __name__ == "__main__":
    input_dir = "data/banking"
    output_dir = "data/banking"
    print("ininininininininininin")
    main(input_dir, output_dir, type="back_translation")
