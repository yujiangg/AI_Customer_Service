import time
import re
from fastapi import FastAPI
import sys
from utils.log import logger
import regex
from dotenv import load_dotenv
import datetime
import torch
import collections
from db import DBhelper
import torch.nn.functional as F
from AI_customer_service_2 import translation_stw

sys.path.append("..")
load_dotenv()
from AI_customer_service import QA_api, ChatGPT_AVD
from AI_Search import AI_Search
from AI_customer_service_2 import AICustomerAPI

description = """
# AI_customer_service
---
"""
tags_metadata = [
    {
        "name": "AI_service",
        "description": "QABOT"
    },
    {
        "name": "AI_Search",
        "description": "AI_Search"
    },
    {
        "name": "judge",
        "description": "judge"
    },
    {
        "name": "get_product",
        "description": "update_status"
    },
    {
        "name": "get_description",
        "description": "get_description"
    },
    {
        "name": "similarity",
        "description": "similarity"
    }
]

###讀取套件
app = FastAPI(title="hodo_ai", description=description, openapi_tags=tags_metadata)

# _AI_Search = AI_Search()
AI_judge = AICustomerAPI()

lang_dict = {'繁體中文': ['chinese', 'Chinese', '中文', '國語', '繁體中文', '简体中文', '簡體中文', '漢語', '普通話', '普通话', '汉语'],
             '英文': ['英文', 'lang']}

def check_status(web_id, group_id):
    timestamp = int(datetime.datetime.now().timestamp()) - 60
    q = f"""SELECT count(*) FROM web_push.AI_service_recommend_status x WHERE web_id ='{web_id}' and group_id = '{group_id}' and status < 2 and `timestamp` > {timestamp}"""
    data = DBhelper('jupiter_new').ExecuteSelect(q)
    return True if data[0][0] else False


def is_only_emoji(text):
    # 定义表情符号的正则表达式
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    # 使用全匹配模式
    return bool(emoji_pattern.fullmatch(text))


def is_pure_emoji(text):
    # 使用正则表达式匹配 emoji
    emoji_pattern = regex.compile("[\p{Emoji_Presentation}\p{Extended_Pictographic}]", flags=regex.UNICODE)
    emojis = emoji_pattern.findall(text)
    # 如果字符串长度等于 emoji 数量，则为纯 emoji
    return len(text) == len(emojis)

def get_tag_embedding():
    q = f"""SELECT web_id, question, ans,question_embedding  FROM AI_service_similarity"""
    print('讀取embedding資料中.....')
    ans_dict = collections.defaultdict(list)
    question_emb = collections.defaultdict(list)
    data = DBhelper('jupiter_new').ExecuteSelect(q)
    for web_id, question, ans, question_embedding in data:
        ans_dict[web_id].append((question, ans))
        question_emb[web_id].append(eval(question_embedding))
    question_emb_tensor = {}
    for web_id, emb in question_emb.items():
        question_emb_tensor[web_id] = torch.tensor(emb)
    return ans_dict, question_emb_tensor


def get_judge_text():
    q = f"""SELECT web_id,beginning ,product_inquiry ,return_or_exchange_request ,general_inquiry ,greeting ,expression_of_gratitude_or_end ,other,nativelang FROM AI_service_config"""
    data = DBhelper('jupiter_new').ExecuteSelect(q)
    dic = {}
    native_lang = {}
    for web_id, a, b, c, d, e, f, g, lang in data:
        if a == '_':
            a = "您好，我是客服機器人小禾！"
        if b == '_':
            b = "正在為您查詢商品,稍等一下呦！"
        if c == '_':
            c = "將為您提供退換貨說明,請稍待～"
        if d == '_':
            d = "請稍等,將為您提供相關資訊！"
        if e == '_':
            e = "謝謝您對我們的關注,祝您愉快！\n請問有什麼需要我幫忙解答的嗎？"
        else:
            e = e.replace("\\n", "\n")
        if f == '_':
            f = "很高興能解決您的問題,祝您愉快！"
        if g == '_':
            g = "請稍候一下我們將盡快為您解答"
        native_lang[web_id] = lang
        dic[web_id] = [a, b, c, d, e, f, g]
    return dic, native_lang


a_dict, q_emb_tensor = get_tag_embedding()
judge_text, native_lang = get_judge_text()


@app.get("/AI_service", tags=["AI_service"])
def ai_service(web_id: str = '', message: str = '', group_id: str = '', product: bool = True, lang: str = '繁體中文', main_web_id: str = '', types: int = 1):
    if web_id == '' or message == '' or group_id == '':
        return {"message": "no sentence input or no web_id", "message": ""}
    return AI_judge.qa(web_id, message, group_id, find_dpa=product, lang=lang, main_web_id=main_web_id, types=types)


@app.get("/update_product", tags=["get_product"])
def ai_update_product(web_id: str = '', group_id: str = '', lang='繁體中文', main_web_id: str = '', types: int = 1):
    main_web_id = web_id if main_web_id == '' else main_web_id
    AI_judge.update_recommend_status(web_id, group_id, 1, lang=lang, main_web_id=main_web_id, types=types)
    return 'ok'


@app.get("/similarity", tags=["similarity"])
def get_similarity_avivid(web_id: str = '', group_id: str = '', text: str = ''):
    if web_id not in q_emb_tensor:
        return
    emb = AI_judge.ask_gpt(message=text, model='gpt-text')
    cos_sim_curr = F.cosine_similarity(q_emb_tensor[web_id], torch.tensor(emb), dim=1)
    if float(cos_sim_curr.topk(1).values[0]) > 0.9:
        print(f'資料庫有相似問題：{a_dict[web_id][int(cos_sim_curr.topk(1).indices[0])][0]}')
        return a_dict[web_id][int(cos_sim_curr.topk(1).indices[0])][1]


@app.get("/get_description", tags=["get_description"])
def ai_description(title: str = ''):
    res = AI_judge.get_des(title)
    return res


@app.get("/judge", tags=["judge"])
def ai_service_judge(web_id: str = '', group_id: str = '', message: str = '', main_web_id: str = ''):
    main_web_id = web_id if main_web_id == '' else main_web_id
    beg, pi, rt, ge, gr, end, oth = judge_text[main_web_id]
    if message.isdigit():
        return 7, "親愛的顧客您好，請您再次描述問題細節，謝謝！\nDear customer, Please provide further details regarding the issue once again. Thank you!", None
    if re.sub('[^\u4e00-\u9fa5]+', '', message) == '好':
        return 5, end, '繁體中文'
    if message.split('_')[-1] in ['ANIMATION', 'STATIC', 'POPUP'] or re.match(r'(^\(\w.+\)$)', message) or message.startswith('http') or is_pure_emoji(message) or is_only_emoji(message):
        reply = "親愛的顧客您好，客服機器人小禾只懂文字敘述，若您有需要協助解答的問題，請協助提供文字提問，小禾將儘快提供回應！"
        eng_reply = "Dear customer, hello! I am the customer service chatbot. I only understand text descriptions. If you need assistance or have any questions, please provide your query in text form, and I will respond as quickly as possible!"

        if native_lang[main_web_id] != 'english':
            reply = AI_judge.translate('繁體中文', reply, native_lang[main_web_id])

        return 7, reply+'\n'+eng_reply, None

    start = time.time()
    status = check_status(web_id, group_id)
    print(f'{group_id}:的狀態是{status}')

    n_lang = native_lang[main_web_id]
    reply = "" if status else beg
    #if main_web_id in ['avividai', 'AviviD']: //全部開放
    tr = True
    lang = AI_judge.check_lang(message)
    print(f'{group_id}:分析出的語言是：{lang},母語是:{n_lang}')
    for i in lang_dict.get(n_lang):
        if i in lang:
            tr = False
            lang = n_lang
            break
    if translation_stw(message) != message:
        tr = False
        lang = '繁體中文'
    custom_judge = AI_judge.get_judge_test(message)
    if 'ok' in message.lower() and len(message) < 5:
        custom_judge = 'expression_of_gratitude_or_end'

    if custom_judge == 'product_inquiry':
        reply += pi
        if tr:
            reply = AI_judge.translate(n_lang, reply, lang)
        types = 1
    elif custom_judge == 'return_or_exchange_request':
        reply += rt
        if tr:
            reply = AI_judge.translate(n_lang, reply, lang)
        types = 2
    elif custom_judge == 'general_inquiry':
        reply += ge
        if tr:
            reply = AI_judge.translate(n_lang, reply, lang)
        types = 3
    elif custom_judge == 'greeting':
        if status:
            reply += AI_judge.translate('繁體中文', '你好！', n_lang)
        reply += gr
        if "hi" in message.lower() or "hello" in message.lower() or "ok" in message.lower():
            reply = reply + '\n' + AI_judge.translate(n_lang, reply, 'english')
            lang = n_lang
        elif tr:
            reply = AI_judge.translate(n_lang, reply, lang)
        types = 4
    elif custom_judge == 'expression_of_gratitude_or_end':
        reply += end
        if "ok" in message.lower() or "thank" in message.lower():
            reply = reply + '\n' + AI_judge.translate(n_lang, reply, 'english')
            lang = n_lang
        elif tr:
            reply = AI_judge.translate(n_lang, reply, lang)
        types = 5
    else:  # Unable to determine intent or other
        reply += oth
        if tr:
            reply = AI_judge.translate(n_lang, reply, lang)
        types = 6
    print(f'回傳判斷：{custom_judge}')
    print(f'judge判斷時間{time.time()-start}')
    return types, translation_stw(reply), lang

# @app.get("/AI_Search", tags=["AI_Search"])
# def AI_serch(web_id:str='',message:str=''):
#     if web_id == '' or message == '':
#         return {"message": "no sentence input or no web_id", "message": ""}
#     res = _AI_Search.main(web_id,message)
#     return res
#
# @app.get("/AI_Search2", tags=["AI_Search"])
# def AI_serch2(web_id:str='',message:str=''):
#     if web_id == '' or message == '':
#         return {"message": "no sentence input or no web_id", "message": ""}
#     res = _AI_Search.main_sim(web_id, message)
#     return res
