from fastapi import FastAPI
from AI_traffic_assistant import AiTraffic
import pandas as pd
from web_id_similarity import web_id_similarity
from db import DBhelper
from slackwarningletter import slack_warning

description = """
traffic_assistant
author:Yu
---
"""
tags_metadata = [
    {
        "name": "generate_title",
        "description": "The title will be generated based on the article corresponding to the entered keywords",
    },
    {
        "name": "generate_sub_title",
        "description": "Generate subtitles based on the title",
    },
    {
        "name": "generate_articles",
        "description": "Depending on the number of input subtitles, articles of different lengths are generated. no TA",
    },
    {
        "name": "generate_articles_TA",
        "description": "Depending on the number of input subtitles, articles of different lengths are generated.",
    },
    {
        "name": "check",
        "description": "confirm connection",
    },
    {
        "name": "similarity",
        "description": "web_id similarity",
    }
]

app = FastAPI(title="traffic_assistant", description=description, openapi_tags=tags_metadata)
slack_letter = slack_warning()
try:
    AI_traffic = AiTraffic()
    slack_letter.send_letter(f'流量小編生文執行成功 <@U03AD4B5D0C>')
except:
    slack_letter.send_letter(f'流量小編生文生文失敗 <@U03AD4B5D0C>')

@app.get("/title", tags=["generate_title"])
def title(web_id: str = 'test', user_id: str = '', keywords: str = '', web_id_main: str = '', article: str = '', types: int = 1, eng: bool = False, mode='openai'):
    if types != 1 and article =='':
        return '請輸入文章內容'
    res_list = AI_traffic.get_title(web_id=web_id, user_id=user_id, keywords=keywords, web_id_main=web_id_main, article=article, types=types, eng=eng,mode=mode)
    if res_list == 'error':
        sensitive_keyword = AI_traffic.check_sensitive_keyword(keywords)
        error_message = f'【您輸入的關鍵字疑似涉及敏感字眼，建議替換關鍵字後重試】,以下是敏感詞：{sensitive_keyword}' if sensitive_keyword != 'None' else '產生錯誤請在嘗試一次'
        if eng:
            error_message = AI_traffic.translate('繁體中文', error_message, '英文')
        return {'message': error_message, 'code': '100'}
    else:
        return {'message': {i+1: v for i, v in enumerate(res_list)}, 'code': '200'}


@app.get("/sub-heading", tags=["generate_sub_title"])
def subtitle(web_id: str = 'test', user_id: str = '', title: str = '', types: int = 1, eng: bool = False,mode='openai'):
    res_list = AI_traffic.get_sub_title(title, user_id, web_id, types, eng, mode)
    if res_list == 'error':
        return {'message': '可能因為標題過於敏感,造成產生失敗,請輸入不同關鍵字在嘗試一次', 'code': '100'}
    else:
        return {'message': res_list, 'code': '200'}


@app.get("/articles", tags=["generate_articles"])
def articles_api(web_id: str = 'test', user_id: str = '', title: str = '', keywords: str = '', subtitles1: str = '',
                 subtitles2: str = '', subtitles3: str = '', subtitles4: str = '', subtitles5: str = '', types: int = 1, eng: bool = False, mode='openai'):
    res = AI_traffic.generate_articles(title=title, keywords=keywords, user_id=user_id, web_id=web_id, types=types,
                                       subtitle_list=[subtitles1, subtitles2, subtitles3, subtitles4, subtitles5], ta=[], eng=eng, mode=mode)
    if not res:
        sensitive_keyword = AI_traffic.check_sensitive_keyword(keywords)
        error_message = f'【您輸入的關鍵字疑似涉及敏感字眼，建議替換關鍵字後重試】,以下是敏感詞：{sensitive_keyword}' if sensitive_keyword != 'None' else '產生錯誤請在嘗試一次'
        if eng:
            error_message = AI_traffic.translate('繁體中文', error_message, '英文')
        return {'message': error_message, 'code': '100'}
    columns = ['user_id', 'web_id', 'type']
    update_data = [user_id, web_id, types]
    d = 0
    if not subtitles1 and not subtitles2 and not subtitles3 and not subtitles4 and not subtitles5:
        columns.append('article_1')
        update_data.append(res[0])
        DBhelper.ExecuteUpdatebyChunk(pd.DataFrame([update_data],
                                        columns=columns), db='sunscribe', table='ai_article',
                                      chunk_size=100000, is_ssh=False)
        return {'message': res, 'code': '200'}
    for i, v in enumerate([subtitles1, subtitles2, subtitles3, subtitles4, subtitles5]):
        if v:
            columns.append(f"article_{i+1}")
            update_data.append(res[d])
            d += 1
    DBhelper.ExecuteUpdatebyChunk(pd.DataFrame([update_data],
                                               columns=columns), db='sunscribe', table='ai_article',
                                  chunk_size=100000, is_ssh=False)
    return {'message': res, 'code': '200'}

@app.get("/articles_ta_2", tags=["generate_articles_TA"])
def articles_ta(web_id: str = 'test', user_id: str = '', title: str = '', keywords: str = '', subtitles1: str = '',
                subtitles2: str = '', subtitles3: str = '', subtitles4: str = '', subtitles5: str = '', types: int = 1,
                gender: str = '', age: str = '', Income: str = '', style: str = '', interests: str = '', occupation: str = '', eng: bool = False, mode='openai'):
    res = AI_traffic.generate_articles(title=title, keywords=keywords, user_id=user_id, web_id=web_id, types=types,
                                       subtitle_list=[subtitles1, subtitles2, subtitles3, subtitles4, subtitles5],
                                       ta=[gender, age, Income, interests, occupation, style], eng=eng, mode=mode)
    if not res:
        sensitive_keyword = AI_traffic.check_sensitive_keyword(keywords)
        error_message = f'【您輸入的關鍵字疑似涉及敏感字眼，建議替換關鍵字後重試】：{sensitive_keyword}' if sensitive_keyword != 'None' else '產生錯誤請在嘗試一次'
        if eng:
            error_message = AI_traffic.translate('繁體中文', error_message, '英文')
        return {'message': error_message, 'code': '100'}
    columns = ['user_id', 'web_id', 'type']
    update_data = [user_id, web_id, types]
    if not subtitles1 and not subtitles2 and not subtitles3 and not subtitles4 and not subtitles5:
        columns.append('article_1')
        update_data.append(res[0])
        DBhelper.ExecuteUpdatebyChunk(pd.DataFrame([update_data],columns=columns), db='sunscribe', table='ai_article',
                                      chunk_size=100000, is_ssh=False)
        return {'message': res, 'code': '200'}
    d = 0
    for i, v in enumerate([subtitles1, subtitles2, subtitles3, subtitles4, subtitles5]):
        if v:
            columns.append(f"article_{i+1}")
            update_data.append(res[d])
            d += 1
    DBhelper.ExecuteUpdatebyChunk(pd.DataFrame([update_data],
                                               columns=columns), db='sunscribe', table='ai_article',
                                  chunk_size=100000, is_ssh=False)
    return {'message': res, 'code': '200'}

@app.get("/check", tags=["check"])
def checkdef(test: str = ''):
    if test:
        return True

@app.get("/similarity", tags=["similarity"])
def web_id_similarity_func(test: int = 1):
    if test:
        web_id_similarity()
        return 'ok'