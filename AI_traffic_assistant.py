from db import DBhelper
import pandas as pd
import datetime
from tqdm import tqdm
import requests
from utils.log import logger
from AI_customer_service import QA_api
from langchain.output_parsers import RetryWithErrorOutputParser
from opencc import OpenCC
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import SystemMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
import json
import os
import random

class Util(QA_api):
    def __init__(self):
        super().__init__('line', logger())
        self.base_web_id = 'nineyi000360'
        self.base_web_id_type = 1
        self.get_web_id()
        self.dateint = self.get_data_intdate(7)
        self.azure_openai_setting()
        self.langchain_model_setting()
        self.api_version = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_version')
        self.chat_check_model = AzureChatOpenAI(azure_deployment="chat-cs-canada-4", temperature=0, openai_api_version=self.api_version)
        self.article_model = AzureChatOpenAI(temperature=0.2, azure_deployment='chat-cs-canada-35', openai_api_version=self.api_version)
        self.article_model_16k = AzureChatOpenAI(temperature=0.2, azure_deployment='chat-cs-canada-35-16k', openai_api_version=self.api_version)
        self.article_4_model = AzureChatOpenAI(temperature=0.2, azure_deployment='chat-cs-canada-4', openai_api_version=self.api_version)

    def get_data_intdate(self, time_delay):
        return int(str(datetime.date.today() - datetime.timedelta(time_delay)).replace('-', ''))

    def get_web_id(self):
        qurey = f"""SELECT web_id,web_id_type,cx FROM missoner_web_id_table WHERE ai_article_enable = 1 """
        web_id_list = DBhelper('dione', is_ssh=True).ExecuteSelect(qurey)
        self.web_id_dict = {i: v for i, v, g in web_id_list}
        self.web_id_cx = {i: g for i, v, g in web_id_list}
        return

    def get_media_keyword_data(self):
        query = f"""
        SELECT k.keyword,k.web_id,k.url,k.image,t.title ,t.content,t.pageviews 
        FROM (SELECT web_id,article_id ,url, keyword,image 
                FROM missoner_keyword_article_new 
                WHERE dateint > '{self.dateint}' and web_id in ("{'","'.join([i for i, v in self.web_id_dict.items() if v == 0])}")) k 
        INNER JOIN (SELECT article_id, pageviews,title,content 
                FROM dione.missoner_article 
                WHERE `date` > {self.dateint} ) t 
        ON k.article_id = t.article_id ORDER BY t.pageviews desc;
        """
        data = DBhelper('dione', is_ssh=True).ExecuteSelect(query)
        return pd.DataFrame(data).drop_duplicates(['keyword', 'title'])

    def get_keyword_data(self, web_id):
        query = f"""
        SELECT k.keyword,k.web_id,k.url,k.image,t.title ,t.content,t.pageviews 
        FROM (SELECT web_id, url, article_id, keyword,image 
                FROM missoner_keyword_article_new 
                WHERE dateint > {self.dateint} and web_id = '{web_id}') k 
        INNER JOIN (SELECT article_id, pageviews,title,content 
                FROM dione.missoner_article
                WHERE
        """
        if self.web_id_dict.get(web_id) != 1:
            query += f""" `date` > {self.dateint} and"""
        query += f""" web_id = '{web_id}') t 
                    ON k.article_id = t.article_id ORDER BY t.pageviews desc;"""
        data = DBhelper('dione', is_ssh=True).ExecuteSelect(query)
        return pd.DataFrame(data).drop_duplicates(['keyword', 'title'])

    def langchain_model_setting(self):
        self.check_model_setting()
        self.title_model_setting()

    def check_model_setting(self):
        response_schemas = [ResponseSchema(name="check", description="判斷是否符合大眾觀賞"), ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        self.check_prompt = ChatPromptTemplate(
            messages=[SystemMessage(
                content=("你會判斷內容是否符合大眾觀賞,返回True或者False")),
                HumanMessagePromptTemplate.from_template(
                    "answer the users content as best as possible.\n{format_instructions}\n{question}")
            ],
            input_variables=["question"],
            partial_variables={"format_instructions": format_instructions}
        )

    def title_model_setting(self):
        self.title_1_prompt = """
                                You are a helpful AI designed to output JSON-formatted responses. Given a set of keywords and their descriptions, your task is to generate a creative title that includes at least one of the provided keywords. The title must not contain any references to news websites.
                                {
                                  "title": "Your Generated Title Including the Keyword"
                                }
                                Please generate a title and output the response in JSON format."""
        self.title_5_prompt = """
                                You are a helpful AI designed to output JSON-formatted responses. Given a set of keywords and their descriptions, your task is to generate five creative titles that includes at least one of the provided keywords. The title must not contain any references to news websites. Please respond in language entered. The JSON output should follow this format:
                                {
                                  "title_1": "Your Generated Title Including the Keyword"
                                  "title_2": "Your Generated Title Including the Keyword"
                                  "title_3": "Your Generated Title Including the Keyword"
                                  "title_4": "Your Generated Title Including the Keyword"
                                  "title_5": "Your Generated Title Including the Keyword"
                                }
                                Please generate a title and output the response in JSON format.        
                                """
        self.sub_title_prompt = """
                                You will act as a JSON generator. I will give you a title, and you are to create five sub-titles related to the given title. Please format your response in JSON, with each sub-title as a separate, numbered field. Here is the structure I need:
                                {
                                    "sub_title_1": "Your Generated Sub-Title 1",
                                    "sub_title_2": "Your Generated Sub-Title 2",
                                    "sub_title_3": "Your Generated Sub-Title 3",
                                    "sub_title_4": "Your Generated Sub-Title 4",
                                    "sub_title_5": "Your Generated Sub-Title 5"
                                }
                                """
        self.translation_prompt = """You are a helpful assistant designed to output JSON. Translate the provided text into the specified target language and format the translation as a JSON object, including fields for the target text, and target language.  
                                    Example:

                                    Input:
                                    Source Text: "Hello, how are you?"
                                    Source Language: English
                                    Target Language: Spanish

                                    Output:
                                    {
                                      "target_text": "Hola, ¿cómo estás?",
                                      "target_language": "Spanish"
                                    }
                                    """
    def azure_openai_setting(self):
        os.environ["AZURE_OPENAI_API_KEY"] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_key')
        os.environ["AZURE_OPENAI_ENDPOINT"] = self.ChatGPT.AZURE_OPENAI_CONFIG.get('api_base')


    def translate(self, lang, out, n_lang='繁體中文'):
        if lang != n_lang:
            print(f"進行翻譯{lang} to {n_lang}")
            m = f"""Source Text: {out},
                Source Language: {lang},
                Target Language: {n_lang}"""
            out = self.ask_gpt([{'role': 'system', 'content': self.translation_prompt},
                                {'role': 'user', 'content': m}], json_format=True, timeout=120)
            json_string = out.replace("'", '"')
            json_data = json.loads(json_string)
            return json_data.get('target_text')
        return out

    def translation_stw(self, text):
        cc = OpenCC('likr-s2twp')
        return cc.convert(text)

    def get_article(self, prompt, title, sub_list):
        if not sub_list:
            response_schemas = [ResponseSchema(name=f"Articles", description=f"Articles")]
            model = self.article_4_model
        else:
            response_schemas = [ResponseSchema(name=f"paragraph_{i + 1}", description=f"Articles with subtitle '{v}'")
                                for i, v in enumerate(sub_list)]
            model = self.article_model
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        _input = ChatPromptTemplate(
            messages=[HumanMessagePromptTemplate.from_template("{question}\n{format_instructions}\n")],
            input_variables=["question"], partial_variables={"format_instructions": format_instructions}).format_prompt(
            question=prompt)
        output = model(_input.to_messages())
        # noinspection PyBroadException
        try:
            gpt_res = output_parser.parse(output.content)
            if not sub_list:
                res = [gpt_res['Articles'].replace('文章標題：', '').replace('文章標題:', '').replace('文章內容：', '').replace('文章內容:', '')]
            else:
                res = [v.replace(f'{i}。', '').replace(f'{i}', '') for i, v in zip(sub_list, gpt_res.values())]
            for i in res:
                if not i:
                    raise
        except:
            k = 0
            while True:
                print('產生失敗！！,重新產生')
                if k > 5:
                    print("產生超失敗")
                    return []
                retry_parser = RetryWithErrorOutputParser.from_llm(parser=output_parser, llm=self.article_model_16k)
                gpt_res = retry_parser.parse_with_prompt(output.content, _input)
                if not sub_list:
                    res = [gpt_res['Articles'].replace('文章標題：', '').replace('文章標題:', '').replace('文章內容：', '').replace('文章內容:', '')]
                else:
                    res = [v.replace(f'{i}。', '').replace(f'{i}', '') for i, v in zip(sub_list, gpt_res.values())]
                for i in res:
                    if not i:
                        k += 1
                        continue
                break
        return res

    def get_generate_articles_prompt(self, title, sub_list, keyword_info_dict, ta_setting, eng=False):
        if not ta_setting:
            print(f"產生無TA文章,段落：{len(sub_list)}")
            style = 'easy'
        else:
            print(f"產生有TA文章,段落：{len(sub_list)}")
            gender, age, income, interests, occupation, style = ta_setting
        if not sub_list:
            prompt = f"""Please use the following provided information to craft the concise article of approximately 100 words,including the title, subtitles, keywords, keyword explanations,Target Audienceand,and other relevant details. The article should be written in an {style} style """
            if not eng:
                prompt += "and in Taiwan Mandarin"
            prompt += f""". Ensure that proper grammar and sentence structure are maintained throughout.

            Title: "{title}"
            """
        else:
            #prompt = f"""Please use the following provided information to craft concise article of approximately {str(len(sub_list)*2)}00 words, including the title, subtitles, keywords, keyword explanations,Target Audience, and other relevant details. The article should be written in an {style} style and in Taiwan Mandarin. Ensure that proper grammar and sentence structure are maintained throughout.Please ensure that the subtitles are not altered in any way."""
            prompt = f"""Please use the following provided information to craft concise article of approximately {str(len(sub_list)*2)}00 words, including the title, subtitles, keywords, keyword explanations,Target Audience, and other relevant details. The article should be written in an {style} style """
            if not eng:
                prompt += "and in Taiwan Mandarin"
            prompt += f""". Ensure that proper grammar and sentence structure are maintained throughout.Please ensure that the subtitles are not altered in any way.

            Title: "{title}"

            Subtitles:
            """
            prompt += '     '.join([f'{str(i + 1)}. "{v}"\n' for i, v in enumerate(sub_list)])
        prompt += """
        Keywords and Explanations:
        """
        prompt += '     '.join(
            [f"""{i + 1}."{data[0]}": "{data[1][0]}"\n""" for i, data in enumerate(keyword_info_dict.items())])
        if ta_setting:
            prompt += f"""
            Target Audience information and Interests:
            - Gender: "{gender}"
            - Age: "{age}"
            - Income bracket: "{income}"
            - Interests: "{interests}"
            - Occupation: "{occupation}"
            """
        else:
            prompt = prompt.replace(",TargetAudience", '')
        prompt += "Please focus on incorporating each keyword in the article in a manner that reflects a clear understanding of its meaning while considering the target audience provided. Be creative, and ensure that each subtitle has its own section in the article.Please ensure that the subtitles are not altered in any way. "
        if not ta_setting:
            prompt = prompt.replace(" while considering the target audience provided", '')
        if not sub_list:
            prompt += "and there should be no explanations within the responses only response article.don't add any subtitles"
        print(f"""PROMPT:{prompt}""")
        return prompt


class AiTraffic(Util):
    def __init__(self):
        super().__init__()
        self.media_keyword_pd = self.get_media_keyword_data()
        self.keyword_all_set = set(self.media_keyword_pd.keyword)
        self.all_keyword_pd = {}
        self.get_keyword_pd()

    def get_keyword_info(self, web_id, keyword):
        print(f"""獲取{web_id}的關鍵字資訊""")
        if web_id not in self.web_id_dict or web_id == 'pure17' or web_id not in self.all_keyword_pd:
            print(f"""不包含{web_id}的關鍵字資訊,更改為base_web_id:{self.base_web_id}""")
            web_id = self.base_web_id
        keyword_list = keyword.split(',')
        keyword_list = [i for i in keyword_list if i]
        cx = self.web_id_cx[web_id]
        df = self.all_keyword_pd[web_id]
        keyword_info_dict = {}
        visited = []
        print(keyword_list)
        for key in keyword_list:
            if len(df) and key in set(df.keyword):
                curr_keyword_info = df[df.keyword == key]
                best_match = None
                for title, content, article_id, img in curr_keyword_info[['title', 'content', 'url', 'image']].values:
                    if any(len(set(other_key) & set(title))/len(set(other_key)) > 0.8 for other_key in keyword_list if other_key != key):
                        best_match = (title, content, web_id, article_id, img)
                        print(f'找到標題包含其他關鍵字的：{key} -> {title}')
                        break  # 一旦找到包含其他關鍵字的標題就停止

                # 如果沒有找到包含其他關鍵字的標題，則使用第一個條目
                if not best_match:
                    title, content, article_id, img = curr_keyword_info.iloc[0][['title', 'content', 'url', 'image']].values
                    best_match = (title, content, web_id, article_id, img)
                    print(f'未找到其他關鍵字,使用內站關鍵字：{key} -> {title}')
                keyword_info_dict[key] = best_match
            elif key in self.keyword_all_set:
                curr_keyword_info = self.media_keyword_pd[self.media_keyword_pd.keyword == key]
                best_match = None
                for title, content, web_id_article, article_id, img in curr_keyword_info[['title', 'content', 'web_id', 'url', 'image']].values:
                    if any(len(set(other_key) & set(title))/len(set(other_key)) > 0.8 for other_key in keyword_list if other_key != key):
                        best_match = (title, content, web_id_article, article_id, img)
                        print(f'找到標題包含其他的外站關鍵字：{key} -> {title}')
                        break  # 一旦找到包含其他關鍵字的標題就停止

                # 如果沒有找到包含其他關鍵字的標題，則使用第一個條目
                if not best_match:
                    title, content, web_id_article, article_id, img = curr_keyword_info.iloc[0][['title', 'content', 'web_id', 'url', 'image']].values
                    best_match = (title, content, web_id, article_id, img)
                    print(f'未找到標題包含其他的外站關鍵字：{key} -> {title}')
                keyword_info_dict[key] = best_match
            else:
                if len(keyword_list) > 2:
                    t = 0
                    while True:
                        if t == 10:
                            search_key = key
                            break
                        r = random.choice(keyword_list)
                        if set([key, r]) in visited or r == key:
                            t += 1
                            continue
                        visited.append(set([key, r]))
                        search_key = f'{key}+{r}'
                        break
                else:
                    search_key = key

                html1 = f'https://www.googleapis.com/customsearch/v1/siterestrict?cx={cx}&key={self.Search.GOOGLE_SEARCH_KEY}&q={key}'
                html2 = f'https://www.googleapis.com/customsearch/v1/siterestrict?cx=41d4033f0c2f04bb8&key={self.Search.GOOGLE_SEARCH_KEY}&q={search_key}'
                for i, html in enumerate([html1, html2]):
                    r = requests.get(html, timeout=5)
                    if r.status_code != 200:
                        continue
                    res = r.json().get('items')
                    if not res:
                        continue
                    for data in res:
                        title = data.get('htmlTitle')
                        sn = data.get('snippet')
                        link = data.get('link')

                        # 檢查標題或摘要中是否包含其他關鍵字
                        if key in title:
                            best_match = (title, sn, 'google', link, '_')
                            print(f'google找到標題包含關鍵字的：{key} -> {title}')
                            keyword_info_dict[key] = best_match
                            break  # 一旦找到符合條件的項目就停止

                    if key in keyword_info_dict:
                        if i == 0:
                            print(f'google搜尋主要關鍵字{key}')
                        if i == 1:
                            print(f'google搜尋主要關鍵字:{key},組合關鍵字:{search_key}')
                        break
                if key not in keyword_info_dict:
                    print(f'無關鍵字：{key}資訊')
                    keyword_info_dict[key] = ('_', '_', 'none', '_', '_')

        print(f"""關鍵字資訊:{keyword_info_dict}""")
        return keyword_info_dict

    def get_keyword_pd(self):
        pbar = tqdm(list(self.web_id_dict.keys()))
        #pbar = tqdm(['innolife'])
        for i, web_id in enumerate(pbar):
            pbar.set_description(web_id)
            self.all_keyword_pd[web_id] = self.get_keyword_data(web_id)

    def check_sensitive_keyword(self, keywords):
        result = self.ChatGPT.ask_gpt(model='gpt-4o', message=[{'role': 'system', 'content': """你是關健字敏感詞判斷機器人,輸入關鍵字,請回傳敏感字,如果沒有回傳None,json格式回傳,                                {
                                                                                "Sensitive_keyword":"Sensitive_keyword1,Sensitive_keyword2..." """},
                                               {'role': 'user', 'content': keywords}], json_format=True,azure=False)
        json_string = result.replace("'", '"')
        json_data = json.loads(json_string)
        return json_data.get("Sensitive_keyword")




    def check_news(self, text):
        _input = self.check_prompt.format_prompt(question=text)
        output = self.chat_check_model(_input.to_messages())
        if 'False' in output.content:
            return False
        return True

    def check_keyword(self, keywords, web_id):
        print('檢查關鍵字是否包含')
        if web_id not in self.web_id_dict:
            web_id = self.base_web_id
        df = self.all_keyword_pd[web_id]
        keywords_set = set(df.keyword)
        keyword_list = keywords.split(',')
        for key in keyword_list:
            if key in self.keyword_all_set or key in keywords_set:
                print(f'關鍵字：{key}')
                return True
        return False

    def get_title(self, web_id: str = 'test', user_id: str = '', keywords: str = '', web_id_main: str = '',
                  article: str = None, types: int = 1, eng: bool = False):
        print(f"""輸入web_id:{web_id}""")
        # check_keyword = self.check_keyword(keywords, web_id_main) if web_id_main else self.check_keyword(keywords, web_id)
        # if not check_keyword:
        #     pass
        keyword_info_dict = self.get_keyword_info(web_id_main, keywords) if web_id_main else self.get_keyword_info(
            web_id, keywords)
        prompt = ''.join([f""" "keyword":{i}\n "description":{v}\n\n""" for i, v in keyword_info_dict.items()])
        if types == 1:
            sys_prompt = self.title_1_prompt + """\nPlease respond in the language of the entered keywords.ex:"keyword":สวัสดี ,respond is "Thai" ,The JSON output should follow this format""" if eng else self.title_1_prompt + "\nPlease respond in traditional Chinese. The JSON output should follow this format"
            k = 0
            while True:
                try:

                    result = self.ChatGPT.ask_gpt(message=[{'role': 'system', 'content': sys_prompt},
                                                  {'role': 'user', 'content': f'{prompt}'}],model='gpt-4o' ,json_format=True)
                    title = eval(result).get('title')
                    if web_id == 'voux' and '預約試穿' in title:
                        raise
                    break
                except:
                    if k == 10:
                        title = '關鍵字可能包含敏感字詞,請返回上一頁修改並重新產生！'
                        return 'error'

                    k += 1
            title = self.translation_stw(title) if not eng else title
            DBhelper.ExecuteUpdatebyChunk(pd.DataFrame([[user_id, web_id, types, keywords, title, json.dumps(
                [{'keyword': keyword, 'title': data[0], 'web_id': data[2], 'url': data[3], 'image': data[4]} for
                 keyword, data in keyword_info_dict.items()]), datetime.datetime.now()]],
                                                       columns=['user_id', 'web_id', 'type', 'inputs', 'title',
                                                                'keyword_dict', 'add_time']), db='sunscribe',
                                          table='ai_article', chunk_size=100000, is_ssh=False)
            return [title]
        else:
            sys_prompt = self.title_5_prompt + "\nPlease respond in language entered. The JSON output should follow this format" if eng else self.title_5_prompt + "\nPlease respond in traditional Chinese. The JSON output should follow this format"
            if not article:
                return {"message": "no article input"}
            k = 0
            while True:
                try:
                    result = self.ChatGPT.ask_gpt(message=[{'role': 'system', 'content': sys_prompt},
                                                           {'role': 'user', 'content': f'{prompt}'}], json_format=True)
                    title = eval(result)
                    break
                except:
                    if k == 10:
                        title = {'title_1': '關鍵字可能包含敏感字詞,請返回上一頁修改並重新產生！',
                                 'title_2': '關鍵字可能包含敏感字詞,請返回上一頁修改並重新產生！',
                                 'title_3': '關鍵字可能包含敏感字詞,請返回上一頁修改並重新產生！',
                                 'title_4': '關鍵字可能包含敏感字詞,請返回上一頁修改並重新產生！',
                                 'title_5': '關鍵字可能包含敏感字詞,請返回上一頁修改並重新產生！'}
                        return 'error'
                    k += 1
            DBhelper.ExecuteUpdatebyChunk(pd.DataFrame([[user_id, web_id, types, keywords,
                                                         self.translation_stw(title['title_1']),
                                                         self.translation_stw(title['title_2']),
                                                         self.translation_stw(title['title_3']),
                                                         self.translation_stw(title['title_4']),
                                                         self.translation_stw(title['title_5']), article, 2, datetime.datetime.now()]],
                                                       columns=['user_id', 'web_id', 'type', 'inputs', 'subheading_1',
                                                                'subheading_2', 'subheading_3', 'subheading_4',
                                                                'subheading_5', 'article_1', 'article_step', 'add_time']),
                                          db='sunscribe', table='ai_article', chunk_size=100000, is_ssh=False)
            return list(title.values())

    def get_sub_title(self, title: str = '', user_id: str = '', web_id: str = 'test', types: int = 1, eng: bool = False):
        print(f"""獲取副標題,標題為:{title}""")
        k = 0
        sub_title_prompt = self.sub_title_prompt + "\nPlease respond in language entered. The JSON output should follow this format" if eng else self.sub_title_prompt + "\nPlease respond in traditional Chinese. The JSON output should follow this format"
        while True:
            try:
                result = self.ChatGPT.ask_gpt(message=[{'role': 'system', 'content': sub_title_prompt},
                                                       {'role': 'user', 'content': f'{title}'}], json_format=True)
                sub_title_dict = eval(result)
                if not sub_title_dict.get('sub_title_1'):
                    raise
                break
            except:
                if k == 10:
                    return 'error'
                    break
                k += 1
        print(f"""副標題產生成功,副標題為:{sub_title_dict}""")
        DBhelper.ExecuteUpdatebyChunk(pd.DataFrame([[user_id, web_id, types, title, sub_title_dict.get('sub_title_1'),
                                                     sub_title_dict.get('sub_title_2'), sub_title_dict.get('sub_title_3'),
                                                     sub_title_dict.get('sub_title_4'), sub_title_dict.get('sub_title_5')]],
                                                   columns=['user_id', 'web_id', 'type', 'title', 'subheading_1',
                                                            'subheading_2', 'subheading_3', 'subheading_4',
                                                            'subheading_5']), db='sunscribe', table='ai_article',
                                      chunk_size=100000, is_ssh=False)
        return {a+1: b for a, b in enumerate(sub_title_dict.values())}

    def generate_articles(self, title: str = '', subtitle_list: list = [], keywords: str = '', user_id: str = '',
                          web_id: str = 'test', types: int = 1, ta: list = [], eng: bool = False):
        query = f"SELECT keyword_dict  FROM web_push.ai_article WHERE web_id = '{web_id}' and user_id  ='{user_id}'"
        keyword_info_db = DBhelper('sunscribe').ExecuteSelect(query)
        sub_list = [i for i in subtitle_list if i]
        if keyword_info_db:
            print('db有keyword_info')
            keyword_info_dict = {i['keyword']: (i['title'], i['web_id'], i['url']) for i in
                                 json.loads(keyword_info_db[0][0])}
        else:
            print('db無keyword_info,可能有問題')
            keyword_info_dict = self.get_keyword_info(web_id, keywords)
        prompt = self.get_generate_articles_prompt(title, sub_list, keyword_info_dict, ta, eng)
        res = self.get_article(prompt, title, sub_list)
        print(f"""產生的文章內容:\n{res}""")
        return res


if __name__ == " __main__":
    AI_traffic = AiTraffic()
