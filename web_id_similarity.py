from db import DBhelper
import pandas as pd
import collections
import datetime

def web_id_similarity():
    date = int((datetime.datetime.utcnow() - datetime.timedelta(days=30)).strftime("%Y%m%d"))
    query = f"""SELECT DISTINCT web_id,tag from all_website_category_tag_2"""
    tag_web_id = DBhelper('sunscribe').ExecuteSelect(query=query)
    query = f"""SELECT DISTINCT web_id FROM dione.pageview_record_day x WHERE `date` >= {date}"""
    web_id_online = {i[0] for i in  DBhelper('dione', is_ssh=True).ExecuteSelect(query=query)}


    tag_dict = collections.defaultdict(set)
    for web_id, tag in tag_web_id:
        if web_id in web_id_online:
            tag_dict[web_id].add(tag)
    ans = collections.defaultdict(list)
    for i, v in tag_dict.items():
        for j, vv in tag_dict.items():
            a = v & vv
            b = v | vv
            if len(a) == 0:
                continue
            ans[i].append((len(a) / (len(v) * len(vv)) ** 0.5, (len(a) / len(b)), -len(vv), j))
        ans[i] = sorted(ans[i], reverse=True)[:50]
    df = pd.DataFrame(columns=['web_id', 'similarity_web_id', 'score1', 'score2', 'sim_web_id_tag_count', 'ranks'])
    for i, v in ans.items():
        for j, g in enumerate(v):
            df.loc[len(df)] = [i, g[3], g[0], g[1], -g[2], j + 1]
    table_name = f"all_website_category_similarity_rank"
    query = f"TRUNCATE TABLE {table_name}"
    print('刪除舊資料')
    DBhelper('sunscribe').ExecuteSelect(query)
    print('新增新資料')
    DBhelper.ExecuteUpdatebyChunk(df, db='sunscribe', table='all_website_category_similarity_rank',chunk_size=100000, is_ssh=False)

    return df
if __name__ == "__main__":
    k = web_id_similarity()
