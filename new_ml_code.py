import pandas as pd
import pandas as np
import time

import re
import requests
from bs4 import BeautifulSoup

from scipy.spatial.distance import cosine

import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration, BartConfig
from sentence_transformers import SentenceTransformer



def news_crwaling(search):
    headers = {'user-agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"}

    # 검색어 설정
    search_query = search

    # 검색 결과 페이지의 URL입니다.
    base_url = 'https://search.naver.com/search.naver'
    url = f"https://search.naver.com/search.naver?query={search_query}"


    # HTTP GET 요청을 보내고 응답을 받습니다.
    news_url = ''
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # 네이버에 원하는 주식명을 검색한 화면에서 증권정보 -> 관련뉴스 탭의 url을 가져옵니다.
    links = soup.find_all("a", href=True)

    for link in links:
        url = link['href']
        if "item/news" in url:   # item/news을 포함한 url은 "관련뉴스"탭의 url을 의미합니다.
            if re.match(r'(http|https)://', url):
                news_url = url

    # url 중 주식 코드만 가져옵니다.
    code = news_url[-6:]

    # 주식 코드 추출 완료

    ## 주식 시가 정보 가져오기

    url = 'https://finance.naver.com/item/sise_day.naver?code=' + code

    # HTTP GET 요청을 보내고 응답을 받습니다.
    html = requests.get(url, headers = {'User-agent' : 'Mozilla/5.0'}).text
    soup = BeautifulSoup(html, "lxml")

    df = None

    req = requests.get(f'{url}&page=1', headers = {'User-agent' : 'Mozilla/5.0'})
    df = pd.concat([df, pd.read_html(req.text, encoding = 'euc-kr')[0]], ignore_index = True)
    df.dropna(inplace = True)
    df.reset_index(drop = True, inplace = True)



    ##  특정 검색어에 해당하는 관련 뉴스들의 url 가져오기

    # 네이버증권 -> 검색어 검색 -> 뉴스 공시에 해당 하는 웹 페이지로 이동
    url = "https://finance.naver.com/item/news_news.nhn?code=" + code

    # HTTP GET 요청을 보내고 응답을 받습니다.
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")

    links =[]
    for link in soup.find_all("a"):
        url = link.get("href")
        if "article" in url:   # article을 포함한 url만 수집, 뉴스 기사들에 해당
            links.append(url)
    link_result = []

    # 수집된 url에 http를 추가하여 저장
    for link in links: 
        add = 'https://finance.naver.com' + link
        link_result.append(add)
    
    ## 기사 제목 추출

    titles = []
    all_td = soup.findAll('td', {'class' : 'title'})

    for td in all_td:
        titles.append(td.find('a').get_text())

    # link,title 추출 완료 
    ## 뉴스 내용 추출

    article_result = []

    for link_news in link_result:

        article_source_code = requests.get(link_news).text
        article_html = BeautifulSoup(article_source_code, "lxml")
  
    # 뉴스 내용

        article_contents = article_html.select('.scr01')
        article_contents=article_contents[0].get_text()
        article_contents = re.sub('\n','',article_contents)
        article_contents = re.sub('\t','',article_contents)

        # cut extra text after Copyright mark
        if "ⓒ" in article_contents:
            article_contents=article_contents[:article_contents.index("ⓒ")]

        article_result.append(article_contents)
 
    return titles, article_result, df


def sum_model(news_cnt_sr):

    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')

    # 모델 구성을 위한 config 로드
    config = BartConfig.from_json_file("config.json")

    # 사전 학습된 가중치를 포함한 모델 로드
    model = BartForConditionalGeneration.from_pretrained("pytorch_model.bin", config=config)

    news_cnt_sr = news_cnt_sr
    news_sum = []
    summary = ''

    for news_cnt in news_cnt_sr:
        text = news_cnt

        text = text.replace('\n', ' ')

        raw_input_ids = tokenizer.encode(text)
        input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

        try:
            summary_ids = model.generate(torch.tensor([input_ids]),  num_beams=4,  max_length=512,  eos_token_id=1)
            summary = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
            if len(summary) < 200: # 너무 긴 summary는 압축되지 못하는 parsing이 잘못되는 경우가 대부분이다.
                news_sum.append(summary)
            else:
                news_sum.append('')
        except:
            news_sum.append('')
    
    
    return news_sum

def cossim_model(news_title, news_sum):
    news_title = news_title
    news_sum = news_sum
    cos_sim = []

    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    # 문장 입력
    for i in range(len(news_title)):
        sentence1 = news_title[i]
        sentence2 = news_sum[i]

        # 두 문장을 벡터로 변환
        embeddings1 = model.encode(sentence1, convert_to_tensor=True)
        embeddings2 = model.encode(sentence2, convert_to_tensor=True)

        # 두 벡터 간 코사인 유사도를 계산
        cosine_similarity = 1 - cosine(embeddings1, embeddings2)

        # 유사도가 0.5 이상만 추출 >> 본문과 요약문이 동떨어진 기사 제거
        if cosine_similarity >= 0.5:
            cos_sim.append(cosine_similarity)
        else: cos_sim.append('')
    return cos_sim

def main(search):
    search = search
    news_title, news_cnt_sr, stock_info =  news_crwaling(search)
    news_sum = sum_model(news_cnt_sr)
    cos_sim = cossim_model(news_title, news_sum)
    
    df = []
    df = pd.DataFrame({'Title':news_title, 'Content':news_sum, 'Similiarity':cos_sim})
    df = df[~(df == '').any(axis=1)]
    # 요약시 발생하는 dummy 값이다. 따라서 제거해준다.
    df = df[~(df == '한국토지자원관리공단은 한국토지공사의 지분참여를 통해 일자리 창출을 도모할 예정이다.').any(axis=1)].reset_index(drop=True)
    df = df[~(df == '소 소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소소').any(axis=1)].reset_index(drop=True)
    # summary = dict(zip(df['Title'].values, df['Content'].values))
    summary = {i : string for i,string in enumerate(df.Content.values)}
    stock = {i : list(string) for i,string in enumerate(stock_info.values)}

    return summary, stock

# def main(search):
#     summary = {
#         0: '앨런 에스테베스 미 상무부 차관이 지난 23일 워싱턴DC에서 열린 한미 경제안보포럼에 참석, 대중국 반도체 장비 수출 규제 관련 계획을 밝히며 중국 반도체 장비 수출 규제 관련 압박 강도를 높이고 있다.',
#         1: '앨런 에스테베스 상무부 산업안보 차관은 23일 한국국제교류재단(KF)과 전략국제문제연구소(CSIS)가 워싱턴DC에서 개최한 한미 경제안보포럼에서 삼성과 SK에 제공한 대중(對中) 반도체 수출통제 1년 유예가 끝난 후 조치에 대해 기업들이 생산할 수 있는 반도체 수준에 한도를 둘 가능성이 크다고 말했다.',
#         2: '미국 정부가 자국 내 생산을 유인하기 위해 반도체 기업에 390억달러(약 50조원)에 달하는 막대한 보조금을 지급하는데, 삼성전자와 SK하이닉스 등 국내 반도체업체들의 고민이 깊어지고 있다.',
#         3: '미국이 28일부터 약 50조원 규모의 반도체 보조금 지원 신청을 받는 가운데, 미국 정부는 중국에 신규 투자를 하지 않는 기업에만 보조금을 주겠다는 입장을 고수하고 있어 삼성전자·SK하이닉스의 고민이 깊어지고 있다.',
#         4: '미국 정부가 자국 내 반도체 생산 장려를 위해 기업에 총 390억달러(약 50조원)를 지급하는 보조금 신청을 다음주부터 받는다.',
#         5: '삼성전자 시스템LSI사업부가 퀄컴과 대만 미디어텍의 입지가 커지고 있는 모바일 AP 시장에서 프리미엄 모델에 준하는 성능을 갖춘 엑시노스 1380·1330을 공개했다.',
#         6: '삼성자산운용의 ‘삼성 밀당 다람쥐 글로벌 EMP 펀드’는 국내외 상장지수펀드(ETF)를 활용해 시장 상황에 따라 탄력적으로 주식과 채권의 비중을 조정하며 투자자들에게 안정적인 현금 흐름을 안겨준다.',
#         7: '지난 16일 한국, 미국, 일본, 대만 정부의 국장급 실무자들이 화상으로 모여 칩4 관련 본회의를 열고 각국의 반도체 공급망 상황을 공유한 가운데 우리나라 정부가 미·중 기술 패권 전쟁 속 곤경에 처한 국내 반도체기업의 입장을 적극 피력할지 이목이 집중되고 있다.',
#         8: "24일(현지시간) 독일 프랑크푸르트에서 삼성전자는 '2023 유럽 테크세미나'를 개최하고 삼성 OLED를 비롯한 신제품을 알린다고 밝혔다.",
#         9: "삼성전전자는 23~24일 독일 프랑크푸르트에서 '2023 유럽 테크세미나'를 열어 2023년형 Neo QLED와 OLED 등 TV 신제품의 기술력을 알린다.",
#         10: "삼성전자는 23일과 24일(현지시간) 양일간 독일 프랑크푸르트에서 '2023 유럽 테크세미나'를 개최하고 2023년형 Neo QLED와 OLED 등 TV 신제품의 기술력을 알린다고 밝혔다.",
#         11: '22일 코스피 코스피지수는 전 거래일 대비 6.06포인트(0.25%) 내린 2445.15를 기록하며 하방 압력을 받으며 2440대에서 오르내리고 있으며 코스닥시장에서는 개인 홀로 1231억원을 순매수한 반면, 외국인과 기관은 각각 822억원과 263억원을 순매도하고 있다.',
#         12: "삼성전자가 23 프랑크푸르트에서 세계 주요 지역 영상·음향 전문가에게 삼성 TV의 최신 기술과 서비스를 소개하고 다양한 의견을 청취하는 '2023 유럽 테크세미나'를 개최한다고 밝혔다.",
#         13: "삼성전자가 오는 프랑크푸르트에서 '2023 유럽 테크세미나'를 열어 최신 TV 기술력을 대거 선보였으며 4월에는 호주와 동남아, 중남미 등 주요 지역에서 순차적으로 테크세미나를 열 계획이다.",
#         14: '삼성전자가 23~24일 독일 프랑크푸르트에서 ‘2023 유럽 테크세미나’를 개최하고 2023년형 ‘Neo QLED’ TV 등 신제품을 알리며 오는 4월 호주와 동남아, 중남미 등 주요 지역에서 순차적으로 테크세미나를 열 계획이다.'
#     }
#     stock = {
#         0: ['2023.02.24', 61300.0, 700.0, 62300.0, 62600.0, 61300.0, 10134614.0],
#         1: ['2023.02.23', 62000.0, 900.0, 61700.0, 62500.0, 61500.0, 13047099.0],
#         2: ['2023.02.22', 61100.0, 1000.0, 61500.0, 61800.0, 61000.0, 11959088.0],
#         3: ['2023.02.21', 62100.0, 600.0, 62700.0, 62800.0, 62000.0, 7665046.0],
#         4: ['2023.02.20', 62700.0, 100.0, 62900.0, 63000.0, 61800.0, 12908073.0],
#         5: ['2023.02.17', 62600.0, 1100.0, 62900.0, 63300.0, 62400.0, 10791265.0],
#         6: ['2023.02.16', 63700.0, 1500.0, 62500.0, 63700.0, 62400.0, 13798831.0],
#         7: ['2023.02.15', 62200.0, 1000.0, 63900.0, 63900.0, 62000.0, 13208103.0],
#         8: ['2023.02.14', 63200.0, 300.0, 63600.0, 63900.0, 63200.0, 9126664.0],
#         9: ['2023.02.13', 62900.0, 100.0, 62900.0, 63000.0, 62300.0, 10730362.0]
#     }

#     return summary, stock

