from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import json
import requests
from playwright.async_api import async_playwright
import subprocess
import tempfile
import resource
import os
import asyncio
import time
from src.tools.crawler import WebCrawler, CrawlOptions, CrawlResult
from src.core.basellm import base_llm
from src.core.prompts import compress_web
from config import BOCHA_API_KEY, BOCHA_API_URL

@dataclass
class Tool:
    """工具定义"""
    name: str
    func: Callable
    description: str

##claude写的爬虫版本
def get_text(url):
    llm = base_llm(system_prompt="你是一个优秀的AI助手")
    async def _run():
        async with WebCrawler() as crawler:
            result = await crawler.crawl(url, CrawlOptions(wait_until='networkidle'))
            if result.content:
                if len(result.content) > 3000:
                    prompt = compress_web(result.content)
                    for i in range(3):
                        try:
                            res = llm.call_with_messages_small(prompt)
                            if len(res)<len(result.content):
                                return res
                            else:
                                return result.content
                        except Exception as e:
                            print(f"LLM call failed: {e}, retrying {i}!")
                            time.sleep(5)
                    raise
                else:
                    return result.content
            else:
                return result.content
    return asyncio.get_event_loop().run_until_complete(_run())

def python_interpreter(code,timeout=60, max_memory_mb=500):
    def set_limits():
        # 限制内存为 500MB（yfinance 需要较多内存）
        resource.setrlimit(resource.RLIMIT_AS, 
                          (max_memory_mb * 1024 * 1024*10, 
                           max_memory_mb * 1024 * 1024*10))
        # 限制 CPU 时间
        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
        # 限制可创建的文件大小
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        result = subprocess.run(
            ['python3', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=set_limits,
            cwd=tempfile.gettempdir()
        )
        return "\n=== 输出 ===\n"+result.stdout+"\n=== 警告/错误 ===\n"+result.stderr
    except subprocess.TimeoutExpired:
        return f"错误: 代码执行超过 {timeout} 秒"
    except Exception as e:
        return f"执行错误: {e}"
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
    
def search_web(query: str, url_count = 5) -> str:
    ##博查 API
    url = BOCHA_API_URL

    payload = json.dumps({
      "query": query,
      "summary": True,
      "count": url_count
    })

    headers = {
      'Authorization': f'Bearer {BOCHA_API_KEY}',
      'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    
    if len(response.json()['data']['webPages']['value'])>0:
        res=dict()
        res['content']=''
        res['urls']=list()
        count = 1
        for item in response.json()['data']['webPages']['value']:
            try:
                text = get_text(item['url'])
                if text:
                    res['content']+= "<信息源"+str(count)+">\n"+"网页地址:\n"+item['url']+'\n'+"页面简介:\n"+item['summary']+"\n详细内容:\n"+text+'\n</信息源'+str(count)+">\n\n"
                    res['urls'].append(item['url'])
                else:
                    res['content']+= "<信息源"+str(count)+">\n"+"网页地址:\n"+item['url']+'\n'+item['summary']+'\n</信息源'+str(count)+">\n\n"
                    res['urls'].append(item['url'])
            except Exception as e:
                print(f"exceptions while getting contents.url:{item['url']},exception: {e}")
                res['content']+= "<信息源"+str(count)+">\n"+"网页地址:\n"+item['url']+'\n'+item['summary']+'\n</信息源'+str(count)+">\n\n"
                res['urls'].append(item['url'])
            count+=1
    else:
        res['content']="搜索接口返回信息为空,建议重新搜索。"
    return res
def ask_gpt(query):
    llm = base_llm(system_prompt="请根据你的知识范围，回答用户询问的问题，对于你不确定的知识内容，请拒绝回答或提示用户可能的错误风险。以下用户提出的问题:")
    res=llm.call_with_messages_V3(query,model_name="Qwen/Qwen3-235B-A22B-Instruct-2507")
    return res
    
# 2. 创建工具列表
tools = [
#     Tool(
#         name="Python_interpreter",
#         func=python_interpreter,
#         description="python代码执行器，返回代码执行过程中的打印结果,请在Action Input中直接提供完整可执行的python代码"
#     ),
    Tool(
        name="Search_web",
        func=search_web,
        description="搜索外部网页获取信息,请在Action Input中直接提供搜索关键词，不要其他内容"
    )
#     Tool(
#         name="Ask_gpt",
#         func=ask_gpt,
#         description="询问GPT大模型相关问题，可用于回答知识百科类问题，请在Action Input中直接提供需要询问的问题"
#     )
]