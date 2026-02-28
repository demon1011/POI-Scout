from src.core.prompts import search_plan_prompt,search_opt_plan_prompt,agent_match_prompt
import json
import re
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy


def extract_json_from_response(response: str) -> str:
    """
    从 LLM 响应中提取 JSON 内容，清理 markdown 代码块标记

    Args:
        response: LLM 返回的原始字符串

    Returns:
        清理后的 JSON 字符串
    """
    # 尝试匹配 ```json ... ``` 或 ``` ... ``` 代码块
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(code_block_pattern, response)
    if match:
        return match.group(1).strip()

    # 如果没有代码块标记，尝试直接匹配 JSON 数组
    json_array_pattern = r'\[[\s\S]*\]'
    match = re.search(json_array_pattern, response)
    if match:
        return match.group(0)

    # 如果都没匹配到，返回原始内容（去除首尾空白）
    return response.strip()

class search_logger():
    def __init__(self,llm,query):
        self.log=dict()
        self.log['plan']=''
        self.log['query']=query
        self.log['process']=dict()
        self.log['final_res']=list()
        self.log['final_feeds']=list()
        self.log['search_logs']=dict()  # 存储每个步骤的搜索日志
        self.llm=llm
    def _clean_keys(self,poi_list, valid_keys=['POI名称', 'POI简介', '好评内容', '差评内容']):
        cleaned_list = []
        for i, poi in enumerate(poi_list):
            cleaned_poi = {}
            removed_keys = []

            # 检查每个key
            for key in poi.keys():
                if key in valid_keys:
                    cleaned_poi[key] = poi[key]
                else:
                    removed_keys.append(key)

            # 如果有不符合规范的key,打印警告信息
            if removed_keys:
                print(f"⚠️ POI索引 {i} (名称: {poi.get('POI名称', '未知')}) 删除了不规范的key: {removed_keys}")

            # 检查是否缺少必需的key
            missing_keys = [k for k in valid_keys if k not in cleaned_poi]
            if missing_keys:
                print(f"⚠️ POI索引 {i} (名称: {poi.get('POI名称', '未知')}) 缺少必需的key: {missing_keys}")
                # 为缺失的key添加默认值
                for key in missing_keys:
                    if key == 'POI名称':
                        cleaned_poi[key] = '未命名POI'
                    elif key == 'POI简介':
                        cleaned_poi[key] = ''
                    elif key in ['好评内容', '差评内容']:
                        cleaned_poi[key] = []

            cleaned_list.append(cleaned_poi)
        return cleaned_list
    def summary_step(self,search_res,feed_list):
        total_pois=len(search_res)
        total_feeds=len(set(feed_list))
        existed_related_pois=0
        new_realted_pois=0
        existed_feeds=0
        new_unrelated_pois=0
        new_comments=0
        ###检查feed_id###
        for feed in list(set(feed_list)):
            if feed in self.log['final_feeds']:
                existed_feeds+=1
            else:
                self.log['final_feeds'].append(feed)
        ###检查poi###
        pois_to_evaluate = []
        search_res_clean=self._clean_keys(search_res)
        for poi_info in search_res_clean:
            flag=0
            for existed_poi in self.log['final_res']:
                if existed_poi['POI名称'].lower() in poi_info['POI名称'].lower():
                    ###如果已有poi中包含该新poi，则直接合并
                    existed_related_pois+=1
                    existed_poi['好评内容'].extend(poi_info['好评内容'])
                    existed_poi['差评内容'].extend(poi_info['差评内容'])
                    new_comments=new_comments+len(poi_info['好评内容'])+len(poi_info['差评内容'])
                    poi_info['是否匹配']=existed_poi['是否匹配']
                    poi_info['判断理由']=existed_poi['判断理由']
                    flag=1
                    break
                elif poi_info['POI名称'].lower() in existed_poi['POI名称'].lower():
                    ###如果新poi中包含该已有poi，则把已有poi的信息合并到新poi，移除已有poi，重新评估新poi
                    existed_related_pois+=1
                    # 把 existed_poi 的评论合并到 poi_info
                    poi_info['好评内容'].extend(existed_poi['好评内容'])
                    poi_info['差评内容'].extend(existed_poi['差评内容'])
                    new_comments=new_comments+len(poi_info['好评内容'])+len(poi_info['差评内容'])
                    # 从 final_res 中移除 existed_poi
                    self.log['final_res'].remove(existed_poi)
                    # 评估 poi_info 而不是 existed_poi
                    pois_to_evaluate.append(poi_info)
                    flag=1
                    break
            if flag == 0:
                pois_to_evaluate.append(poi_info)
        if pois_to_evaluate:
            with ThreadPoolExecutor(max_workers=10) as executor:
                # 提交所有评估任务
                future_to_poi = {executor.submit(self.match_eval, poi_info): poi_info 
                                for poi_info in pois_to_evaluate}
                # 获取评估结果
                for future in as_completed(future_to_poi):
                    poi_info = future_to_poi[future]
                    try:
                        match = future.result()
                        poi_info['是否匹配'] = match['是否匹配']
                        poi_info['判断理由'] = match['判断理由']
                        if poi_info['是否匹配'] == "是":
                            poi_info['好评内容']=list(set(poi_info['好评内容']))
                            poi_info['差评内容']=list(set(poi_info['差评内容']))
                            if poi_info not in self.log['final_res']:
                                new_realted_pois += 1
                                self.log['final_res'].append(poi_info)
                                new_comments=new_comments+len(poi_info['好评内容'])+len(poi_info['差评内容'])
                        else:
                            if poi_info in self.log['final_res']:
                                self.log['final_res'].remove(poi_info)
                            else:
                                new_unrelated_pois += 1
                    except Exception as e:
                        print(f"评估POI时出错: {e}")
                        poi_info['是否匹配'] = "不确定"
                        poi_info['判断理由'] = "评估是API执行出错，按照不确定处理"
                        new_unrelated_pois += 1
        ####总结内容写入####
        summary="本次搜索任务共搜索到%d个网页，其中%d个是之前已经搜索过的重复网页。新搜索到的网页中共发现%d个候选POI，其中%d个是之前被搜索到的POI,%d个是与用户请求相关的新候选POI，%d个被认为是与用户请求不相关的POI。共搜到%d个与已有或新增POI相关的评论。"%(total_feeds,existed_feeds,total_pois,existed_related_pois,new_realted_pois,new_unrelated_pois,new_comments)
        return summary,search_res_clean
    def add_searchlog(self,step):
        self.log['process'][str(step['行动步骤'])]=step
    def match_eval(self,poi_info):
        prompt=agent_match_prompt(self.log['query'],poi_info)
        try:
            match_res=self.llm.call_with_messages_small(prompt,temp=0)
            return json.loads(match_res)
        except:
            return {"是否匹配":"不确定","判断理由":"调用大模型时出错,没有给出可用答案"}

class search_agent():
    def __init__(self,poi_search,base_llm,query):
        self.poi_tool=poi_search
        self.llm=base_llm
        self.query=query
        self.logger=search_logger(base_llm,query)
    def self_plan(self,sample_temperature=0):
        prompt=search_plan_prompt(self.query)
        res_plan=self.llm.call_with_messages_V3(prompt,temp=sample_temperature)
        # 清理 markdown 代码块标记并解析 JSON
        clean_plan = extract_json_from_response(res_plan)
        self.logger.log['plan']=clean_plan
        return json.loads(clean_plan)
    def execution(self,plan):
        # 第一阶段：并行执行所有搜索任务（不打印中间结果）
        search_results = {}  # step_index -> (search_res, feed_list, search_record, run_log)
        advice = "-最终将按照以下标准判断候选POI搜索结果的质量：1.搜索出来的尽量多的候选POI；2.每个候选POI都必须与用户的请求相关；3.对每个POI都有尽量多的多方面评价，既有好评又有差评。"

        print(f"开始并行搜索 {len(plan)} 个步骤...")

        def run_search(step_index, step):
            search_res, feed_list, search_record, run_log = self.poi_tool.run(
                step['搜索请求'], advice=advice, verbose=False
            )
            return step_index, search_res, feed_list, search_record, run_log

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(run_search, i, step): i for i, step in enumerate(plan)}
            for future in as_completed(futures):
                step_index, search_res, feed_list, search_record, run_log = future.result()
                search_results[step_index] = (search_res, feed_list, search_record, run_log)
                print(f"  ✓ 步骤 {step_index + 1} 搜索完成，发现 {len(search_res)} 个候选POI")

        print(f"\n所有搜索完成，开始汇总结果...")

        # 第二阶段：按顺序汇总结果（保证 summary_step 的串行执行）
        for i, step in enumerate(plan):
            search_res, feed_list, search_record, run_log = search_results[i]
            # 保存搜索日志
            self.logger.log['search_logs'][str(step['行动步骤'])] = run_log
            summary, match_res = self.logger.summary_step(search_res, feed_list)
            step['执行总结'] = summary
            step['执行结果'] = match_res
            step['搜索过程'] = search_record + '\nPOI与用户请求匹配结果:\n' + str([item['POI名称']+",是否匹配:"+item['是否匹配'] for item in match_res])
            step['本步骤搜索网页'] = feed_list
            step['本步骤搜索POI'] = search_res
            self.logger.add_searchlog(step)
            # 打印汇总结果
            print(f"  步骤 {i + 1}: {summary}")

        print(f"\n初始搜索完成，共找到 {len(self.logger.log['final_res'])} 个相关POI")
        return self.logger.log['final_res']
    def revise_execution(self,opt_steps):
        self.logger.log['final_res']=list()
        self.logger.log['final_feeds']=list()
        print("开始优化执行...")
        advice = "-最终将按照以下标准判断候选POI搜索结果的质量：1.搜索出来的尽量多的候选POI；2.每个候选POI都必须与用户的请求相关；3.对每个POI都有尽量多的多方面评价，既有好评又有差评。"
        # 统一 opt_steps 的键为字符串
        opt_steps_str = {str(k): v for k, v in opt_steps.items()}
        plan_steps = json.loads(self.logger.log['plan'])

        # 第一阶段：分类步骤并更新 plan
        steps_to_search = []  # 需要搜索的步骤 (step_key, search_query, case_type)
        cached_results = {}   # 缓存结果 step_key -> (search_res, feed_list, search_record)

        for step in plan_steps:
            step_key = str(step["行动步骤"])
            if step_key not in opt_steps_str and step_key in self.logger.log['process']:
                # Case 1: 复用缓存
                print(f"  步骤 {step_key}: 复用缓存结果")
                cached_results[step_key] = (
                    copy.deepcopy(self.logger.log['process'][step_key]['本步骤搜索POI']),
                    copy.deepcopy(self.logger.log['process'][step_key]['本步骤搜索网页']),
                    self.logger.log['process'][step_key]['搜索过程']
                )
            elif step_key in opt_steps_str:
                # Case 2: 需要优化搜索，先更新 plan
                cur_plan = json.loads(self.logger.log['plan'])
                for plan_step in cur_plan:
                    if str(plan_step['行动步骤']) == step_key:
                        plan_step['行动规划'] = opt_steps_str[step_key]['行动规划']
                        plan_step['搜索请求'] = opt_steps_str[step_key]['搜索请求']
                        step['行动规划'] = opt_steps_str[step_key]['行动规划']
                        step['搜索请求'] = opt_steps_str[step_key]['搜索请求']
                        break
                self.logger.log['plan'] = json.dumps(cur_plan, ensure_ascii=False)
                steps_to_search.append((step_key, opt_steps_str[step_key]['搜索请求'], 'optimize'))
            else:
                # Case 3: 之前未执行，需要执行
                steps_to_search.append((step_key, step['搜索请求'], 'new'))

        # 第二阶段：并行执行需要搜索的步骤
        search_results = {}  # step_key -> (search_res, feed_list, search_record, run_log)
        if steps_to_search:
            print(f"\n并行搜索 {len(steps_to_search)} 个步骤...")

            def run_search(step_key, search_query, case_type):
                search_res, feed_list, search_record, run_log = self.poi_tool.run(
                    search_query, advice=advice, verbose=False
                )
                return step_key, search_res, feed_list, search_record, run_log, case_type

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(run_search, step_key, query, case_type): step_key
                    for step_key, query, case_type in steps_to_search
                }
                for future in as_completed(futures):
                    step_key, search_res, feed_list, search_record, run_log, case_type = future.result()
                    search_results[step_key] = (search_res, feed_list, search_record, run_log)
                    case_desc = "优化搜索" if case_type == 'optimize' else "新执行"
                    print(f"  ✓ 步骤 {step_key} ({case_desc}) 完成，发现 {len(search_res)} 个候选POI")

        # 第三阶段：按顺序串行汇总结果
        print(f"\n汇总结果...")
        for step in plan_steps:
            step_key = str(step["行动步骤"])
            if step_key in cached_results:
                # Case 1: 使用缓存
                search_res, feed_list, search_record = cached_results[step_key]
            else:
                # Case 2/3: 使用搜索结果
                search_res, feed_list, search_record, run_log = search_results[step_key]
                self.logger.log['search_logs'][step_key] = run_log

            summary, match_res = self.logger.summary_step(search_res, feed_list)
            step['执行总结'] = summary
            step['执行结果'] = match_res
            step['搜索过程'] = search_record + '\nPOI与用户请求匹配结果:\n' + str([item['POI名称']+",是否匹配:"+item['是否匹配']+",理由:"+item['判断理由'] for item in match_res])
            step['本步骤搜索网页'] = feed_list
            step['本步骤搜索POI'] = search_res
            self.logger.add_searchlog(step)
            print(f"  步骤 {step_key}: {summary}")

        print(f"\n优化执行完成，共找到 {len(self.logger.log['final_res'])} 个相关POI")
        return self.logger.log['final_res']
    def self_plan_advice(self,advices,sample_temperature=0):
        prompt=search_opt_plan_prompt(self.query,advice=advices)
        res_plan=self.llm.call_with_messages_V3(prompt,temp=sample_temperature)
        # 清理 markdown 代码块标记并解析 JSON
        clean_plan = extract_json_from_response(res_plan)
        self.logger.log['plan']=clean_plan
        return json.loads(clean_plan)