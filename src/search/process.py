from src.search.searchagent import search_agent,search_logger
from src.core.prompts import troubleshoot_prompt,refine_prompt,new_step_prompt,new_sample_prompt
from src.core.basellm import base_llm
from src.agent.react import Search_ReActAgent
from src.tools.tools import tools
import json
import re

def summary_log(agent):
    res_log=dict()
    res_log['执行日志']=list()
    process_log=agent.logger.log['process']
    total_comments=0
    step_count=0
    for i in process_log.keys():
        res_step=dict()
        log=process_log[i]
        res_step['行动步骤']=log['行动步骤']
        res_step['行动规划']=log['行动规划']
        res_step['搜索请求']=log['搜索请求']
        res_step['搜索POI']=[item['POI名称']+",简介:"+item['POI简介']+",是否匹配:"+item['是否匹配'] for item in log['执行结果']]
        res_step['步骤总结']=log['执行总结']
        res_log['执行日志'].append(res_step)
        step_count+=1
    ##总comments##
    for item in agent.logger.log['final_res']:
        total_comments+=len(item['好评内容'])+len(item['差评内容'])
    total_pois=len(agent.logger.log['final_res'])
    ###任务总结###
    res_log['任务总结']="本次任务共执行%d次,共搜索到%d个候选POI,以及%d个与候选POI相关评论。"%(step_count,total_pois,total_comments)
    return res_log

def search_process(topic,on_policy_opt = True, maximum_opt_iterations = 10, use_advice = False):
    llm = base_llm(system_prompt="")
    executor = Search_ReActAgent(llm_client=llm, tools=tools)
    total_log=dict()
    for retry in range(3):
        try:
            print(f"start initial plan of request: {topic}.\nrepeats {retry}")
            searcher=search_agent(executor,llm,topic)
            if use_advice:
                with open('data/skills.json', 'r', encoding='utf-8') as f:
                    advices = json.load(f)
                total_advice=''
                advice_count=0
                for item in advices:
                    for advice in item['content']:
                        if advice_count<100:
                            total_advice+=advice+'\n'
                            advice_count+=1
                plan=searcher.self_plan_advice(total_advice,sample_temperature=0.1)
                result=searcher.execution(plan)
                break
            else:
                plan=searcher.self_plan()
                result=searcher.execution(plan)
                break
        except Exception as e:
            print(e)
    log_ref=summary_log(searcher)
    log=log_ref.copy()
    if on_policy_opt:
        total_log=dict()
        total_log[0]=log_ref
        optimization_steps=[]
        optimization_maxtimes=3
        for j in range(maximum_opt_iterations):
            for retry in range(3):
                try:
                    print(f"start optimization trial of {j+1},retry {retry}")
                    new_steps=dict()
                    crit="-最终将按照以下标准判断候选POI搜索结果的质量：1.搜索出来的尽量多的候选POI；2.每个候选POI都必须与用户的请求相关；"
                    troubleshoot=troubleshoot_prompt(topic,str(log),crit)
                    troubleshoot_res=llm.call_with_messages_R1(troubleshoot,temp=retry*0.2)
                    match = re.search(r'\[[\s\S]*\]', troubleshoot_res)
                    json_str = match.group()
                    opt_res=json.loads(json_str)
                    print(opt_res)
                    for item in opt_res:
                        search_content=searcher.logger.log['process'][item['行动步骤']]['搜索过程']
                        for step_plan in eval(searcher.logger.log['plan']):
                            if step_plan['行动步骤'] == item['行动步骤']:
                                cur_plan=step_plan
                                break
                        if optimization_steps.count(item['行动步骤'])<optimization_maxtimes:
                            print(f"start optimization solution for step {item['行动步骤']}!")
                            cur_problem=item['当前问题']
                            opt_prompt=refine_prompt(search_content,cur_problem,cur_plan)
                            opt_advice=llm.call_with_messages_R1(opt_prompt,temp=0)
                            suggestion=eval(opt_advice)['修改建议']
                            item['修改建议']=suggestion
                            print(suggestion)
                            new_prompt=new_step_prompt(searcher.query,cur_plan,str(item))
                            new_step=llm.call_with_messages_V3(new_prompt,temp=0)
                            pattern = r"\{[^{}]*(?:'[^']*'[^{}]*)*\}"
                            match=re.search(pattern, new_step)
                            new_step_res=json.loads(match.group())
                            new_steps[item['行动步骤']]=new_step_res
                            optimization_steps.append(item['行动步骤'])
                        else:
                            print(f"start new sample solution for step {item['行动步骤']}!")
                            new_sampleprompt=new_sample_prompt(topic,searcher.logger.log['plan'])
                            new_step=llm.call_with_messages_V3(new_sampleprompt,temp=0.6)
                            new_step_res=json.loads(new_step)
                            new_step_res['行动步骤']=item['行动步骤']
                            new_steps[item['行动步骤']]=new_step_res
                            optimization_steps = [x for x in optimization_steps if x != item['行动步骤']]
                    result=searcher.revise_execution(new_steps)
                    break
                except Exception as e:
                    print(f"exceptions while optimization of topic {topic} on {j+1} rounds,repeats {retry}!")
                    print(e)
            log=summary_log(searcher)
            total_log[j+1]=log
    return log,log_ref