from datetime import datetime
import re
import json
from typing import List, Dict, Any, Callable, Optional
from src.tools.tools import Tool
from src.core.prompts import poi_parse_prompt

class Search_ReActAgent:
    """
    ReAct Agent 实现
    
    ReAct 模式：Thought -> Action -> Observation -> (循环) -> Answer
    """
    
    def __init__(self, llm_client, tools: List[Tool], max_iterations: int = 5):
        """
        初始化 Agent
        
        Args:
            llm_client: LLM 客户端（如 OpenAI API）
            tools: 可用工具列表
            max_iterations: 最大迭代次数
        """
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        tools_desc = "\n".join([
            f"- {name}: {tool.description}" 
            for name, tool in self.tools.items()
        ])
        
        return f'''你是一个旅行助手，现在需要使用工具搜索可能满足用户请求的候选POI，及其简介和评价。

可用工具：
{tools_desc}

你必须按照以下格式思考和行动：

Thought: 分析当前情况，思考下一步该做什么
Action: 选择要使用的工具名称
Action Input: 工具的输入参数
Observation: [系统会自动填充工具执行结果]

请基于<历史记录>中的信息规划 Thought/Action/Action Input 的思考和行动内容，直到你有足够信息回答问题。

当你准备好给出最终答案时，使用：
Thought: 我现在知道最终答案了
Final Answer: [你的答案]

重要规则：
1. 你每次只能使用一个工具,或执行一个步骤,不能一次执行多个步骤;
2. 你必须严格按照格式输出Thought/Action/Action Input/Final Answer；
3. 如果遇到错误，分析原因并尝试其他方法
当前的时间是{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}:

'''

    def _parse_llm_output(self, text: str) -> Dict[str, str]:
        """解析 LLM 输出"""
        result = {}
        
        # 提取 Thought
        thought_match = re.search(r'Thought:(.*?)(?=Action:|Final Answer:|$)', 
                                 text, re.DOTALL)
        if thought_match:
            result['thought'] = thought_match.group(1).strip()
        
        # 提取 Final Answer
        final_match = re.search(r'Final Answer:(.*?)$', text, re.DOTALL)
        if final_match:
            result['final_answer'] = final_match.group(1).strip()
        
        # 提取 Action
        action_match = re.search(r'Action:(.*?)(?=Action Input:|$)', 
                                text, re.DOTALL)
        if action_match:
            result['action'] = action_match.group(1).strip()
        
        # 提取 Action Input
        input_match = re.search(r'Action Input:(.*?)$', text, re.DOTALL)
        if input_match:
            result['action_input'] = input_match.group(1).strip()
        
        return result

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """执行工具"""
        if tool_name not in self.tools:
            return f"错误: 工具 '{tool_name}' 不存在。可用工具: {list(self.tools.keys())}"
        
        try:
            tool = self.tools[tool_name]
            result = tool.func(tool_input)
            return str(result)
        except Exception as e:
            return f"工具执行错误: {str(e)}"

    def run(self, question: str, advice: str = "",verbose: bool = True) -> str:
        """
        运行 Agent

        Args:
            question: 用户问题
            verbose: 是否打印中间过程

        Returns:
            最终答案, feed_list, search_record, run_log (当verbose=False时)
        """
        feed_list=[]
        run_log = []  # 收集运行日志
        system_prompt = self._build_system_prompt()
        conversation_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"用户请求: {question}\n行动建议:{advice}"}
        ]

        for iteration in range(self.max_iterations):
            iter_header = f"\n{'='*50}\n迭代 {iteration + 1}/{self.max_iterations}\n{'='*50}"
            if verbose:
                print(iter_header)
            else:
                run_log.append(iter_header)

            # 调用 LLM
            response = self._call_llm(conversation_history)
            ##幻觉消除，大模型自己输出外部response
            if "Observation" in response:
                response = response.split("Observation")[0]
            llm_output = f"\nLLM 输出:\n{response}"
            if verbose:
                print(llm_output)
            else:
                run_log.append(llm_output)

            # 解析输出
            parsed = self._parse_llm_output(response)
             # 检查是否得到最终答案
            if 'final_answer' in parsed and 'action' not in parsed:
                final_msg = f"\n✓ 找到最终答案！"
                if verbose:
                    print(final_msg)
                else:
                    run_log.append(final_msg)
                res,search_record=self._final_summary(conversation_history)
                return res,feed_list,search_record,run_log

            # 检查是否有有效的 action
            if 'action' not in parsed or 'action_input' not in parsed:
                retry_msg = "\n✗ 无法解析有效的 Action，尝试重新生成..."
                if verbose:
                    print(retry_msg)
                else:
                    run_log.append(retry_msg)
                conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                conversation_history.append({
                    "role": "user",
                    "content": "请按照正确的格式输出 Thought, Action 和 Action Input"
                })
                continue

            # 执行工具
            action = parsed['action']
            action_input = parsed['action_input']

            tool_msg = f"\n执行工具: {action}\n输入参数: {action_input}"
            if verbose:
                print(tool_msg)
            else:
                run_log.append(tool_msg)
            try:
                observation_res = self._execute_tool(action, action_input)
                observation = eval(observation_res)['content']
                observation += f"\n剩余可用迭代次数:{self.max_iterations-iteration-1}"
                feed_list.extend(eval(observation_res)['urls'])

                obs_msg = f"观察结果: {observation}"
                if verbose:
                    print(obs_msg)
                else:
                    run_log.append(obs_msg)

                # 更新对话历史
                conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                conversation_history.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })
            except:
                observation ="工具执行错误。"+ f"\n剩余可用迭代次数:{self.max_iterations-iteration-1}"
                obs_msg = f"观察结果: {observation}"
                if verbose:
                    print(obs_msg)
                else:
                    run_log.append(obs_msg)

                # 更新对话历史
                conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                conversation_history.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })
        timeout_msg = f"抱歉，在 {self.max_iterations} 次迭代内未能找到答案。"
        if verbose:
            print(timeout_msg)
        else:
            run_log.append(timeout_msg)
        res,search_record=self._final_summary(conversation_history)
        return res,feed_list,search_record,run_log
    def _final_summary(self,messages):
        content= "\n<用户请求>\n"+messages[1]['content']+"\n<\用户请求>\n"
        if len(messages)>2:
            content +="\n<搜索记录>\n"+'\n'.join(mes['content'] for mes in messages[2:])+"\n<\搜索记录>\n"
        prompt=poi_parse_prompt(content)
        for retry in range(3):
            try:
                result = self.llm.call_with_messages_V3(prompt,temp=0)
                match = re.search(r'\[[\s\S]*\]', result)
                json_str = match.group()
                res=json.loads(json_str)
                return res,content
            except Exception as e:
                print(e)
                print("retrying summary process!")
                continue
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        prompt=messages[0]['content']+'\n'+messages[1]['content']
        if len(messages)>2:
            prompt +="\n<历史记录>\n"+'\n'.join(mes['content'] for mes in messages[2:])+"\n<\历史记录>\n"
        result = self.llm.call_with_messages_V3(prompt,temp=0)
        return result
    
    
class Answer_ReActAgent:
    """
    ReAct Agent 实现
    
    ReAct 模式：Thought -> Action -> Observation -> (循环) -> Answer
    """
    
    def __init__(self, llm_client, tools: List[Tool], max_iterations: int = 5):
        """
        初始化 Agent
        
        Args:
            llm_client: LLM 客户端（如 OpenAI API）
            tools: 可用工具列表
            max_iterations: 最大迭代次数
        """
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        tools_desc = "\n".join([
            f"- {name}: {tool.description}" 
            for name, tool in self.tools.items()
        ])
        
        return f'''你是一个人工智能助手，现在需要使用工具帮助你搜索信息，回答用户问题。

可用工具：
{tools_desc}

你必须按照以下格式思考和行动：

Thought: 分析当前情况，思考下一步该做什么
Action: 选择要使用的工具名称
Action Input: 工具的输入参数
Observation: [系统会自动填充工具执行结果]

请基于<历史记录>中的信息规划 Thought/Action/Action Input 的思考和行动内容，直到你有足够信息回答问题。

当你准备好给出最终答案时，使用：
Thought: 我现在知道最终答案了
Final Answer: [你的答案]

重要规则：
1. 你每次只能使用一个工具,或执行一个步骤,不能一次执行多个步骤;
2. 你必须严格按照格式输出Thought/Action/Action Input/Final Answer；
3. 如果遇到错误，分析原因并尝试其他方法
当前的时间是{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}:

'''

    def _parse_llm_output(self, text: str) -> Dict[str, str]:
        """解析 LLM 输出"""
        result = {}
        
        # 提取 Thought
        thought_match = re.search(r'Thought:(.*?)(?=Action:|Final Answer:|$)', 
                                 text, re.DOTALL)
        if thought_match:
            result['thought'] = thought_match.group(1).strip()
        
        # 提取 Final Answer
        final_match = re.search(r'Final Answer:(.*?)$', text, re.DOTALL)
        if final_match:
            result['final_answer'] = final_match.group(1).strip()
        
        # 提取 Action
        action_match = re.search(r'Action:(.*?)(?=Action Input:|$)', 
                                text, re.DOTALL)
        if action_match:
            result['action'] = action_match.group(1).strip()
        
        # 提取 Action Input
        input_match = re.search(r'Action Input:(.*?)$', text, re.DOTALL)
        if input_match:
            result['action_input'] = input_match.group(1).strip()
        
        return result

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """执行工具"""
        if tool_name not in self.tools:
            return f"错误: 工具 '{tool_name}' 不存在。可用工具: {list(self.tools.keys())}"
        
        try:
            tool = self.tools[tool_name]
            result = tool.func(tool_input)
            return str(result)
        except Exception as e:
            return f"工具执行错误: {str(e)}"

    def run(self, question: str, advice: str = "",verbose: bool = True) -> str:
        """
        运行 Agent
        
        Args:
            question: 用户问题
            verbose: 是否打印中间过程
            
        Returns:
            最终答案
        """
        feed_list=[]
        system_prompt = self._build_system_prompt()
        conversation_history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"用户请求: {question}\n行动建议:{advice}"}
        ]
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n{'='*50}")
                print(f"迭代 {iteration + 1}/{self.max_iterations}")
                print(f"{'='*50}")
            
            # 调用 LLM
            response = self._call_llm(conversation_history)
            ##幻觉消除，大模型自己输出外部response
            if "Observation" in response:
                response = response.split("Observation")[0]
            if verbose:
                print(f"\nLLM 输出:\n{response}")
            
            # 解析输出
            parsed = self._parse_llm_output(response)
             # 检查是否得到最终答案
            if 'final_answer' in parsed and 'action' not in parsed:
                if verbose:
                    print(f"\n✓ 找到最终答案！")
                return parsed['final_answer'] 
            
            # 检查是否有有效的 action
            if 'action' not in parsed or 'action_input' not in parsed:
                if verbose:
                    print("\n✗ 无法解析有效的 Action，尝试重新生成...")
                conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                conversation_history.append({
                    "role": "user",
                    "content": "请按照正确的格式输出 Thought, Action 和 Action Input"
                })
                continue
            
            # 执行工具
            action = parsed['action']
            action_input = parsed['action_input']
            
            if verbose:
                print(f"\n执行工具: {action}")
                print(f"输入参数: {action_input}")
            try:
                observation_res = self._execute_tool(action, action_input)
                observation = eval(observation_res)['content']
                observation += f"\n剩余可用迭代次数:{self.max_iterations-iteration-1}"
                feed_list.extend(eval(observation_res)['urls'])

                if verbose:
                    print(f"观察结果: {observation}")

                # 更新对话历史
                conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                conversation_history.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })
            except:
                observation ="工具执行错误。"+ f"\n剩余可用迭代次数:{self.max_iterations-iteration-1}"
                if verbose:
                    print(f"观察结果: {observation}")

                # 更新对话历史
                conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                conversation_history.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })
        return f"抱歉，在 {self.max_iterations} 次迭代内未能找到答案。"
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        prompt=messages[0]['content']+'\n'+messages[1]['content']
        if len(messages)>2:
            prompt +="\n<历史记录>\n"+'\n'.join(mes['content'] for mes in messages[2:])+"\n<\历史记录>\n"
        result = self.llm.call_with_messages_V3(prompt,temp=0)
        return result
