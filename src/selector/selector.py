"""
POI选择系统 - 通过二选一问答帮助用户筛选POI

功能：
1. 根据POI列表和用户请求，使用LLM生成二选一问题
2. 预先构建决策树，每个节点包含问题和基于答案的POI筛选结果
3. 支持用户交互，包括换题、提前退出、回退
4. 维护对话历史日志，帮助LLM更准确地生成问题和筛选POI
5. 【新增】决策树与交互过程分离，支持基于同一决策树多次选择
6. 【新增】决策树持久化存储，支持保存和加载
7. 【新增】选择完成后支持立即重新选择
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, Any
from enum import Enum
from datetime import datetime
from src.core.basellm import base_llm




# ============== 数据结构定义 ==============

class UserChoice(Enum):
    """用户选择枚举"""
    OPTION_A = "A"
    OPTION_B = "B"
    CHANGE_QUESTION = "C"
    EXIT = "D"
    GO_BACK = "E"
    RESTART = "R"  # 【新增】重新开始选择
    LOAD = "L"     # 【新增】加载已有决策树
    USER_INPUT = "F"   # 【新增】用户自定义输入
    ASK_AGENT = "G"    # 【新增】向agent提问
    SHOW_POIS = "H"    # 【新增】显示当前所有POI

@dataclass
class POI:
    """POI数据结构"""
    name: str
    description: str
    
    def to_string(self) -> str:
        return f"{self.name}: {self.description}"
    
    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description}
    
    @classmethod
    def from_dict(cls, data: dict) -> 'POI':
        return cls(name=data["name"], description=data["description"])
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, POI):
            return self.name == other.name
        return False


@dataclass
class QARecord:
    """问答记录"""
    question: str
    option_a: str
    option_b: str
    user_choice: Optional[str] = None
    record_type: str = "qa"  # 【新增】记录类型: "qa", "user_input", "agent_qa"
    user_input_text: Optional[str] = None  # 【新增】用户自定义输入的文本
    agent_question: Optional[str] = None   # 【新增】用户向agent提的问题
    agent_answer: Optional[str] = None     # 【新增】agent的回答
    
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "option_a": self.option_a,
            "option_b": self.option_b,
            "user_choice": self.user_choice,
            "record_type": self.record_type,
            "user_input_text": self.user_input_text,
            "agent_question": self.agent_question,
            "agent_answer": self.agent_answer
        }


@dataclass
class ConversationLogger:
    """对话历史日志记录器"""
    user_request: str = ""
    qa_history: list[QARecord] = field(default_factory=list)
    
    def add_record(self, record: QARecord):
        self.qa_history.append(record)
    
    def get_history_prompt(self) -> str:
        if not self.qa_history:
            return """
【历史问答记录】
说明：以下是用户与系统之间的历史问答记录，记录了用户的偏好选择过程。
状态：暂无历史问答记录。
"""
        
        history_text = """
【历史问答记录】
说明：以下是用户与系统之间的历史问答记录，记录了用户的偏好选择过程。
      请根据这些历史选择来理解用户的真实需求和偏好，生成更精准的问题或筛选POI。
记录内容：
"""
        for i, record in enumerate(self.qa_history, 1):
            if record.record_type == "user_input":
                # 用户自定义输入类型
                history_text += f"""
  第{i}轮 - 用户自定义输入:
    - 用户补充信息: {record.user_input_text}
"""
            elif record.record_type == "agent_qa":
                # Agent问答类型
                history_text += f"""
  第{i}轮 - 用户向Agent提问:
    - 用户问题: {record.agent_question}
    - Agent回答: {record.agent_answer}
"""
            else:
                # 普通问答类型
                choice_text = ""
                if record.user_choice == "A":
                    choice_text = f"用户选择了A: {record.option_a}"
                elif record.user_choice == "B":
                    choice_text = f"用户选择了B: {record.option_b}"
                
                history_text += f"""
  第{i}轮问答:
    - 问题: {record.question}
    - 选项A: {record.option_a}
    - 选项B: {record.option_b}
    - 结果: {choice_text}
"""
        return history_text
    
    def clear(self):
        """清空历史记录"""
        self.qa_history = []


@dataclass
class DecisionNode:
    """决策树节点"""
    question: str
    option_a: str
    option_b: str
    current_pois: list[POI]
    pois_if_a: list[POI] = field(default_factory=list)
    pois_if_b: list[POI] = field(default_factory=list)
    child_a: Optional['DecisionNode'] = None
    child_b: Optional['DecisionNode'] = None
    depth: int = 0
    
    def to_dict(self) -> dict:
        """转换为可序列化的字典"""
        return {
            "question": self.question,
            "option_a": self.option_a,
            "option_b": self.option_b,
            "current_pois": [poi.to_dict() for poi in self.current_pois],
            "pois_if_a": [poi.to_dict() for poi in self.pois_if_a],
            "pois_if_b": [poi.to_dict() for poi in self.pois_if_b],
            "child_a": self.child_a.to_dict() if self.child_a else None,
            "child_b": self.child_b.to_dict() if self.child_b else None,
            "depth": self.depth
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DecisionNode':
        """从字典重建节点"""
        node = cls(
            question=data["question"],
            option_a=data["option_a"],
            option_b=data["option_b"],
            current_pois=[POI.from_dict(p) for p in data["current_pois"]],
            pois_if_a=[POI.from_dict(p) for p in data["pois_if_a"]],
            pois_if_b=[POI.from_dict(p) for p in data["pois_if_b"]],
            depth=data["depth"]
        )
        if data["child_a"]:
            node.child_a = cls.from_dict(data["child_a"])
        if data["child_b"]:
            node.child_b = cls.from_dict(data["child_b"])
        return node


@dataclass
class DecisionTreeData:
    """【新增】决策树完整数据，用于持久化存储"""
    user_request: str
    pois: list[POI]
    root: Optional[DecisionNode]
    created_at: str
    max_depth: int
    min_pois_to_continue: int
    
    def to_dict(self) -> dict:
        return {
            "user_request": self.user_request,
            "pois": [poi.to_dict() for poi in self.pois],
            "root": self.root.to_dict() if self.root else None,
            "created_at": self.created_at,
            "max_depth": self.max_depth,
            "min_pois_to_continue": self.min_pois_to_continue
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DecisionTreeData':
        return cls(
            user_request=data["user_request"],
            pois=[POI.from_dict(p) for p in data["pois"]],
            root=DecisionNode.from_dict(data["root"]) if data["root"] else None,
            created_at=data["created_at"],
            max_depth=data["max_depth"],
            min_pois_to_continue=data["min_pois_to_continue"]
        )


# ============== 决策树构建器（与交互分离） ==============

class DecisionTreeBuilder:
    """【新增】决策树构建器 - 负责构建决策树，与用户交互分离"""
    
    def __init__(
        self,
        large_llm_api: Callable[[str], str],
        small_llm_api: Callable[[str], str],
        max_depth: int = 10,
        min_pois_to_continue: int = 2
    ):
        self.large_llm_api = large_llm_api
        self.small_llm_api = small_llm_api
        self.max_depth = max_depth
        self.min_pois_to_continue = min_pois_to_continue
        self.logger = ConversationLogger()
    
    def _format_poi_list(self, pois: list[POI]) -> str:
        return "\n".join([f"- {poi.to_string()}" for poi in pois])
    
    def _generate_question(self, pois: list[POI], depth: int, base_temp: float = 0.0) -> tuple[str, str, str]:
        """
        生成二选一问题

        Args:
            pois: 当前POI列表
            depth: 当前深度
            base_temp: 基础temperature，用于控制生成多样性

        Returns:
            (question, option_a, option_b) 元组
        """
        prompt = f"""你是一个帮助用户选择地点(POI)的助手，需要通过提问来帮助用户缩小选择范围。

【用户原始请求】
{self.logger.user_request}

{self.logger.get_history_prompt()}

【当前待筛选的POI列表】
说明：以下是目前还在候选范围内的POI，共{len(pois)}个。
列表内容：
{self._format_poi_list(pois)}

【任务要求】
请根据用户的原始请求、历史问答记录和当前POI列表，生成一个二选一的问题。

生成问题的原则：
1. 问题应与用户的原始请求相关，帮助理解用户真实需求
2. 问题应能有效区分当前的POI列表，让每个选项都能筛掉一部分POI
3. 不要与历史问题重复或过于相似
4. 问题要具体、易于回答，避免过于抽象
5. 两个选项应该互斥且覆盖大部分情况

【输出格式】
请以JSON格式返回，包含以下字段:
- question: 问题文本
- option_a: 选项A的描述
- option_b: 选项B的描述

只返回JSON，不要其他内容。
"""
        for i in range(3):
            try:
                temp = base_temp + i * 0.2
                response = self.large_llm_api(prompt, temp=temp)
                result = json.loads(response)
                question = result["question"]
                option_a = result["option_a"]
                option_b = result["option_b"]
                return question, option_a, option_b
            except Exception as e:
                print(f"[警告] LLM响应解析失败: {e},重新尝试{i+1}次生成...")
        return "您有什么特别的偏好吗？", "是的，有特定要求", "没有，都可以"
    
    def _filter_pois_by_choice(
        self, 
        pois: list[POI], 
        question: str, 
        option_a: str, 
        option_b: str, 
        choice: str
    ) -> list[POI]:
        chosen_option = option_a if choice == "A" else option_b
        filtered_pois = []
        
        for poi in pois:
            prompt = f"""你是一个POI筛选助手，需要判断单个POI是否符合用户的选择条件。

【用户原始请求】
{self.logger.user_request}

{self.logger.get_history_prompt()}

【当前问答轮次】
问题: {question}
选项A: {option_a}
选项B: {option_b}
用户选择: {choice} ({chosen_option})

【待判断的POI】
名称: {poi.name}
简介: {poi.description}

【判断任务】
请根据用户的选择，判断这个POI是否符合用户的偏好。
- 如果这个POI符合用户选择的"{chosen_option}"这一偏好，返回"保留"
- 如果这个POI不符合用户选择的偏好，返回"过滤"

请以JSON格式返回，包含以下字段:
- decision: "保留" 或 "过滤"
- reason: 简短说明判断理由（一句话）

只返回JSON，不要其他内容。
"""
            
            response = self.small_llm_api(prompt)
            try:
                result = json.loads(response)
                decision = result.get("decision", "保留")
                if decision == "保留":
                    filtered_pois.append(poi)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[警告] 判断POI '{poi.name}' 时LLM响应解析失败: {e}")
                filtered_pois.append(poi)

        # 允许返回空列表，不再保留第一个POI作为兜底
        return filtered_pois
    
    def _build_subtree(self, pois: list[POI], depth: int) -> Optional[DecisionNode]:
        if depth >= self.max_depth:
            return None
        if len(pois) <= self.min_pois_to_continue:
            return None

        # 第一次生成问题并筛选
        question, option_a, option_b = self._generate_question(pois, depth, base_temp=0.0)
        pois_if_a = self._filter_pois_by_choice(pois, question, option_a, option_b, "A")
        pois_if_b = self._filter_pois_by_choice(pois, question, option_a, option_b, "B")

        # 检查是否有空集合
        has_empty = len(pois_if_a) == 0 or len(pois_if_b) == 0

        if has_empty:
            # 发现空集合，检查候选POIs数量
            if len(pois) < self.min_pois_to_continue * 2:
                # 候选POIs太少，直接设为叶节点
                print(f"[信息] 候选POI数量({len(pois)})小于阈值({self.min_pois_to_continue * 2})，设为叶节点")
                return None

            # 候选POIs足够，重试生成问题（最多再试2次，逐次升高temperature）
            max_retries = 3
            for retry in range(1, max_retries):  # 从1开始，因为第0次已经做过了
                base_temp = retry * 0.3  # 0.3, 0.6
                print(f"[警告] 问题生成导致空集合，正在重试 ({retry + 1}/{max_retries})，提高temperature以增加多样性...")

                question, option_a, option_b = self._generate_question(pois, depth, base_temp)
                pois_if_a = self._filter_pois_by_choice(pois, question, option_a, option_b, "A")
                pois_if_b = self._filter_pois_by_choice(pois, question, option_a, option_b, "B")

                # 检查是否还有空集合
                has_empty = len(pois_if_a) == 0 or len(pois_if_b) == 0
                if not has_empty:
                    # 没有空集合了，可以使用这个问题
                    break

            # 如果3次都有空集合，保留空集合答案
            if has_empty:
                print(f"[警告] 3次生成问题都有空集合答案，保留空集合（用户选择该选项时将提示无匹配POI）")

        node = DecisionNode(
            question=question,
            option_a=option_a,
            option_b=option_b,
            current_pois=pois,
            pois_if_a=pois_if_a,
            pois_if_b=pois_if_b,
            depth=depth
        )

        temp_record_a = QARecord(question, option_a, option_b, "A")
        self.logger.add_record(temp_record_a)

        # 只有当选项A对应的POI数量大于阈值时才递归构建子树
        if len(node.pois_if_a) > self.min_pois_to_continue:
            node.child_a = self._build_subtree(node.pois_if_a, depth + 1)

        self.logger.qa_history.pop()
        temp_record_b = QARecord(question, option_a, option_b, "B")
        self.logger.add_record(temp_record_b)

        # 只有当选项B对应的POI数量大于阈值时才递归构建子树
        if len(node.pois_if_b) > self.min_pois_to_continue:
            node.child_b = self._build_subtree(node.pois_if_b, depth + 1)

        self.logger.qa_history.pop()

        return node
    
    def build(self, pois: list[POI], user_request: str) -> DecisionTreeData:
        """
        构建决策树并返回完整数据对象
        
        Args:
            pois: POI列表
            user_request: 用户请求
            
        Returns:
            DecisionTreeData: 包含决策树和元数据的完整数据对象
        """
        print("正在构建决策树，请稍候...")
        self.logger = ConversationLogger()
        self.logger.user_request = user_request
        
        root = self._build_subtree(pois, 0)
        
        tree_data = DecisionTreeData(
            user_request=user_request,
            pois=pois,
            root=root,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            max_depth=self.max_depth,
            min_pois_to_continue=self.min_pois_to_continue
        )
        
        print("决策树构建完成！")
        return tree_data


# ============== 决策树存储管理器 ==============

class DecisionTreeStorage:
    """【新增】决策树存储管理器 - 负责保存和加载决策树"""
    
    DEFAULT_SAVE_DIR = "./data/decision_trees"
    
    def __init__(self, save_dir: str = None):
        self.save_dir = save_dir or self.DEFAULT_SAVE_DIR
        os.makedirs(self.save_dir, exist_ok=True)
    
    def _generate_filename(self, user_request: str) -> str:
        """根据用户请求和时间生成文件名"""
        # 清理用户请求，只保留中文和字母数字
        safe_request = "".join(c for c in user_request if c.isalnum() or '\u4e00' <= c <= '\u9fff')
        safe_request = safe_request[:30]  # 限制长度
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"tree_{safe_request}_{timestamp}.json"
    
    def save(self, tree_data: DecisionTreeData, filename: str = None) -> str:
        """
        保存决策树到文件
        
        Args:
            tree_data: 决策树数据
            filename: 可选的自定义文件名
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            filename = self._generate_filename(tree_data.user_request)
        
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tree_data.to_dict(), f, ensure_ascii=False, indent=2)
        
        print(f"决策树已保存到: {filepath}")
        return filepath
    
    def load(self, filepath: str) -> DecisionTreeData:
        """
        从文件加载决策树
        
        Args:
            filepath: 文件路径
            
        Returns:
            DecisionTreeData: 加载的决策树数据
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tree_data = DecisionTreeData.from_dict(data)
        print(f"决策树已从 {filepath} 加载")
        print(f"  用户请求: {tree_data.user_request}")
        print(f"  创建时间: {tree_data.created_at}")
        print(f"  POI数量: {len(tree_data.pois)}")
        return tree_data
    
    def list_saved_trees(self) -> list[dict]:
        """
        列出所有已保存的决策树
        
        Returns:
            包含文件信息的字典列表
        """
        trees = []
        if not os.path.exists(self.save_dir):
            return trees
        
        for filename in os.listdir(self.save_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.save_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    trees.append({
                        "filename": filename,
                        "filepath": filepath,
                        "user_request": data.get("user_request", "未知"),
                        "created_at": data.get("created_at", "未知"),
                        "poi_count": len(data.get("pois", []))
                    })
                except Exception as e:
                    print(f"[警告] 读取文件 {filename} 失败: {e}")
        
        # 按创建时间排序（最新的在前）
        trees.sort(key=lambda x: x["created_at"], reverse=True)
        return trees


# ============== 交互式选择器 ==============

class InteractiveSelector:
    """【新增】交互式选择器 - 负责用户交互，与决策树构建分离"""
    
    def __init__(
        self,
        tree_data: DecisionTreeData,
        large_llm_api: Callable[[str], str],
        small_llm_api: Callable[[str], str],
        agent_api: Callable[[str], str],
        storage: DecisionTreeStorage = None,
        current_filepath: str = None
    ):
        """
        初始化交互式选择器
        
        Args:
            tree_data: 决策树数据
            large_llm_api: 大参数LLM API（用于换题时重新生成）
            small_llm_api: 小参数LLM API（用于换题时重新筛选）
            agent_api: Agent API（用于回答用户问题）
            storage: 决策树存储管理器（用于换题后保存）
            current_filepath: 当前决策树的文件路径（用于换题后覆盖保存）
        """
        self.tree_data = tree_data
        self.large_llm_api = large_llm_api
        self.small_llm_api = small_llm_api
        self.agent_api = agent_api
        self.storage = storage
        self.current_filepath = current_filepath
        self.logger = ConversationLogger()
        self.logger.user_request = tree_data.user_request
        
        # 用于换题时重新构建子树
        self._tree_builder = DecisionTreeBuilder(
            large_llm_api=large_llm_api,
            small_llm_api=small_llm_api,
            max_depth=tree_data.max_depth,
            min_pois_to_continue=tree_data.min_pois_to_continue
        )
    
    def _is_notebook(self) -> bool:
        try:
            from IPython import get_ipython
            ipy = get_ipython()
            if ipy is None:
                return False
            shell = ipy.__class__.__name__
            return shell == 'ZMQInteractiveShell'
        except (ImportError, NameError, AttributeError):
            return False
    
    def _display_question(self, node: DecisionNode, can_go_back: bool = False):
        if self._is_notebook():
            self._display_question_notebook(node, can_go_back)
        else:
            self._display_question_terminal(node, can_go_back)
    
    def _display_question_terminal(self, node: DecisionNode, can_go_back: bool = False):
        print("\n" + "=" * 50)
        print(f"问题 (剩余{len(node.current_pois)}个POI):")
        print(f"  {node.question}")
        print("-" * 50)
        print(f"  A. {node.option_a}")
        print(f"  B. {node.option_b}")
        print("-" * 50)
        print(f"  C. 换一个问题")
        print(f"  D. 退出问答，直接返回所有符合条件的POI")
        if can_go_back:
            print(f"  E. 回退到上一个问题")
        print(f"  F. 输入自定义偏好信息")
        print(f"  G. 向Agent提问获取信息")
        print(f"  H. 查看当前所有候选POI")
        print("=" * 50)
    
    def _display_question_notebook(self, node: DecisionNode, can_go_back: bool = False):
        from IPython.display import display, HTML
        
        go_back_html = ""
        if can_go_back:
            go_back_html = '<p style="margin: 5px 0;"><strong>E.</strong> 回退到上一个问题</p>'
        
        html_content = f"""
        <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f9f9f9;">
            <h3 style="color: #2196F3; margin-top: 0;">📋 问题 <span style="font-size: 14px; color: #666;">(剩余 {len(node.current_pois)} 个POI)</span></h3>
            <p style="font-size: 16px; font-weight: bold; color: #333;">{node.question}</p>
            <hr style="border: 1px dashed #ddd;">
            <div style="margin: 10px 0;">
                <p style="margin: 5px 0;"><strong style="color: #4CAF50;">A.</strong> {node.option_a}</p>
                <p style="margin: 5px 0;"><strong style="color: #2196F3;">B.</strong> {node.option_b}</p>
            </div>
            <hr style="border: 1px dashed #ddd;">
            <div style="margin: 10px 0; color: #888;">
                <p style="margin: 5px 0;"><strong>C.</strong> 换一个问题</p>
                <p style="margin: 5px 0;"><strong>D.</strong> 退出问答，直接返回所有符合条件的POI</p>
                {go_back_html}
                <p style="margin: 5px 0;"><strong style="color: #FF9800;">F.</strong> 输入自定义偏好信息</p>
                <p style="margin: 5px 0;"><strong style="color: #9C27B0;">G.</strong> 向Agent提问获取信息</p>
                <p style="margin: 5px 0;"><strong style="color: #607D8B;">H.</strong> 查看当前所有候选POI</p>
            </div>
        </div>
        """
        display(HTML(html_content))
    
    def _display_result(self, pois: list[POI], show_restart: bool = True):
        if self._is_notebook():
            self._display_result_notebook(pois, show_restart)
        else:
            self._display_result_terminal(pois, show_restart)
    
    def _display_result_terminal(self, pois: list[POI], show_restart: bool = True):
        print("\n" + "=" * 50)
        print("🎯 推荐的POI结果:")
        print("=" * 50)
        if not pois:
            print("抱歉，没有找到符合条件的POI。")
        else:
            for i, poi in enumerate(pois, 1):
                print(f"{i}. {poi.name}")
                print(f"   {poi.description}")
                print()
        print("=" * 50)
        
        if show_restart:
            print("\n💡 您可以：")
            print("  R. 重新开始选择（使用同一决策树）")
            print("  Q. 退出")
    
    def _display_result_notebook(self, pois: list[POI], show_restart: bool = True):
        from IPython.display import display, HTML
        
        restart_html = ""
        if show_restart:
            restart_html = """
            <div style="margin-top: 15px; padding: 10px; background-color: #fff3e0; border-radius: 5px;">
                <p style="margin: 0; color: #e65100;">💡 <strong>您可以：</strong></p>
                <p style="margin: 5px 0 0 20px;">R. 重新开始选择（使用同一决策树）</p>
                <p style="margin: 5px 0 0 20px;">Q. 退出</p>
            </div>
            """
        
        if not pois:
            html_content = f"""
            <div style="border: 2px solid #f44336; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #ffebee;">
                <h3 style="color: #f44336;">😔 抱歉</h3>
                <p>没有找到符合条件的POI。</p>
                {restart_html}
            </div>
            """
        else:
            poi_items = ""
            for i, poi in enumerate(pois, 1):
                poi_items += f"""
                <div style="background-color: #fff; border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin: 5px 0;">
                    <h4 style="margin: 0; color: #4CAF50;">{i}. {poi.name}</h4>
                    <p style="margin: 5px 0 0 0; color: #666;">{poi.description}</p>
                </div>
                """
            
            html_content = f"""
            <div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f1f8e9;">
                <h3 style="color: #4CAF50; margin-top: 0;">🎯 推荐的POI结果 ({len(pois)}个)</h3>
                {poi_items}
                {restart_html}
            </div>
            """
        
        display(HTML(html_content))
    
    def _get_user_input(self, can_go_back: bool = False) -> UserChoice:
        if self._is_notebook():
            return self._get_user_input_notebook(can_go_back)
        else:
            return self._get_user_input_terminal(can_go_back)
    
    def _get_user_input_terminal(self, can_go_back: bool = False) -> UserChoice:
        valid_choices = ["A", "B", "C", "D", "F", "G", "H"]
        if can_go_back:
            valid_choices.insert(4, "E")  # 在D后面插入E
        
        prompt = f"请输入您的选择 ({'/'.join(valid_choices)}): "
        
        while True:
            choice = input(prompt).strip().upper()
            if choice in valid_choices:
                return UserChoice(choice)
            print(f"无效输入，请输入 {' 、'.join(valid_choices)}")
    
    def _get_user_input_notebook(self, can_go_back: bool = False) -> UserChoice:
        from IPython.display import display, HTML
        
        valid_choices = ["A", "B", "C", "D", "F", "G", "H"]
        if can_go_back:
            valid_choices.insert(4, "E")  # 在D后面插入E
        
        display(HTML(f'<p style="color: #2196F3; font-weight: bold;">请输入您的选择 ({"/".join(valid_choices)}):</p>'))
        
        while True:
            choice = input("您的选择: ").strip().upper()
            if choice in valid_choices:
                return UserChoice(choice)
            display(HTML(f'<p style="color: #f44336;">⚠️ 无效输入，请输入 {" 、".join(valid_choices)}</p>'))
    
    def _get_restart_choice(self) -> bool:
        """获取用户是否要重新选择"""
        if self._is_notebook():
            from IPython.display import display, HTML
            display(HTML('<p style="color: #2196F3; font-weight: bold;">请输入您的选择 (R/Q):</p>'))
        
        while True:
            choice = input("您的选择: ").strip().upper()
            if choice == "R":
                return True
            elif choice == "Q":
                return False
            msg = "无效输入，请输入 R（重新选择）或 Q（退出）"
            if self._is_notebook():
                from IPython.display import display, HTML
                display(HTML(f'<p style="color: #f44336;">⚠️ {msg}</p>'))
            else:
                print(msg)
    
    def _display_message(self, message: str, style: str = "info"):
        if self._is_notebook():
            from IPython.display import display, HTML
            colors = {
                "info": "#2196F3",
                "success": "#4CAF50", 
                "warning": "#FF9800",
                "error": "#f44336"
            }
            color = colors.get(style, "#333")
            display(HTML(f'<p style="color: {color};">{message}</p>'))
        else:
            print(message)
    
    def _display_history(self):
        if self._is_notebook():
            from IPython.display import display, HTML
            html_content = f"""
            <div style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin: 10px 0; background-color: #fafafa;">
                <h4 style="color: #666;">📝 问答历史记录</h4>
                <pre style="font-size: 12px; color: #555;">{self.logger.get_history_prompt()}</pre>
            </div>
            """
            display(HTML(html_content))
        else:
            print("\n📝 问答历史记录:")
            print("-" * 50)
            print(self.logger.get_history_prompt())
    
    def _regenerate_subtree(self, pois: list[POI], depth: int) -> Optional[DecisionNode]:
        """使用tree_builder重新生成子树"""
        self._tree_builder.logger = ConversationLogger()
        self._tree_builder.logger.user_request = self.tree_data.user_request
        # 复制当前历史到builder
        for record in self.logger.qa_history:
            self._tree_builder.logger.add_record(record)
        return self._tree_builder._build_subtree(pois, depth)
    
    def _update_tree_node(self, old_node: DecisionNode, new_node: DecisionNode, parent_choice: str = None):
        """
        更新决策树中的节点，并保存到文件
        
        Args:
            old_node: 被替换的旧节点
            new_node: 新生成的节点
            parent_choice: 从父节点到达此节点的选择（"A" 或 "B"），None表示是根节点
        """
        # 这个方法在换题时被调用，用于将新节点更新到tree_data中
        # 由于我们在run()中直接操作current_node，这里主要负责保存
        pass
    
    def _filter_pois_for_user_input(
        self,
        pois: list[POI],
        question: str,
        option_a: str,
        option_b: str,
        choice: str
    ) -> list[POI]:
        """
        根据用户输入或Agent回答筛选POI
        
        Args:
            pois: 当前POI列表
            question: 问题（用户提供偏好信息/用户的问题）
            option_a: 选项A（用户的偏好/Agent的回答）
            option_b: 选项B（与A相反）
            choice: 用户选择（"A" 或 "B"）
        
        Returns:
            筛选后的POI列表
        """
        chosen_option = option_a if choice == "A" else option_b
        filtered_pois = []
        
        for poi in pois:
            prompt = f"""你是一个POI筛选助手，需要判断单个POI是否符合用户的选择条件。

【用户原始请求】
{self.tree_data.user_request}

{self.logger.get_history_prompt()}

【当前问答轮次】
问题: {question}
选项A: {option_a}
选项B: {option_b}
用户选择: {choice} ({chosen_option})

【待判断的POI】
名称: {poi.name}
简介: {poi.description}

【判断任务】
请根据用户的选择，判断这个POI是否符合用户的偏好。
- 如果这个POI符合用户选择的"{chosen_option}"这一偏好，返回"保留"
- 如果这个POI不符合用户选择的偏好，返回"过滤"

请以JSON格式返回，包含以下字段:
- decision: "保留" 或 "过滤"
- reason: 简短说明判断理由（一句话）

只返回JSON，不要其他内容。
"""
            
            response = self.small_llm_api(prompt)
            try:
                result = json.loads(response)
                decision = result.get("decision", "保留")
                if decision == "保留":
                    filtered_pois.append(poi)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[警告] 判断POI '{poi.name}' 时LLM响应解析失败: {e}")
                filtered_pois.append(poi)

        # 允许返回空列表，交互时会提示用户
        return filtered_pois
    
    def _save_tree_if_needed(self):
        """如果有storage和filepath，保存决策树"""
        if self.storage and self.current_filepath:
            # 从filepath中提取文件名
            filename = os.path.basename(self.current_filepath)
            self.storage.save(self.tree_data, filename=filename)
            self._display_message("✅ 决策树已更新并保存", "success")
    
    def run(self) -> list[POI]:
        """
        运行一次交互式选择流程
        
        Returns:
            筛选后的POI列表
        """
        if self.tree_data.root is None:
            self._display_message("POI数量太少，无需问答，直接返回结果：", "info")
            self._display_result(self.tree_data.pois, show_restart=False)
            return self.tree_data.pois
        
        # 清空日志
        self.logger.clear()
        
        current_node = self.tree_data.root
        current_pois = self.tree_data.pois.copy()
        
        # 历史状态栈
        history_stack: list[tuple[DecisionNode, list[POI], Optional[QARecord]]] = []
        
        # 显示欢迎信息
        if self._is_notebook():
            from IPython.display import display, HTML
            welcome_html = f"""
            <div style="border: 2px solid #2196F3; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #e3f2fd;">
                <h2 style="color: #2196F3; margin-top: 0;">🎯 开始POI选择！</h2>
                <p><strong>您的请求:</strong> {self.tree_data.user_request}</p>
                <p><strong>待筛选POI数量:</strong> {len(self.tree_data.pois)} 个</p>
                <p style="font-size: 12px; color: #666;"><strong>决策树创建时间:</strong> {self.tree_data.created_at}</p>
            </div>
            """
            display(HTML(welcome_html))
        else:
            print(f"\n🎯 开始POI选择！")
            print(f"您的请求: {self.tree_data.user_request}")
            print(f"共有 {len(self.tree_data.pois)} 个POI待筛选")
            print(f"决策树创建时间: {self.tree_data.created_at}\n")
        
        while current_node is not None:
            can_go_back = len(history_stack) > 0
            
            self._display_question(current_node, can_go_back)
            choice = self._get_user_input(can_go_back)
            
            if choice == UserChoice.EXIT:
                self._display_message("\n您选择了退出问答。", "info")
                self._display_result(current_pois)
                return current_pois
            
            elif choice == UserChoice.GO_BACK:
                if history_stack:
                    prev_node, prev_pois, prev_record = history_stack.pop()
                    if self.logger.qa_history:
                        self.logger.qa_history.pop()
                    current_node = prev_node
                    current_pois = prev_pois
                    self._display_message("\n⬅️ 已回退到上一个问题", "warning")
                    self._display_message(f"当前剩余 {len(current_pois)} 个POI", "info")
                else:
                    self._display_message("\n已经是第一个问题，无法回退", "error")
            
            elif choice == UserChoice.CHANGE_QUESTION:
                self._display_message("\n正在生成新问题...", "warning")
                new_node = self._regenerate_subtree(current_pois, current_node.depth)
                if new_node:
                    # 更新决策树结构
                    if len(history_stack) == 0:
                        # 当前是根节点，直接替换
                        self.tree_data.root = new_node
                    else:
                        # 获取父节点和到达当前节点的选择
                        parent_node, _, parent_record = history_stack[-1]
                        if parent_record.user_choice == "A":
                            parent_node.child_a = new_node
                        else:
                            parent_node.child_b = new_node
                    
                    current_node = new_node
                    
                    # 保存更新后的决策树
                    self._save_tree_if_needed()
                else:
                    self._display_message("无法生成新问题，返回当前结果。", "error")
                    self._display_result(current_pois)
                    return current_pois
            
            elif choice == UserChoice.OPTION_A:
                # 检查选项A对应的POI集合是否为空
                if len(current_node.pois_if_a) == 0:
                    self._display_message(f"\n⚠️ 抱歉，当前候选的POI中没有符合「{current_node.option_a}」这个条件的结果", "warning")
                    self._display_message("请选择其他选项，或按E回退到上一个问题，或按C换一个问题", "info")
                    continue

                record = QARecord(
                    current_node.question,
                    current_node.option_a,
                    current_node.option_b,
                    "A"
                )
                history_stack.append((current_node, current_pois, record))
                self.logger.add_record(record)
                current_pois = current_node.pois_if_a
                current_node = current_node.child_a
                self._display_message(f"\n✅ 您选择了A: {record.option_a}", "success")
                self._display_message(f"剩余 {len(current_pois)} 个POI", "info")

            elif choice == UserChoice.OPTION_B:
                # 检查选项B对应的POI集合是否为空
                if len(current_node.pois_if_b) == 0:
                    self._display_message(f"\n⚠️ 抱歉，当前候选的POI中没有符合「{current_node.option_b}」这个条件的结果", "warning")
                    self._display_message("请选择其他选项，或按E回退到上一个问题，或按C换一个问题", "info")
                    continue

                record = QARecord(
                    current_node.question,
                    current_node.option_a,
                    current_node.option_b,
                    "B"
                )
                history_stack.append((current_node, current_pois, record))
                self.logger.add_record(record)
                current_pois = current_node.pois_if_b
                current_node = current_node.child_b
                self._display_message(f"\n✅ 您选择了B: {record.option_b}", "success")
                self._display_message(f"剩余 {len(current_pois)} 个POI", "info")
            
            elif choice == UserChoice.USER_INPUT:
                # 用户自定义输入
                self._display_message("\n📝 请输入您的偏好信息（输入完成后按回车）:", "info")
                user_text = input("您的输入: ").strip()
                
                if not user_text:
                    self._display_message("输入为空，已取消", "warning")
                    continue
                
                self._display_message(f"\n收到您的输入: {user_text}", "success")
                self._display_message("正在根据您的偏好筛选POI...", "warning")
                
                # 构造一个虚拟的问答节点
                virtual_question = "用户提供了偏好信息"
                virtual_option_a = user_text
                virtual_option_b = "与上述偏好相反"
                
                # 使用small_llm筛选POI（模拟用户选择了A）
                filtered_pois = self._filter_pois_for_user_input(
                    current_pois, virtual_question, virtual_option_a, virtual_option_b, "A"
                )
                
                # 创建记录并添加到日志（记录用户选择了A）
                input_record = QARecord(
                    question=virtual_question,
                    option_a=virtual_option_a,
                    option_b=virtual_option_b,
                    user_choice="A",
                    record_type="user_input",
                    user_input_text=user_text
                )
                
                # 保存当前状态到历史栈（用于回退）
                history_stack.append((current_node, current_pois, input_record))
                self.logger.add_record(input_record)
                
                # 更新当前POI列表
                current_pois = filtered_pois
                
                self._display_message(f"✅ 已根据您的偏好筛选，剩余 {len(current_pois)} 个POI", "success")
                
                # 检查是否需要继续
                if len(current_pois) <= self.tree_data.min_pois_to_continue:
                    self._display_message("\n🎉 POI已筛选完毕！", "success")
                    current_node = None
                    break
                
                # 生成新的子节点（不保存到文件）
                self._display_message("正在生成接下来的问题...", "warning")
                new_node = self._regenerate_subtree(current_pois, current_node.depth + 1)
                if new_node:
                    current_node = new_node
                    # 注意：这里不调用 _save_tree_if_needed()，不保存修改
                else:
                    self._display_message("无法生成新问题，返回当前结果。", "error")
                    self._display_result(current_pois)
                    return current_pois
            
            elif choice == UserChoice.ASK_AGENT:
                # 向Agent提问
                self._display_message("\n🤖 请输入您想问Agent的问题（关于这些POI的任何问题）:", "info")
                agent_question = input("您的问题: ").strip()
                
                if not agent_question:
                    self._display_message("问题为空，已取消", "warning")
                    continue
                
                self._display_message(f"\n您的问题: {agent_question}", "info")
                self._display_message("正在查询信息，请稍候...", "warning")
                
                # 调用Agent API获取回答
                try:
                    agent_answer = self.agent_api(agent_question)
                    self._display_message(f"\n🤖 Agent回答: {agent_answer}", "success")
                except Exception as e:
                    self._display_message(f"Agent查询失败: {e}", "error")
                    continue
                
                # 构造问答节点：问题是用户的问题，选项A是Agent的回答（正确答案）
                virtual_option_b = "与A选项相反"
                
                self._display_message("正在根据查询结果筛选POI...", "warning")
                
                # 使用small_llm筛选POI（Agent的回答是正确答案，自动选A）
                filtered_pois = self._filter_pois_for_user_input(
                    current_pois, agent_question, agent_answer, virtual_option_b, "A"
                )
                
                # 创建记录并添加到日志
                agent_record = QARecord(
                    question=agent_question,
                    option_a=agent_answer,
                    option_b=virtual_option_b,
                    user_choice="A",
                    record_type="agent_qa",
                    agent_question=agent_question,
                    agent_answer=agent_answer
                )
                
                # 保存当前状态到历史栈（用于回退）
                history_stack.append((current_node, current_pois, agent_record))
                self.logger.add_record(agent_record)
                
                # 更新当前POI列表
                current_pois = filtered_pois
                
                self._display_message(f"✅ 已根据查询结果筛选，剩余 {len(current_pois)} 个POI", "success")
                
                # 检查是否需要继续
                if len(current_pois) <= self.tree_data.min_pois_to_continue:
                    self._display_message("\n🎉 POI已筛选完毕！", "success")
                    current_node = None
                    break
                
                # 生成新的子节点（不保存到文件）
                self._display_message("正在生成接下来的问题...", "warning")
                new_node = self._regenerate_subtree(current_pois, current_node.depth + 1)
                if new_node:
                    current_node = new_node
                    # 注意：这里不调用 _save_tree_if_needed()，不保存修改
                else:
                    self._display_message("无法生成新问题，返回当前结果。", "error")
                    self._display_result(current_pois)
                    return current_pois
            elif choice == UserChoice.SHOW_POIS:
                # 显示当前所有候选POI
                self._display_message(f"\n📍 当前候选POI列表（共{len(current_pois)}个）:", "info")
                for i, poi in enumerate(current_pois, 1):
                    self._display_message(f"  {i}. {poi.name}: {poi.description}", "info")
                # 不改变状态，继续循环显示当前问题
                continue
            if len(current_pois) <= self.tree_data.min_pois_to_continue:
                self._display_message("\n🎉 POI已筛选完毕！", "success")
                break
        
        self._display_result(current_pois)
        self._display_history()
        
        return current_pois
    
    def run_with_restart(self) -> list[POI]:
        """
        【新增】运行交互式选择，支持选择完成后重新开始
        
        Returns:
            最后一次选择的POI列表
        """
        while True:
            result = self.run()
            
            # 询问是否重新选择
            if self._get_restart_choice():
                self._display_message("\n🔄 重新开始选择...\n", "info")
                continue
            else:
                self._display_message("\n👋 感谢使用，再见！", "info")
                return result