from src.selector.selector import *
from src.search.process import search_process
from src.search.gen_advice import gen_advice
from src.agent.react import Answer_ReActAgent
from src.tools.tools import tools
from src.core.basellm import base_llm
import json
from datetime import datetime
import re
import argparse

class POISelector:
    """POI选择器主类 - 整合构建、存储、交互功能"""
    
    def __init__(
        self,
        large_llm_api: Callable[[str], str],
        small_llm_api: Callable[[str], str],
        agent_api: Callable[[str], str],
        max_depth: int = 10,
        min_pois_to_continue: int = 1,
        save_dir: str = None
    ):
        self.large_llm_api = large_llm_api
        self.small_llm_api = small_llm_api
        self.agent_api = agent_api
        self.max_depth = max_depth
        self.min_pois_to_continue = min_pois_to_continue
        
        # 初始化各组件
        self.builder = DecisionTreeBuilder(
            large_llm_api=large_llm_api,
            small_llm_api=small_llm_api,
            max_depth=max_depth,
            min_pois_to_continue=min_pois_to_continue
        )
        self.storage = DecisionTreeStorage(save_dir=save_dir)
        
        # 当前决策树数据和文件路径
        self.current_tree: Optional[DecisionTreeData] = None
        self.current_filepath: Optional[str] = None
    
    def _is_notebook(self) -> bool:
        try:
            from IPython import get_ipython
            ipy = get_ipython()
            if ipy is None:
                return False
            return ipy.__class__.__name__ == 'ZMQInteractiveShell'
        except (ImportError, NameError, AttributeError):
            return False
    
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
    
    def build_tree(self, pois: list[POI], user_request: str, auto_save: bool = True) -> DecisionTreeData:
        """
        构建新的决策树
        
        Args:
            pois: POI列表
            user_request: 用户请求
            auto_save: 是否自动保存
            
        Returns:
            构建的决策树数据
        """
        self.current_tree = self.builder.build(pois, user_request)
        
        if auto_save:
            self.current_filepath = self.storage.save(self.current_tree)
        else:
            self.current_filepath = None
        
        return self.current_tree
    
    def load_tree(self, filepath: str) -> DecisionTreeData:
        """
        加载已保存的决策树
        
        Args:
            filepath: 文件路径
            
        Returns:
            加载的决策树数据
        """
        self.current_tree = self.storage.load(filepath)
        self.current_filepath = filepath
        return self.current_tree
    
    def list_saved_trees(self) -> list[dict]:
        """列出所有已保存的决策树"""
        return self.storage.list_saved_trees()
    
    def run_interactive(
        self, 
        pois: list[POI] = None, 
        user_request: str = None,
        tree_data: DecisionTreeData = None,
        allow_restart: bool = True
    ) -> list[POI]:
        """
        运行交互式选择流程
        
        Args:
            pois: POI列表（如果提供tree_data则忽略）
            user_request: 用户请求（如果提供tree_data则忽略）
            tree_data: 已有的决策树数据（优先使用）
            allow_restart: 是否允许选择后重新开始
            
        Returns:
            筛选后的POI列表
        """
        # 确定使用的决策树
        if tree_data is not None:
            self.current_tree = tree_data
        elif self.current_tree is None:
            if pois is None or user_request is None:
                raise ValueError("必须提供pois和user_request，或者提供tree_data")
            self.build_tree(pois, user_request)
        
        # 创建交互式选择器
        selector = InteractiveSelector(
            tree_data=self.current_tree,
            large_llm_api=self.large_llm_api,
            small_llm_api=self.small_llm_api,
            agent_api=self.agent_api,
            storage=self.storage,
            current_filepath=self.current_filepath
        )
        
        # 运行选择
        if allow_restart:
            return selector.run_with_restart()
        else:
            return selector.run()
    
    def show_main_menu(self):
        """【新增】显示主菜单，让用户选择操作"""
        if self._is_notebook():
            from IPython.display import display, HTML
            html_content = """
            <div style="border: 2px solid #9C27B0; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #f3e5f5;">
                <h2 style="color: #9C27B0; margin-top: 0;">🎯 POI-Scout</h2>
                <p>请选择操作：</p>
                <div style="margin: 10px 0;">
                    <p><strong>1.</strong> 创建新的POI搜索任务</p>
                    <p><strong>2.</strong> 加载已保存的搜索结果并进行筛选</p>
                    <p><strong>3.</strong> 查看所有已保存的搜索任务结果</p>
                    <p><strong>Q.</strong> 退出</p>
                </div>
            </div>
            """
            display(HTML(html_content))
        else:
            print("\n" + "=" * 50)
            print("🎯 POI选择系统")
            print("=" * 50)
            print("请选择操作：")
            print("  1. 创建新的POI搜索任务")
            print("  2. 加载已保存的搜索结果并进行筛选")
            print("  3. 查看所有已保存的搜索任务结果")
            print("  Q. 退出")
            print("=" * 50)
    
    def _display_saved_trees(self):
        """显示已保存的决策树列表"""
        trees = self.list_saved_trees()
        
        if not trees:
            self._display_message("暂无已保存的搜索任务。", "warning")
            return None
        
        if self._is_notebook():
            from IPython.display import display, HTML
            items_html = ""
            for i, tree in enumerate(trees, 1):
                items_html += f"""
                <div style="background-color: #fff; border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin: 5px 0;">
                    <p style="margin: 0;"><strong>{i}.</strong> {tree['user_request']}</p>
                    <p style="margin: 5px 0 0 20px; font-size: 12px; color: #666;">
                        创建时间: {tree['created_at']} | POI数量: {tree['poi_count']}
                    </p>
                    <p style="margin: 0 0 0 20px; font-size: 11px; color: #999;">
                        文件: {tree['filename']}
                    </p>
                </div>
                """
            
            html_content = f"""
            <div style="border: 2px solid #2196F3; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #e3f2fd;">
                <h3 style="color: #2196F3; margin-top: 0;">📁 已保存的
                
                
                ({len(trees)}个)</h3>
                {items_html}
            </div>
            """
            display(HTML(html_content))
        else:
            print("\n📁 已保存的搜索任务:")
            print("-" * 50)
            for i, tree in enumerate(trees, 1):
                print(f"{i}. {tree['user_request']}")
                print(f"   创建时间: {tree['created_at']} | POI数量: {tree['poi_count']}")
                print(f"   文件: {tree['filename']}")
                print()
        
        return trees
    
    def run_with_menu(self,online_opt,opt_iterations, use_skill, create_skill):
        """
        【新增】运行带菜单的完整流程
        
        Args:
            default_pois: 默认的POI列表（创建新决策树时使用）
        """
        while True:
            self.show_main_menu()
            
            choice = input("请输入您的选择: ").strip().upper()
            
            if choice == "Q":
                self._display_message("\n👋 感谢使用，再见！", "info")
                break
            
            elif choice == "1":
                user_request = input("请输入您的需求（如：我想找一个适合周末去玩的地方）: ").strip()
                if not user_request:
                    self._display_message("需求不能为空", "error")
                    continue
                self._display_message("已接收需求,搜索候选POI中(可能需要花费较长时间)...","info")
                log,log_ref=search_process(user_request,on_policy_opt=online_opt,maximum_opt_iterations=opt_iterations,
                                           use_advice=use_skill)
                
                self._display_message("总结搜索经验中,后续搜索中可能会用到这些经验...","info")
                if create_skill:
                    self._display_message(log['任务总结'],"info")
                    match = re.search(r'搜索到(\d+)个候选POI', log['任务总结'])
                    poi_count = int(match.group(1))
                    match1 = re.search(r'搜索到(\d+)个候选POI', log_ref['任务总结'])
                    poi_count_1 = int(match1.group(1))
                    if poi_count > poi_count_1*1.5:
                        gen_advices=gen_advice(topic=user_request,log=log,log_ref=log_ref)
                        with open('data/skills.json', 'r', encoding='utf-8') as f:
                            advices = json.load(f)
                            advices.append({
                                    "title": f"{user_request}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                    "content": ["如果搜索"+i['POI类别']+":"+i['经验总结'] for i in gen_advices]
                            })
                        with open('data/skills.json', 'w', encoding='utf-8') as f:
                            json.dump(advices, f, ensure_ascii=False, indent=2)
                        self._display_message(f"本次任务经验写入data/skills.json中, title:{user_request}_{datetime.now().strftime('%Y%m%d_%H%M%S')}...","info")
                    else:
                        self._display_message("在线优化后提升不明显，本次任务不总结经验...","info")
                ###获取log结果里的poi_list
                POI_list=[]
                for item in log['执行日志']:
                    POIs=item['搜索POI']
                    for POI_item in POIs:
                        if POI_item.endswith('是'):
                            poi=POI(POI_item.split(',')[0],POI_item.split(',')[1][3:])
                            POI_list.append(poi)
                pois=list(set(POI_list))
                
                ###build决策树###
                self.build_tree(pois, user_request)
                self.run_interactive(allow_restart=True)
            
            elif choice == "2":
                # 加载已保存的决策树
                trees = self._display_saved_trees()
                if trees is None:
                    continue
                
                try:
                    idx = int(input("请输入要加载的搜索任务编号: ").strip()) - 1
                    if 0 <= idx < len(trees):
                        self.load_tree(trees[idx]['filepath'])
                        self.run_interactive(allow_restart=True)
                    else:
                        self._display_message("无效的编号", "error")
                except ValueError:
                    self._display_message("请输入有效的数字", "error")
            
            elif choice == "3":
                # 查看已保存的决策树
                self._display_saved_trees()
                input("按回车键继续...")
            
            else:
                self._display_message("无效的选择，请重试", "error")


# ============== 示例使用 ==============


def demo_menu_mode(online_opt,opt_iterations, use_skill, create_skill):
    """演示菜单模式"""
    llm=base_llm(system_prompt="")
    agent=Answer_ReActAgent(llm,tools=tools)
    selector = POISelector(
        max_depth=5,
        min_pois_to_continue=2,
        large_llm_api=llm.call_with_messages_R1,
        small_llm_api=llm.call_with_messages_small,
        agent_api=agent.run
        
    )
    
    # 运行带菜单的完整流程
    selector.run_with_menu(online_opt,opt_iterations, use_skill, create_skill)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="POI-Scout: 智能旅行目的地搜索Agent")
    
    # 布尔参数（使用 store_true，默认为 False，传入时为 True）
    parser.add_argument('--online-opt', action='store_true', 
                        help='启用Online在线优化模式')
    parser.add_argument('--use-skill', action='store_true', 
                        help='从历史经验中使用建议，存储在data/skills.json文件中')
    parser.add_argument('--create-skill', action='store_true', 
                        help='基于本次任务和线上优化结果生成建议，供以后借鉴')
    
    # int 参数（可选，带默认值）
    parser.add_argument('--opt-iterations', type=int, default=10, 
                        help='在线优化迭代次数（默认: 10）')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("POI-Scout: 智能旅行目的地搜索Agent")
    print("=" * 60)
    print(f"\n参数设置：")
    print(f"  online_opt: {args.online_opt}")
    print(f"  opt_iterations: {args.opt_iterations}")
    print(f"  use_skill: {args.use_skill}")
    print(f"  create_skill: {args.create_skill}")
    print("=" * 60)
    
    demo_menu_mode(args.online_opt, args.opt_iterations, args.use_skill, args.create_skill)

if __name__ == "__main__":
    main()
