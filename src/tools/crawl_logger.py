"""
爬虫日志记录模块 - 记录爬取过程中的详细信息，用于后续分析和优化
"""
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse


@dataclass
class CrawlLogEntry:
    """单次爬取日志记录"""
    timestamp: str              # 爬取时间 (ISO 格式)
    url: str                    # 目标 URL
    success: bool               # 是否成功
    error_type: Optional[str]   # 错误类型 (timeout/network/blocked/parse_error/unknown)
    error_message: Optional[str] # 错误详细信息
    status_code: Optional[int]  # HTTP 状态码 (如果能获取)
    response_time: float        # 响应时间（秒）
    content_length: Optional[int]   # 内容长度
    retry_count: int            # 重试次数
    domain: str                 # 域名（便于分析哪些站点问题多）

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return asdict(self)

    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class CrawlLogger:
    """爬虫日志管理器 - 实时写入 JSON Lines 文件"""

    def __init__(self, log_dir: str = "./data/crawl_logs"):
        """
        初始化日志管理器

        Args:
            log_dir: 日志文件存储目录
        """
        self.log_dir = log_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now().isoformat()

        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)

        # 创建会话日志文件 (JSON Lines 格式)
        self.log_file_path = os.path.join(log_dir, f"crawl_log_{self.session_id}.jsonl")
        self.summary_file_path = os.path.join(log_dir, f"crawl_log_{self.session_id}_summary.json")

        # 统计数据
        self._total_requests = 0
        self._success_count = 0
        self._failure_count = 0
        self._error_breakdown: dict[str, int] = {}
        self._domain_stats: dict[str, dict] = {}

        # 打开日志文件
        self._file = open(self.log_file_path, 'a', encoding='utf-8')

    def log(self, entry: CrawlLogEntry) -> None:
        """
        实时追加一条日志到文件

        Args:
            entry: 爬取日志条目
        """
        # 写入 JSON Lines 文件
        self._file.write(entry.to_json() + '\n')
        self._file.flush()  # 确保实时写入

        # 更新统计数据
        self._total_requests += 1
        if entry.success:
            self._success_count += 1
        else:
            self._failure_count += 1
            # 记录错误类型
            error_type = entry.error_type or "unknown"
            self._error_breakdown[error_type] = self._error_breakdown.get(error_type, 0) + 1

        # 更新域名统计
        domain = entry.domain
        if domain not in self._domain_stats:
            self._domain_stats[domain] = {"total": 0, "failed": 0}
        self._domain_stats[domain]["total"] += 1
        if not entry.success:
            self._domain_stats[domain]["failed"] += 1

    def get_session_stats(self) -> dict:
        """
        获取当前会话统计

        Returns:
            包含统计信息的字典
        """
        success_rate = (
            self._success_count / self._total_requests
            if self._total_requests > 0 else 0.0
        )
        return {
            "session_id": self.session_id,
            "total_requests": self._total_requests,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "success_rate": round(success_rate, 4),
            "error_breakdown": self._error_breakdown.copy(),
            "domain_stats": self._domain_stats.copy(),
        }

    def close(self) -> None:
        """关闭会话，写入统计摘要"""
        # 关闭日志文件
        if self._file:
            self._file.close()
            self._file = None

        # 生成并写入摘要文件
        summary = {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": datetime.now().isoformat(),
            "total_requests": self._total_requests,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "success_rate": round(
                self._success_count / self._total_requests
                if self._total_requests > 0 else 0.0, 4
            ),
            "error_breakdown": self._error_breakdown,
            "domain_stats": self._domain_stats,
        }

        with open(self.summary_file_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


def classify_error(error: Exception) -> tuple[str, str]:
    """
    对异常进行分类

    Args:
        error: 捕获的异常

    Returns:
        (error_type, error_message) 元组
    """
    error_str = str(error).lower()
    error_type_name = type(error).__name__

    # 超时错误
    if "timeout" in error_str or "TimeoutError" in error_type_name:
        return "timeout", str(error)

    # 网络错误
    if any(keyword in error_str for keyword in [
        "network", "connection", "refused", "reset", "dns", "unreachable"
    ]):
        return "network", str(error)

    # 被阻止/反爬虫
    if any(keyword in error_str for keyword in [
        "403", "forbidden", "blocked", "captcha", "rate limit", "too many requests"
    ]):
        return "blocked", str(error)

    # 解析错误
    if any(keyword in error_str for keyword in [
        "parse", "decode", "encoding", "invalid", "malformed"
    ]):
        return "parse_error", str(error)

    # 未知错误
    return "unknown", str(error)


def extract_domain(url: str) -> str:
    """
    从 URL 中提取域名

    Args:
        url: 完整 URL

    Returns:
        域名字符串
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc or "unknown"
    except Exception:
        return "unknown"
