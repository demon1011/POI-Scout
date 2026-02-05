"""
内容提取器 - 从 HTML 中提取结构化内容

支持两种提取模式：
1. ContentExtractor - 基础提取，适用于简单页面
2. SmartExtractor - 智能提取，支持：
   - 基于文本密度的主内容识别
   - 噪声过滤（广告、导航、推荐）
   - 结构化数据提取（标题、作者、日期）
"""
import re
from dataclasses import dataclass, field
from typing import Optional
from bs4 import BeautifulSoup, Tag, NavigableString
from urllib.parse import urljoin


@dataclass
class ExtractedContent:
    """提取的内容数据类"""
    title: Optional[str] = None
    text: Optional[str] = None
    images: list[str] = None
    links: list[str] = None

    def __post_init__(self):
        if self.images is None:
            self.images = []
        if self.links is None:
            self.links = []


@dataclass
class ArticleData:
    """结构化文章数据"""
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    content: Optional[str] = None
    summary: Optional[str] = None
    images: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "author": self.author,
            "date": self.date,
            "content": self.content,
            "summary": self.summary,
            "images": self.images,
            "links": self.links,
        }


class ContentExtractor:
    """HTML 内容提取器"""

    # 常见的内容区域选择器
    CONTENT_SELECTORS = [
        "article",
        "main",
        "[role='main']",
        ".content",
        ".post-content",
        ".article-content",
        ".article-body",
        ".entry-content",
        "#content",
        "#main-content",
        "#mw-content-text",  # Wikipedia
    ]

    # 需要过滤的标签
    NOISE_TAGS = [
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        "noscript",
        "iframe",
        "form",
    ]

    def __init__(self, base_url: str = ""):
        """
        初始化提取器

        Args:
            base_url: 基础 URL，用于转换相对链接
        """
        self.base_url = base_url

    def extract_from_html(self, html: str) -> ExtractedContent:
        """
        从 HTML 字符串提取内容

        Args:
            html: HTML 字符串

        Returns:
            ExtractedContent: 提取的内容
        """
        soup = BeautifulSoup(html, "lxml")

        # 移除噪声标签
        for tag in self.NOISE_TAGS:
            for element in soup.find_all(tag):
                element.decompose()

        return ExtractedContent(
            title=self._extract_title(soup),
            text=self._extract_text(soup),
            images=self._extract_images(soup),
            links=self._extract_links(soup),
        )

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """提取页面标题"""
        # 尝试从 title 标签获取
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            return title_tag.string.strip()

        # 尝试从 h1 标签获取
        h1_tag = soup.find("h1")
        if h1_tag:
            return h1_tag.get_text(strip=True)

        return None

    def _extract_text(self, soup: BeautifulSoup) -> Optional[str]:
        """提取主要文本内容"""
        # 尝试找到主要内容区域
        for selector in self.CONTENT_SELECTORS:
            # 处理类选择器
            if selector.startswith("."):
                element = soup.find(class_=selector[1:])
            # 处理 ID 选择器
            elif selector.startswith("#"):
                element = soup.find(id=selector[1:])
            # 处理属性选择器
            elif selector.startswith("["):
                # 简单处理 role 属性
                if "role=" in selector:
                    role = selector.split("=")[1].strip("']\"")
                    element = soup.find(attrs={"role": role})
                else:
                    element = None
            # 处理标签选择器
            else:
                element = soup.find(selector)

            if element:
                text = element.get_text(separator="\n", strip=True)
                if text and len(text) > 100:
                    return text

        # 回退到 body
        body = soup.find("body")
        if body:
            return body.get_text(separator="\n", strip=True)

        return None

    def _extract_images(self, soup: BeautifulSoup) -> list[str]:
        """提取所有图片 URL"""
        images = []
        for img in soup.find_all("img"):
            src = img.get("src")
            if src:
                absolute_url = urljoin(self.base_url, src)
                images.append(absolute_url)
        return images

    def _extract_links(self, soup: BeautifulSoup) -> list[str]:
        """提取所有链接"""
        links = []
        for a in soup.find_all("a"):
            href = a.get("href")
            if href and not href.startswith(("#", "javascript:", "mailto:", "tel:")):
                absolute_url = urljoin(self.base_url, href)
                links.append(absolute_url)
        return links


class SmartExtractor:
    """
    智能内容提取器 - 基于文本密度和语义分析

    功能:
    - 识别文章主体内容区域
    - 过滤导航栏、侧边栏、页脚
    - 过滤广告和推荐内容
    - 提取结构化数据（标题、作者、日期、正文）
    """

    # 需要完全移除的标签
    REMOVE_TAGS = [
        "script", "style", "noscript", "iframe", "svg",
        "canvas", "video", "audio", "map", "object", "embed",
    ]

    # 噪声区域的选择器（类名、ID）
    # 使用更精确的模式避免误匹配
    NOISE_PATTERNS = [
        # 导航相关 - 使用边界匹配
        r"\bnav\b", r"\bmenu\b", r"\bnavigation\b", r"\bnavbar\b", r"\bheader-nav\b",
        # 侧边栏
        r"\bsidebar\b", r"\bside-bar\b", r"\baside\b", r"\bwidget\b",
        # 页脚
        r"\bfooter\b", r"\bfoot\b",
        # 广告 - 使用边界避免误匹配 "loading" 等词
        r"\bad\b", r"\bads\b", r"\badvert", r"\badvertisement\b", r"\bsponsor",
        r"\bbanner\b", r"\bpromo\b", r"\bpromotion\b",
        # 推荐/相关内容
        r"\brelated-", r"\brecommend", r"\bsuggestion\b", r"\bmore-from\b",
        r"\balso-like\b", r"\byou-may\b",
        # 评论区
        r"\bcomment", r"\bdiscuss", r"\breply\b", r"\bresponses?\b",
        # 社交分享
        r"\bshare\b", r"\bsocial\b", r"\bfollow\b",
        # 订阅/注册
        r"\bsubscribe\b", r"\bnewsletter\b", r"\bsignup\b", r"\bregister\b",
        # 弹窗/模态框
        r"\bmodal\b", r"\bpopup\b", r"\boverlay\b", r"\bdialog\b",
        # 面包屑
        r"\bbreadcrumb\b",
        # 搜索框
        r"\bsearch-box\b", r"\bsearchform\b",
        # 工具栏
        r"\btoolbar\b", r"\btool-bar\b",
    ]

    # 保护模式 - 匹配这些的元素不会被删除
    PROTECTED_PATTERNS = [
        r"mw-content",  # Wikipedia 内容区
        r"article",
        r"post-content",
        r"entry-content",
        r"main-content",
        r"page-content",
        r"story-body",
    ]

    # 文章内容区域的选择器（优先级从高到低）
    CONTENT_SELECTORS = [
        # 语义化标签
        "article",
        "[role='article']",
        "[itemtype*='Article']",
        # 常见类名
        ".post-content",
        ".article-content",
        ".article-body",
        ".entry-content",
        ".post-body",
        ".story-body",
        ".news-content",
        ".content-body",
        ".main-content",
        ".page-content",
        # 常见 ID
        "#article",
        "#content",
        "#main-content",
        "#post-content",
        "#story",
        # Wikipedia 特殊
        "#mw-content-text",
        ".mw-parser-output",
        # 通用回退
        "main",
        "[role='main']",
    ]

    # 标题选择器
    TITLE_SELECTORS = [
        "h1.title",
        "h1.post-title",
        "h1.article-title",
        "h1.entry-title",
        ".headline h1",
        "article h1",
        "main h1",
        "h1",
    ]

    # 作者选择器
    AUTHOR_SELECTORS = [
        "[rel='author']",
        "[itemprop='author']",
        ".author",
        ".byline",
        ".by-author",
        ".post-author",
        ".article-author",
        ".writer",
        "a[href*='/author/']",
        "a[href*='/authors/']",
    ]

    # 日期选择器
    DATE_SELECTORS = [
        "time[datetime]",
        "[itemprop='datePublished']",
        "[itemprop='dateModified']",
        ".date",
        ".post-date",
        ".published",
        ".pub-date",
        ".article-date",
        ".entry-date",
        ".timestamp",
        ".time",
    ]

    # 日期正则模式
    DATE_PATTERNS = [
        # ISO 格式: 2024-01-15, 2024/01/15
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        # 美式: January 15, 2024
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4}",
        # 欧式: 15 January 2024
        r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}",
        # 中文: 2024年1月15日
        r"\d{4}年\d{1,2}月\d{1,2}日",
    ]

    def __init__(self, base_url: str = ""):
        self.base_url = base_url
        self._noise_pattern = re.compile(
            "|".join(self.NOISE_PATTERNS),
            re.IGNORECASE
        )
        self._protected_pattern = re.compile(
            "|".join(self.PROTECTED_PATTERNS),
            re.IGNORECASE
        )

    def extract(self, html: str) -> ArticleData:
        """
        从 HTML 中智能提取文章内容

        Args:
            html: HTML 字符串

        Returns:
            ArticleData: 结构化文章数据
        """
        soup = BeautifulSoup(html, "lxml")

        # 第一步：移除脚本和样式
        self._remove_tags(soup)

        # 第二步：提取元数据（在移除噪声前）
        title = self._extract_title(soup)
        author = self._extract_author(soup)
        date = self._extract_date(soup)

        # 保存一份去除脚本/样式后但未去噪的 body 文本，用于最终兜底
        body_fallback = soup.find("body")
        fallback_text = None
        if body_fallback:
            fallback_text = body_fallback.get_text(separator="\n", strip=True)

        # 第三步：移除噪声区域
        self._remove_noise(soup)

        # 第四步：找到主内容区域
        main_content = self._find_main_content(soup)

        # 第五步：提取内容
        content = self._extract_text(main_content) if main_content else None

        # 第六步：兜底 — 如果智能提取的内容过短，回退到 body 全文
        if (not content or len(content) <= 100) and fallback_text and len(fallback_text) > 100:
            content = fallback_text

        summary = self._generate_summary(content) if content else None
        images = self._extract_images(main_content or soup)
        links = self._extract_links(main_content or soup)

        return ArticleData(
            title=title,
            author=author,
            date=date,
            content=content,
            summary=summary,
            images=images,
            links=links,
        )

    def _remove_tags(self, soup: BeautifulSoup) -> None:
        """移除不需要的标签"""
        for tag in self.REMOVE_TAGS:
            for element in soup.find_all(tag):
                element.decompose()

    def _remove_noise(self, soup: BeautifulSoup) -> None:
        """移除噪声区域"""
        # 先收集需要删除的元素，避免在遍历时修改
        elements_to_remove = []
        for element in soup.find_all(True):
            if self._is_noise_element(element):
                elements_to_remove.append(element)

        # 然后删除
        for element in elements_to_remove:
            try:
                element.decompose()
            except Exception:
                pass  # 元素可能已被父元素删除

    def _is_noise_element(self, element: Tag) -> bool:
        """判断元素是否为噪声"""
        if not isinstance(element, Tag):
            return False

        # 检查元素是否还在文档中
        if element.parent is None:
            return False

        # 永远不删除关键结构元素
        tag_name = element.name
        if tag_name in ["html", "body", "head", "main", "article"]:
            return False

        # 获取 class 和 id
        classes = element.get("class") or []
        if isinstance(classes, list):
            classes = " ".join(classes)
        element_id = element.get("id") or ""
        combined = f"{classes} {element_id}"

        # 检查是否是受保护的元素
        if self._protected_pattern.search(combined):
            return False

        # 检查是否是噪声标签
        if tag_name in ["nav", "aside", "footer"]:
            return True

        # header 需要特殊处理，可能是文章头部
        if tag_name == "header":
            # 如果 header 在 article 内，保留它
            if element.find_parent("article"):
                return False
            return True

        # 检查 class 和 id 是否匹配噪声模式
        if self._noise_pattern.search(combined):
            return True

        # 检查 role 属性
        role = (element.get("role") or "").lower()
        if role in ["navigation", "banner", "complementary", "contentinfo"]:
            return True

        return False

    def _find_main_content(self, soup: BeautifulSoup) -> Optional[Tag]:
        """找到主内容区域"""
        # 尝试使用选择器
        for selector in self.CONTENT_SELECTORS:
            try:
                if selector.startswith("."):
                    element = soup.find(class_=selector[1:])
                elif selector.startswith("#"):
                    element = soup.find(id=selector[1:])
                elif selector.startswith("["):
                    # 处理属性选择器
                    element = self._find_by_attribute(soup, selector)
                else:
                    element = soup.find(selector)

                if element and self._is_good_content(element):
                    return element
            except Exception:
                continue

        # 回退：使用文本密度算法
        return self._find_by_text_density(soup)

    def _find_by_attribute(self, soup: BeautifulSoup, selector: str) -> Optional[Tag]:
        """根据属性选择器查找元素"""
        # 解析 [attr='value'] 或 [attr*='value']
        match = re.match(r"\[(\w+)(\*?=)?'?([^'\]]+)'?\]", selector)
        if not match:
            return None

        attr, op, value = match.groups()
        if op == "*=":
            return soup.find(attrs={attr: re.compile(value, re.I)})
        elif op == "=":
            return soup.find(attrs={attr: value})
        else:
            return soup.find(attrs={attr: True})

    def _is_good_content(self, element: Tag) -> bool:
        """判断元素是否为好的内容区域"""
        text = element.get_text(strip=True)
        # 内容长度要足够
        if len(text) < 200:
            return False

        # 计算链接密度（链接文本 / 总文本）
        link_text_len = sum(len(a.get_text(strip=True)) for a in element.find_all("a"))
        if len(text) > 0:
            link_density = link_text_len / len(text)
            # 链接密度太高说明是导航区域
            if link_density > 0.5:
                return False

        return True

    def _find_by_text_density(self, soup: BeautifulSoup) -> Optional[Tag]:
        """使用文本密度算法找到主内容区域"""
        body = soup.find("body")
        if not body:
            return None

        best_element = None
        best_score = 0

        # 遍历所有块级元素
        for element in body.find_all(["div", "section", "article", "main"]):
            score = self._calculate_content_score(element)
            if score > best_score:
                best_score = score
                best_element = element

        return best_element

    def _calculate_content_score(self, element: Tag) -> float:
        """计算元素的内容得分"""
        text = element.get_text(strip=True)
        text_len = len(text)

        if text_len < 100:
            return 0

        # 基础分：文本长度
        score = text_len

        # 加分：段落数量
        p_count = len(element.find_all("p"))
        score += p_count * 50

        # 加分：有标题结构
        heading_count = len(element.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]))
        score += heading_count * 30

        # 减分：链接密度
        link_text_len = sum(len(a.get_text(strip=True)) for a in element.find_all("a"))
        if text_len > 0:
            link_density = link_text_len / text_len
            score *= (1 - link_density)

        # 减分：嵌套过深的 div
        div_count = len(element.find_all("div"))
        if div_count > 20:
            score *= 0.8

        return score

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """提取文章标题"""
        # 尝试选择器
        for selector in self.TITLE_SELECTORS:
            try:
                if selector.startswith("."):
                    element = soup.select_one(selector)
                else:
                    element = soup.find(selector.split()[0])
                    if len(selector.split()) > 1:
                        element = soup.select_one(selector)

                if element:
                    title = element.get_text(strip=True)
                    if title and len(title) > 3:
                        return title
            except Exception:
                continue

        # 回退到 title 标签
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
            # 清理常见后缀
            for sep in [" | ", " - ", " :: ", " — ", " · "]:
                if sep in title:
                    title = title.split(sep)[0].strip()
            return title

        return None

    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """提取作者信息"""
        for selector in self.AUTHOR_SELECTORS:
            try:
                if selector.startswith("["):
                    element = soup.select_one(selector)
                elif selector.startswith("."):
                    element = soup.find(class_=selector[1:])
                elif selector.startswith("a["):
                    element = soup.select_one(selector)
                else:
                    element = soup.find(selector)

                if element:
                    author = element.get_text(strip=True)
                    # 清理常见前缀
                    for prefix in ["By ", "by ", "Author: ", "Written by ", "作者："]:
                        if author.startswith(prefix):
                            author = author[len(prefix):]
                    if author and len(author) > 1 and len(author) < 100:
                        return author
            except Exception:
                continue

        return None

    def _extract_date(self, soup: BeautifulSoup) -> Optional[str]:
        """提取发布日期"""
        # 尝试选择器
        for selector in self.DATE_SELECTORS:
            try:
                if selector.startswith("["):
                    element = soup.select_one(selector)
                elif selector.startswith("."):
                    element = soup.find(class_=selector[1:])
                else:
                    element = soup.find(selector)

                if element:
                    # 优先使用 datetime 属性
                    date = element.get("datetime")
                    if date:
                        return date[:10] if len(date) > 10 else date

                    # 从文本中提取
                    text = element.get_text(strip=True)
                    extracted_date = self._parse_date_from_text(text)
                    if extracted_date:
                        return extracted_date
            except Exception:
                continue

        return None

    def _parse_date_from_text(self, text: str) -> Optional[str]:
        """从文本中解析日期"""
        for pattern in self.DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    def _extract_text(self, element: Tag) -> Optional[str]:
        """从元素中提取清理后的文本"""
        if not element:
            return None

        # 获取所有文本段落
        paragraphs = []

        for p in element.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]):
            text = p.get_text(strip=True)
            if text and len(text) > 20:
                paragraphs.append(text)

        if paragraphs:
            return "\n\n".join(paragraphs)

        # 回退到整体文本
        text = element.get_text(separator="\n", strip=True)
        # 清理多余空白行
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines) if lines else None

    def _generate_summary(self, content: str) -> Optional[str]:
        """生成内容摘要（取前几段）"""
        if not content:
            return None

        # 取前 500 个字符作为摘要
        if len(content) <= 500:
            return content

        # 找到合适的断点
        summary = content[:500]
        last_period = max(
            summary.rfind("。"),
            summary.rfind("."),
            summary.rfind("！"),
            summary.rfind("!"),
        )
        if last_period > 200:
            return summary[:last_period + 1]

        return summary + "..."

    def _extract_images(self, element: Tag) -> list[str]:
        """从内容区域提取图片"""
        images = []
        for img in element.find_all("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            if src:
                # 过滤小图标和跟踪像素
                width = img.get("width", "")
                height = img.get("height", "")
                try:
                    if (width and int(width) < 50) or (height and int(height) < 50):
                        continue
                except ValueError:
                    pass

                # 过滤常见的无意义图片
                if any(x in src.lower() for x in ["pixel", "beacon", "tracking", "ad.", "ads/", "logo"]):
                    continue

                absolute_url = urljoin(self.base_url, src)
                if absolute_url not in images:
                    images.append(absolute_url)

        return images

    def _extract_links(self, element: Tag) -> list[str]:
        """从内容区域提取链接"""
        links = []
        for a in element.find_all("a"):
            href = a.get("href")
            if href and not href.startswith(("#", "javascript:", "mailto:", "tel:")):
                # 过滤社交分享链接
                if any(x in href.lower() for x in ["share", "facebook.com/share", "twitter.com/intent", "linkedin.com/share"]):
                    continue

                absolute_url = urljoin(self.base_url, href)
                if absolute_url not in links:
                    links.append(absolute_url)

        return links
