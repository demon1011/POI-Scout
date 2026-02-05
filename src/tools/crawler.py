"""
核心爬虫类 - 使用 Playwright 加载和爬取网页
"""
import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Optional, Literal
from urllib.parse import urljoin

from playwright.async_api import async_playwright, Page, Browser, BrowserContext

from src.tools.extractor import SmartExtractor, ArticleData
from src.tools.crawl_logger import CrawlLogger, CrawlLogEntry, classify_error, extract_domain

logger = logging.getLogger(__name__)

# 常用 User-Agent 列表
USER_AGENTS = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    # Chrome on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    # Firefox on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Safari on macOS
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    # Edge on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
]

# Cookie 同意弹窗的常见选择器和文本
COOKIE_CONSENT_SELECTORS = [
    # 常见的按钮选择器
    "[data-testid='cookie-policy-manage-dialog-accept-button']",
    "[data-testid='accept-cookies']",
    "button[id*='accept']",
    "button[id*='consent']",
    "button[class*='accept']",
    "button[class*='consent']",
    "button[class*='cookie']",
    "[class*='cookie-banner'] button",
    "[class*='cookie-consent'] button",
    "[class*='cookieConsent'] button",
    "[id*='cookie-banner'] button",
    "[id*='cookie-consent'] button",
    "[id*='onetrust-accept']",
    "#onetrust-accept-btn-handler",
    ".cookie-notice button",
    ".gdpr-consent button",
    # 通用模式
    "button:has-text('Accept')",
    "button:has-text('Accept All')",
    "button:has-text('Accept Cookies')",
    "button:has-text('I Accept')",
    "button:has-text('Got it')",
    "button:has-text('OK')",
    "button:has-text('Agree')",
    "button:has-text('Allow')",
    "button:has-text('Allow All')",
    # 中文
    "button:has-text('接受')",
    "button:has-text('同意')",
    "button:has-text('确定')",
]

# 页面加载等待策略
WaitUntilType = Literal["domcontentloaded", "load", "networkidle", "commit"]


@dataclass
class CrawlResult:
    """爬取结果数据类"""
    url: str
    success: bool
    title: Optional[str] = None
    content: Optional[str] = None
    images: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    error: Optional[str] = None
    # 智能提取的结构化数据
    article: Optional[ArticleData] = None
    # 分页相关
    pages_crawled: int = 1  # 爬取的页数
    all_page_urls: list[str] = field(default_factory=list)  # 所有爬取的页面 URL

    def to_dict(self) -> dict:
        """转换为字典格式"""
        result = {
            "url": self.url,
            "success": self.success,
            "title": self.title,
            "content": self.content,
            "images": self.images,
            "links": self.links,
            "error": self.error,
            "pages_crawled": self.pages_crawled,
            "all_page_urls": self.all_page_urls,
        }
        if self.article:
            result["article"] = self.article.to_dict()
        return result


@dataclass
class CrawlOptions:
    """爬取选项配置"""
    wait_until: WaitUntilType = "domcontentloaded"
    wait_for_selector: Optional[str] = None
    wait_for_selector_timeout: int = 10000
    scroll_to_load: bool = False
    scroll_times: int = 3
    scroll_delay: float = 0.5
    # 智能内容提取选项
    smart_extract: bool = False  # 是否使用智能内容提取
    # 反爬虫选项
    random_user_agent: bool = False  # 是否使用随机 User-Agent
    user_agent: Optional[str] = None  # 指定 User-Agent，优先级高于 random
    simulate_human: bool = False  # 是否模拟人类行为
    human_delay_range: tuple = (0.5, 2.0)  # 人类行为延迟范围（秒）
    handle_cookie_consent: bool = False  # 是否处理 Cookie 同意弹窗
    stealth_mode: bool = False  # 是否启用隐身模式（组合多种反检测措施）
    # 分页和无限滚动选项
    handle_pagination: bool = False  # 是否处理分页
    max_pages: int = 5  # 最大爬取页数
    pagination_selector: Optional[str] = None  # 分页链接选择器
    pagination_next_text: list = None  # 下一页按钮文本列表
    infinite_scroll: bool = False  # 是否处理无限滚动
    infinite_scroll_max: int = 10  # 无限滚动最大次数
    infinite_scroll_delay: float = 1.0  # 滚动后等待时间
    infinite_scroll_selector: Optional[str] = None  # 新内容选择器（用于检测加载完成）

    def __post_init__(self):
        if self.pagination_next_text is None:
            self.pagination_next_text = [
                "Next", "next", "下一页", "»", "›", "→",
                "Next Page", "Next page", "NEXT",
            ]


class WebCrawler:
    """网页爬虫类"""

    def __init__(
        self,
        headless: bool = True,
        timeout: int = 30000,
        default_wait_until: WaitUntilType = "domcontentloaded",
        crawl_logger: Optional[CrawlLogger] = None,
    ):
        """
        初始化爬虫

        Args:
            headless: 是否无头模式运行
            timeout: 页面加载超时时间（毫秒）
            default_wait_until: 默认页面加载等待策略
            crawl_logger: 可选的日志记录器，用于记录爬取详情
        """
        self.headless = headless
        self.timeout = timeout
        self.default_wait_until = default_wait_until
        self._browser: Optional[Browser] = None
        self._playwright = None
        self._crawl_logger = crawl_logger

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    async def start(self):
        """启动浏览器"""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        logger.info("Browser started")

    async def close(self):
        """关闭浏览器"""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info("Browser closed")

    def _get_user_agent(self, opts: CrawlOptions) -> str:
        """获取 User-Agent"""
        if opts.user_agent:
            return opts.user_agent
        if opts.random_user_agent or opts.stealth_mode:
            return random.choice(USER_AGENTS)
        return None

    async def _create_context(self, opts: CrawlOptions) -> BrowserContext:
        """创建浏览器上下文，支持反爬虫配置"""
        context_options = {
            "ignore_https_errors": True,
        }

        # 设置 User-Agent
        user_agent = self._get_user_agent(opts)
        if user_agent:
            context_options["user_agent"] = user_agent
            logger.debug(f"Using User-Agent: {user_agent[:50]}...")

        # 设置视口大小（模拟真实浏览器）
        if opts.stealth_mode or opts.simulate_human:
            # 常见的屏幕分辨率
            viewports = [
                {"width": 1920, "height": 1080},
                {"width": 1366, "height": 768},
                {"width": 1536, "height": 864},
                {"width": 1440, "height": 900},
                {"width": 1280, "height": 720},
            ]
            context_options["viewport"] = random.choice(viewports)

        # 设置语言和时区
        if opts.stealth_mode:
            context_options["locale"] = "en-US"
            context_options["timezone_id"] = "America/New_York"

        context = await self._browser.new_context(**context_options)

        # 隐身模式：注入反检测脚本
        if opts.stealth_mode:
            await self._inject_stealth_scripts(context)

        return context

    async def _inject_stealth_scripts(self, context: BrowserContext):
        """注入反检测脚本"""
        # 在每个页面加载前注入脚本
        await context.add_init_script("""
            // 覆盖 navigator.webdriver
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });

            // 覆盖 chrome 对象
            window.chrome = {
                runtime: {}
            };

            // 覆盖权限查询
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );

            // 覆盖插件数组长度
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });

            // 覆盖语言
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
        """)
        logger.debug("Stealth scripts injected")

    async def _simulate_human_behavior(self, page: Page, opts: CrawlOptions):
        """模拟人类行为"""
        delay_min, delay_max = opts.human_delay_range

        # 随机延迟
        await asyncio.sleep(random.uniform(delay_min, delay_max))

        # 随机移动鼠标
        viewport_size = page.viewport_size
        if viewport_size:
            x = random.randint(100, viewport_size["width"] - 100)
            y = random.randint(100, viewport_size["height"] - 100)
            await page.mouse.move(x, y)
            logger.debug(f"Mouse moved to ({x}, {y})")

        # 随机滚动一小段
        scroll_amount = random.randint(100, 300)
        await page.evaluate(f"window.scrollBy(0, {scroll_amount})")
        await asyncio.sleep(random.uniform(0.3, 0.8))

        # 再滚动回来一点
        scroll_back = random.randint(50, 100)
        await page.evaluate(f"window.scrollBy(0, -{scroll_back})")
        await asyncio.sleep(random.uniform(0.2, 0.5))

        logger.debug("Human behavior simulated")

    async def _handle_cookie_consent(self, page: Page) -> bool:
        """
        处理 Cookie 同意弹窗

        Returns:
            bool: 是否成功处理了弹窗
        """
        for selector in COOKIE_CONSENT_SELECTORS:
            try:
                # 尝试查找并点击按钮
                button = await page.query_selector(selector)
                if button:
                    # 检查按钮是否可见
                    is_visible = await button.is_visible()
                    if is_visible:
                        await button.click()
                        logger.info(f"Cookie consent handled via: {selector}")
                        await asyncio.sleep(0.5)  # 等待弹窗关闭
                        return True
            except Exception as e:
                # 某些选择器可能不适用，继续尝试下一个
                logger.debug(f"Cookie selector '{selector}' failed: {e}")
                continue

        logger.debug("No cookie consent dialog found or handled")
        return False

    async def _find_next_page_link(self, page: Page, opts: CrawlOptions) -> Optional[str]:
        """
        查找下一页链接

        Args:
            page: Playwright 页面对象
            opts: 爬取选项

        Returns:
            Optional[str]: 下一页 URL，如果没有则返回 None
        """
        # 如果指定了分页选择器，优先使用
        if opts.pagination_selector:
            try:
                next_link = await page.query_selector(opts.pagination_selector)
                if next_link:
                    href = await next_link.get_attribute("href")
                    if href:
                        return urljoin(page.url, href)
            except Exception as e:
                logger.debug(f"Pagination selector '{opts.pagination_selector}' failed: {e}")

        # 尝试通过文本查找下一页链接
        for text in opts.pagination_next_text:
            try:
                # 使用 text 选择器
                selectors = [
                    f"a:has-text('{text}')",
                    f"button:has-text('{text}')",
                    f"[class*='next'] a",
                    f"[class*='pagination'] a:has-text('{text}')",
                    f"a[rel='next']",
                    f"link[rel='next']",
                ]

                for selector in selectors:
                    try:
                        element = await page.query_selector(selector)
                        if element:
                            # 检查是否可见且可点击
                            is_visible = await element.is_visible()
                            if is_visible:
                                href = await element.get_attribute("href")
                                if href:
                                    return urljoin(page.url, href)
                    except Exception:
                        continue
            except Exception as e:
                logger.debug(f"Finding next page with text '{text}' failed: {e}")
                continue

        # 尝试常见的分页模式
        common_selectors = [
            "a.next",
            ".pagination a.next",
            ".pagination .next a",
            "[class*='pagination'] [class*='next'] a",
            ".pager .next a",
            "a[aria-label='Next']",
            "a[aria-label='Next page']",
        ]

        for selector in common_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    is_visible = await element.is_visible()
                    if is_visible:
                        href = await element.get_attribute("href")
                        if href:
                            return urljoin(page.url, href)
            except Exception:
                continue

        logger.debug("No next page link found")
        return None

    async def _handle_infinite_scroll(
        self,
        page: Page,
        opts: CrawlOptions
    ) -> tuple[list[str], list[str], int]:
        """
        处理无限滚动页面

        Args:
            page: Playwright 页面对象
            opts: 爬取选项

        Returns:
            tuple: (所有内容列表, 所有链接列表, 滚动次数)
        """
        all_content = []
        all_links = set()
        scroll_count = 0
        last_height = 0
        no_change_count = 0

        for i in range(opts.infinite_scroll_max):
            scroll_count = i + 1

            # 获取当前内容数量（如果指定了选择器）
            current_items = 0
            if opts.infinite_scroll_selector:
                try:
                    items = await page.query_selector_all(opts.infinite_scroll_selector)
                    current_items = len(items)
                except Exception:
                    pass

            # 滚动到底部
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            logger.debug(f"Infinite scroll {scroll_count}/{opts.infinite_scroll_max}")

            # 等待新内容加载
            await asyncio.sleep(opts.infinite_scroll_delay)

            # 检查是否有新内容加载
            current_height = await page.evaluate("document.body.scrollHeight")

            # 如果指定了选择器，检查项目数量变化
            if opts.infinite_scroll_selector:
                try:
                    items = await page.query_selector_all(opts.infinite_scroll_selector)
                    new_items = len(items)
                    if new_items == current_items:
                        no_change_count += 1
                    else:
                        no_change_count = 0
                except Exception:
                    pass

            # 如果高度没有变化，可能已经到底
            if current_height == last_height:
                no_change_count += 1
                if no_change_count >= 3:
                    logger.info(f"Infinite scroll completed after {scroll_count} scrolls (no more content)")
                    break
            else:
                no_change_count = 0

            last_height = current_height

            # 提取当前可见的链接
            try:
                a_elements = await page.query_selector_all("a[href]")
                for a in a_elements:
                    href = await a.get_attribute("href")
                    if href and not href.startswith(("#", "javascript:", "mailto:", "tel:")):
                        all_links.add(urljoin(page.url, href))
            except Exception:
                pass

        logger.info(f"Infinite scroll: {scroll_count} scrolls, {len(all_links)} unique links found")
        return all_content, list(all_links), scroll_count

    async def crawl(
        self,
        url: str,
        options: Optional[CrawlOptions] = None,
        retry_count: int = 0,
    ) -> CrawlResult:
        """
        爬取指定 URL

        Args:
            url: 要爬取的 URL
            options: 爬取选项，如果为 None 则使用默认选项
            retry_count: 当前重试次数（用于日志记录）

        Returns:
            CrawlResult: 爬取结果
        """
        if not self._browser:
            raise RuntimeError("Browser not started. Call start() first or use async context manager.")

        # 使用默认选项或自定义选项
        opts = options or CrawlOptions(wait_until=self.default_wait_until)

        context = None
        page = None
        start_time = asyncio.get_event_loop().time()
        content_length = None
        status_code = None

        try:
            # 创建带有反爬虫配置的上下文
            context = await self._create_context(opts)
            page = await context.new_page()
            page.set_default_timeout(self.timeout)

            # 加载页面
            response = await page.goto(url, wait_until=opts.wait_until)
            if response:
                status_code = response.status
            logger.info(f"Loaded: {url} (wait_until={opts.wait_until})")

            # 等待页面稳定（处理重定向/导航）
            try:
                await page.wait_for_load_state("load", timeout=10000)
            except Exception:
                pass  # 超时没关系，只是尽力等待

            # 处理 Cookie 同意弹窗
            if opts.handle_cookie_consent or opts.stealth_mode:
                await self._handle_cookie_consent(page)

            # 模拟人类行为
            if opts.simulate_human or opts.stealth_mode:
                try:
                    await self._simulate_human_behavior(page, opts)
                except Exception as e:
                    logger.warning(f"Human behavior simulation failed (non-fatal): {e}")

            # 等待自定义选择器
            if opts.wait_for_selector:
                try:
                    await page.wait_for_selector(
                        opts.wait_for_selector,
                        timeout=opts.wait_for_selector_timeout
                    )
                    logger.info(f"Selector '{opts.wait_for_selector}' found")
                except Exception as e:
                    logger.warning(f"Wait for selector '{opts.wait_for_selector}' failed: {e}")

            # 处理懒加载内容 - 滚动页面
            if opts.scroll_to_load:
                try:
                    await self._scroll_page(page, opts.scroll_times, opts.scroll_delay)
                except Exception as e:
                    logger.warning(f"Scroll to load failed (non-fatal): {e}")

            # 等待页面在模拟行为后再次稳定
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=5000)
            except Exception:
                pass

            # 提取内容（带重试，处理导航导致的 context 销毁）
            article = None
            extract_attempts = 2
            for attempt in range(extract_attempts):
                try:
                    if opts.smart_extract:
                        # 使用智能提取器
                        html = await page.content()
                        extractor = SmartExtractor(base_url=page.url)
                        article = extractor.extract(html)
                        title = article.title
                        content = article.content
                        images = article.images
                        links = article.links
                    else:
                        # 使用基础提取
                        title = await self._extract_title(page)
                        content = await self._extract_content(page)
                        images = await self._extract_images(page, page.url)
                        links = await self._extract_links(page, page.url)
                    break  # 提取成功，跳出重试循环
                except Exception as extract_err:
                    err_msg = str(extract_err).lower()
                    if attempt < extract_attempts - 1 and (
                        "context" in err_msg or "navigating" in err_msg or "destroyed" in err_msg
                    ):
                        logger.warning(f"Content extraction failed (attempt {attempt + 1}), retrying after wait: {extract_err}")
                        await asyncio.sleep(2)
                        try:
                            await page.wait_for_load_state("load", timeout=10000)
                        except Exception:
                            pass
                    else:
                        raise

            # 计算内容长度
            if content:
                content_length = len(content)

            # 计算响应时间
            response_time = asyncio.get_event_loop().time() - start_time

            # 记录成功日志
            if self._crawl_logger:
                from datetime import datetime
                log_entry = CrawlLogEntry(
                    timestamp=datetime.now().isoformat(),
                    url=url,
                    success=True,
                    error_type=None,
                    error_message=None,
                    status_code=status_code,
                    response_time=round(response_time, 3),
                    content_length=content_length,
                    retry_count=retry_count,
                    domain=extract_domain(url),
                )
                self._crawl_logger.log(log_entry)

            return CrawlResult(
                url=url,
                success=True,
                title=title,
                content=content,
                images=images,
                links=links,
                article=article,
            )

        except Exception as e:
            logger.error(f"Failed to crawl {url}: {e}")

            # 计算响应时间
            response_time = asyncio.get_event_loop().time() - start_time

            # 分类错误并记录日志
            error_type, error_message = classify_error(e)
            if self._crawl_logger:
                from datetime import datetime
                log_entry = CrawlLogEntry(
                    timestamp=datetime.now().isoformat(),
                    url=url,
                    success=False,
                    error_type=error_type,
                    error_message=error_message,
                    status_code=status_code,
                    response_time=round(response_time, 3),
                    content_length=None,
                    retry_count=retry_count,
                    domain=extract_domain(url),
                )
                self._crawl_logger.log(log_entry)

            return CrawlResult(
                url=url,
                success=False,
                error=str(e),
            )

        finally:
            if page:
                await page.close()
            if context:
                await context.close()

    async def crawl_with_pagination(
        self,
        url: str,
        options: Optional[CrawlOptions] = None,
    ) -> CrawlResult:
        """
        爬取带分页的页面，自动处理分页和无限滚动

        Args:
            url: 起始 URL
            options: 爬取选项

        Returns:
            CrawlResult: 包含所有页面合并内容的结果
        """
        if not self._browser:
            raise RuntimeError("Browser not started. Call start() first or use async context manager.")

        # 使用默认选项或自定义选项
        opts = options or CrawlOptions(wait_until=self.default_wait_until)

        all_content = []
        all_images = []
        all_links = set()
        all_page_urls = [url]
        title = None
        pages_crawled = 0
        current_url = url

        context = None
        page = None

        try:
            # 创建带有反爬虫配置的上下文
            context = await self._create_context(opts)
            page = await context.new_page()
            page.set_default_timeout(self.timeout)

            while current_url and pages_crawled < opts.max_pages:
                pages_crawled += 1
                logger.info(f"Crawling page {pages_crawled}/{opts.max_pages}: {current_url}")

                # 加载页面
                await page.goto(current_url, wait_until=opts.wait_until)

                # 处理 Cookie 同意弹窗（仅第一页）
                if pages_crawled == 1 and (opts.handle_cookie_consent or opts.stealth_mode):
                    await self._handle_cookie_consent(page)

                # 模拟人类行为
                if opts.simulate_human or opts.stealth_mode:
                    await self._simulate_human_behavior(page, opts)

                # 等待自定义选择器
                if opts.wait_for_selector:
                    try:
                        await page.wait_for_selector(
                            opts.wait_for_selector,
                            timeout=opts.wait_for_selector_timeout
                        )
                    except Exception as e:
                        logger.warning(f"Wait for selector failed: {e}")

                # 处理无限滚动
                if opts.infinite_scroll:
                    _, scroll_links, _ = await self._handle_infinite_scroll(page, opts)
                    all_links.update(scroll_links)
                elif opts.scroll_to_load:
                    await self._scroll_page(page, opts.scroll_times, opts.scroll_delay)

                # 提取标题（仅第一页）
                if pages_crawled == 1:
                    title = await self._extract_title(page)

                # 提取内容
                content = await self._extract_content(page)
                if content:
                    all_content.append(content)

                # 提取图片和链接
                images = await self._extract_images(page, current_url)
                links = await self._extract_links(page, current_url)
                all_images.extend(images)
                all_links.update(links)

                # 如果启用了分页处理，查找下一页
                if opts.handle_pagination and pages_crawled < opts.max_pages:
                    next_url = await self._find_next_page_link(page, opts)
                    if next_url and next_url not in all_page_urls:
                        current_url = next_url
                        all_page_urls.append(next_url)
                        # 添加页面间延迟
                        if opts.simulate_human or opts.stealth_mode:
                            await asyncio.sleep(random.uniform(1.0, 3.0))
                    else:
                        logger.info(f"No more pages found after page {pages_crawled}")
                        break
                else:
                    break

            # 合并所有内容
            merged_content = "\n\n---\n\n".join(all_content) if all_content else None

            # 去重图片
            unique_images = list(dict.fromkeys(all_images))

            return CrawlResult(
                url=url,
                success=True,
                title=title,
                content=merged_content,
                images=unique_images,
                links=list(all_links),
                pages_crawled=pages_crawled,
                all_page_urls=all_page_urls,
            )

        except Exception as e:
            logger.error(f"Failed to crawl with pagination {url}: {e}")
            return CrawlResult(
                url=url,
                success=False,
                error=str(e),
                pages_crawled=pages_crawled,
                all_page_urls=all_page_urls,
            )

        finally:
            if page:
                await page.close()
            if context:
                await context.close()

    async def crawl_infinite_scroll(
        self,
        url: str,
        options: Optional[CrawlOptions] = None,
    ) -> CrawlResult:
        """
        专门处理无限滚动页面的爬取方法

        Args:
            url: 要爬取的 URL
            options: 爬取选项

        Returns:
            CrawlResult: 爬取结果
        """
        # 确保启用无限滚动
        opts = options or CrawlOptions()
        opts.infinite_scroll = True

        if not self._browser:
            raise RuntimeError("Browser not started. Call start() first or use async context manager.")

        context = None
        page = None

        try:
            context = await self._create_context(opts)
            page = await context.new_page()
            page.set_default_timeout(self.timeout)

            # 加载页面
            await page.goto(url, wait_until=opts.wait_until)
            logger.info(f"Loaded: {url}")

            # 处理 Cookie 同意弹窗
            if opts.handle_cookie_consent or opts.stealth_mode:
                await self._handle_cookie_consent(page)

            # 模拟人类行为
            if opts.simulate_human or opts.stealth_mode:
                await self._simulate_human_behavior(page, opts)

            # 等待选择器
            if opts.wait_for_selector:
                try:
                    await page.wait_for_selector(
                        opts.wait_for_selector,
                        timeout=opts.wait_for_selector_timeout
                    )
                except Exception as e:
                    logger.warning(f"Wait for selector failed: {e}")

            # 执行无限滚动
            _, scroll_links, scroll_count = await self._handle_infinite_scroll(page, opts)

            # 提取最终内容
            title = await self._extract_title(page)
            content = await self._extract_content(page)
            images = await self._extract_images(page, url)

            # 合并无限滚动收集的链接
            all_links = list(set(scroll_links + await self._extract_links(page, url)))

            return CrawlResult(
                url=url,
                success=True,
                title=title,
                content=content,
                images=images,
                links=all_links,
                pages_crawled=scroll_count,
                all_page_urls=[url],
            )

        except Exception as e:
            logger.error(f"Failed to crawl infinite scroll {url}: {e}")
            return CrawlResult(
                url=url,
                success=False,
                error=str(e),
            )

        finally:
            if page:
                await page.close()
            if context:
                await context.close()

    async def _scroll_page(self, page: Page, scroll_times: int, scroll_delay: float):
        """
        滚动页面以加载懒加载内容

        Args:
            page: Playwright 页面对象
            scroll_times: 滚动次数
            scroll_delay: 每次滚动后的等待时间（秒）
        """
        for i in range(scroll_times):
            # 获取当前滚动位置和页面高度
            scroll_height = await page.evaluate("document.body.scrollHeight")
            current_scroll = await page.evaluate("window.scrollY")
            viewport_height = await page.evaluate("window.innerHeight")

            # 滚动一个视口高度
            new_scroll = min(current_scroll + viewport_height, scroll_height)
            await page.evaluate(f"window.scrollTo(0, {new_scroll})")

            logger.debug(f"Scroll {i+1}/{scroll_times}: {current_scroll} -> {new_scroll}")

            # 等待内容加载
            await asyncio.sleep(scroll_delay)

            # 如果已经到底部，尝试等待新内容加载
            if new_scroll >= scroll_height - viewport_height:
                await asyncio.sleep(scroll_delay * 2)  # 多等一会
                new_height = await page.evaluate("document.body.scrollHeight")
                if new_height == scroll_height:
                    logger.info("Reached bottom of page, no more content to load")
                    break

    async def _extract_title(self, page: Page) -> Optional[str]:
        """提取页面标题"""
        try:
            return await page.title()
        except Exception as e:
            logger.warning(f"Failed to extract title: {e}")
            return None

    async def _extract_content(self, page: Page) -> Optional[str]:
        """提取页面主要文本内容"""
        # 尝试多个常见的内容选择器
        content_selectors = [
            "article",
            "main",
            "[role='main']",
            ".content",
            ".post-content",
            ".article-content",
            ".entry-content",
            "#content",
            "#main-content",
        ]

        for selector in content_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    text = await element.inner_text()
                    if text and len(text.strip()) > 100:
                        return text.strip()
            except Exception:
                continue

        # 如果找不到特定内容区域，回退到 body
        try:
            body = await page.query_selector("body")
            if body:
                return (await body.inner_text()).strip()
        except Exception as e:
            logger.warning(f"Failed to extract content: {e}")

        return None

    async def _extract_images(self, page: Page, base_url: str) -> list[str]:
        """提取所有图片 URL"""
        images = []
        try:
            img_elements = await page.query_selector_all("img[src]")
            for img in img_elements:
                src = await img.get_attribute("src")
                if src:
                    # 转换为绝对 URL
                    absolute_url = urljoin(base_url, src)
                    images.append(absolute_url)
        except Exception as e:
            logger.warning(f"Failed to extract images: {e}")

        return images

    async def _extract_links(self, page: Page, base_url: str) -> list[str]:
        """提取所有链接"""
        links = []
        try:
            a_elements = await page.query_selector_all("a[href]")
            for a in a_elements:
                href = await a.get_attribute("href")
                if href and not href.startswith(("#", "javascript:", "mailto:", "tel:")):
                    # 转换为绝对 URL
                    absolute_url = urljoin(base_url, href)
                    links.append(absolute_url)
        except Exception as e:
            logger.warning(f"Failed to extract links: {e}")

        return links
