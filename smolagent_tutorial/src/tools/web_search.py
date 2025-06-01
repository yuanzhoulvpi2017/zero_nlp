from smolagents import Tool
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time


class BaiduSearch:
    def __init__(self, query: str, max_results: int = 20):
        self.query = query
        self.max_results = max_results
        self.service = Service(
            executable_path="/Users/yuanz/Downloads/chromedriver-mac-arm64/chromedriver"
        )
        self.driver = webdriver.Chrome(service=self.service)

    def query2search_url(self) -> list[dict]:
        # 打开百度首页
        self.driver.get("https://www.baidu.com")

        # 找到搜索框，输入关键词
        search_box = self.driver.find_element(By.ID, "kw")
        search_box.send_keys("尼康z63")
        search_box.send_keys(Keys.RETURN)

        # 等待几秒钟以便查看结果
        time.sleep(4)
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)

        results = []

        # 获取搜索结果
        search_results = self.driver.find_elements(
            By.CSS_SELECTOR, "div[class*='_aladdin_1ml43_1']"
        )
        for result in search_results:
            title = result.find_element(By.TAG_NAME, "h3").text
            link = result.find_element(By.TAG_NAME, "a").get_attribute("href")
            # 尝试提取时间信息
            try:
                date_element = result.find_element(
                    By.CSS_SELECTOR,
                    "span[class*='cos-space-mr-3xs cos-color-text-minor']",
                )
                date = date_element.text
            except:
                date = "未找到日期"
            results.append({"title": title, "link": link, "date": date})
        return results

    def sub_url_text(self, url: str):
        # 在新标签页中打开第一个搜索结果的链接
        self.driver.execute_script("window.open('{}', '_blank');".format(url))

        # 切换到新打开的标签页
        self.driver.switch_to.window(self.driver.window_handles[-1])

        # 等待3秒
        time.sleep(3)
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # 获取页面所有文本内容
        page_text = self.driver.find_element(By.TAG_NAME, "body").text

        return page_text

    def search_result(self):
        results = self.query2search_url()
        postprocessed_results = []
        for result in results[: self.max_results]:
            title = result["title"]
            link = result["link"]
            date = result["date"]
            text = self.sub_url_text(link)
            # 其实这里的text太长了，最好还是做一下summary、或者使用传统nlp提取关键信息
            if len(text.strip()) > 20:
                postprocessed_results.append(
                    f"[{title}]({link})\n发布时间: {date} \n {text}"
                )

        self.driver.quit()
        return postprocessed_results


class BaiDuSearchTool(Tool):
    name = "baidu_search"
    description = """Performs a 百度 search based on your query (think a Google search) then returns the top search results."""
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to perform.(长度在10个字以内 中文)",
        },
    }
    output_type = "string"

    def __init__(self, max_results=20, **kwargs):
        super().__init__()
        self.max_results = max_results

    def forward(self, query: str) -> str:
        try:
            bs = BaiduSearch(query, max_results=self.max_results)
            postprocessed_results = bs.search_result()
            return "## Search Results\n\n" + "\n\n".join(postprocessed_results)

        except Exception as e:
            raise Exception("No results found! Try a less restrictive/shorter query.")
