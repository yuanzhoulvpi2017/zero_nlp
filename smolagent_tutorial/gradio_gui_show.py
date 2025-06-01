import os
from urllib.request import getproxies

from smolagents import CodeAgent, GradioUI, OpenAIServerModel, stream_to_gradio

import uuid
import gradio as gr
from logging.handlers import TimedRotatingFileHandler
import logging
import json

# from src.tools.weather import AMapWeatherTool
from src.tools.web_search import BaiDuSearchTool


os.makedirs("log", exist_ok=True)  # 确保日志目录存在


logger = logging.getLogger(__name__)

file_handler = TimedRotatingFileHandler(
    "log/search_query.log", when="midnight", interval=1, backupCount=7
)
file_handler.setLevel(logging.INFO)  # 设置文件处理器的日志级别
logger.addHandler(file_handler)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
# 关闭 elasticsearch 的所有日志
logging.getLogger("elasticsearch").setLevel(logging.CRITICAL)
# 关闭 elasticsearch 相关的 urllib3 日志
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# https://github.com/QwenLM/Qwen-Agent/issues/223
proxies = getproxies()
SESSION_AGENTS = {}

# os.environ["HTTP_PROXY"] = os.environ["http_proxy"] = proxies["http"]
# os.environ["HTTPS_PROXY"] = os.environ["https_proxy"] = proxies["https"]
# os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost, 127.0.0.1/8, ::1,0.0.0.0"


MODEL_NAME_1 = "Qwen2.5-72B-Instruct"
MODEL_NAME_2 = "model_2"


def create_agent(model_name=MODEL_NAME_1):
    """为每个新会话创建新的Agent实例"""
    search_tool = BaiDuSearchTool()
    # wather_tool = AMapWeatherTool()

    if model_name == MODEL_NAME_1:
        model = OpenAIServerModel(
            model_id="Qwen2.5-72B-Instruct",
            api_base="http://10.136.0.65:7008/v1",
            api_key="EMPTY",
        )
    elif model_name == MODEL_NAME_2:
        model = OpenAIServerModel(
            model_id="Qwen2.5-72B-Instruct",
            api_base="http://10.136.0.65:7008/v1",
            api_key="EMPTY",
        )

    # 每次调用此函数时创建新的Agent实例
    return CodeAgent(
        tools=[search_tool],
        model=model,
    )


def model_change(model_name):
    return gr.Info(f"模型已切换至: {model_name}", duration=5)


class CustomGradioUI(GradioUI):
    def __init__(self, agent_creator):
        """初始化自定义Gradio UI，使用agent_creator函数而不是agent实例。

        Args:
            agent_creator: 创建新agent实例的函数
        """
        self.agent_creator = agent_creator
        super().__init__(agent_creator(MODEL_NAME_2))

    def interact_with_agent(self, prompt, messages, session_id, model_name):
        import gradio as gr

        # 确保会话ID有对应的agent
        if (
            session_id not in SESSION_AGENTS
            or SESSION_AGENTS[session_id]["model"] != model_name
        ):
            SESSION_AGENTS[session_id] = {
                "agent": self.agent_creator(model_name),
                "model": model_name,
            }
            logger.info(
                json.dumps(
                    {"prompt": prompt, "session_id": session_id}, ensure_ascii=False
                )
            )
        try:
            messages.append(gr.ChatMessage(role="user", content=prompt))
            yield messages

            for msg in stream_to_gradio(
                SESSION_AGENTS[session_id]["agent"],
                task=prompt,
                reset_agent_memory=False,
            ):
                messages.append(msg)
                yield messages

            yield messages
        except Exception as e:
            print(f"Error in interaction: {str(e)}")
            messages.append(
                gr.ChatMessage(role="assistant", content=f"Error: {str(e)}")
            )
            yield messages

    def create_app(self):
        import gradio as gr

        with gr.Blocks(theme="ocean", fill_height=True) as demo:
            # Add session state to store session-specific data
            session_id = gr.State(str(uuid.uuid4()))
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])

            # 每当访问页面时，创建一个新的会话ID
            demo.load(lambda: str(uuid.uuid4()), [], [session_id])

            with gr.Sidebar():
                gr.Markdown(f"# {self.name.replace('_', ' ').capitalize()}")
                # gr.Markdown(
                #     f"# {self.name.replace('_', ' ').capitalize()}"
                #     "\n> This web ui allows you to interact with a `smolagents` agent that can use tools and execute steps to complete tasks."
                #     + (
                #         f"\n\n**Agent description:**\n{self.description}"
                #         if self.description
                #         else ""
                #     )
                # )
                # 添加模型选择下拉菜单
                model_selector = gr.Dropdown(
                    choices=[MODEL_NAME_1, MODEL_NAME_2],
                    value=MODEL_NAME_1,
                    label="Select Model",
                    interactive=True,
                )
                # 添加模型切换事件处理
                model_selector.change(
                    fn=model_change,
                    inputs=[model_selector],
                    outputs=None,
                )

                with gr.Group():
                    gr.Markdown("**Your request**", container=True)
                    text_input = gr.Textbox(
                        lines=3,
                        label="Chat Message",
                        container=False,
                        placeholder="Enter your prompt here and press Shift+Enter or press the button",
                    )
                    submit_btn = gr.Button("Submit", variant="primary")

                # If an upload folder is provided, enable the upload feature
                if self.file_upload_folder is not None:
                    upload_file = gr.File(label="Upload a file")
                    upload_status = gr.Textbox(
                        label="Upload Status", interactive=False, visible=False
                    )
                    upload_file.change(
                        self.upload_file,
                        [upload_file, file_uploads_log],
                        [upload_status, file_uploads_log],
                    )

                # gr.HTML("<br><br><h4><center>Powered by:</center></h4>")
                #     with gr.Row():
                #         gr.HTML("""<div style="display: flex; align-items: center; gap: 8px; font-family: system-ui, -apple-system, sans-serif;">
                # <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png" style="width: 32px; height: 32px; object-fit: contain;" alt="logo">
                # <a target="_blank" href="https://github.com/huggingface/smolagents"><b>huggingface/smolagents</b></a>
                # </div>""")
                gr.Examples(
                    examples=[
                        ["尼康z63的参数"],
                        ["杭州天气怎么样"],
                    ],
                    inputs=text_input,
                )

            # Main chat interface
            chatbot = gr.Chatbot(
                label="Agent",
                type="messages",
                avatar_images=(
                    None,
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                ),
                resizeable=True,
                scale=1,
            )

            # Set up event handlers
            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input, submit_btn],
            ).then(
                self.interact_with_agent,
                [stored_messages, chatbot, session_id, model_selector],
                [chatbot],
            ).then(
                lambda: (
                    gr.Textbox(
                        interactive=True,
                        placeholder="Enter your prompt here and press Shift+Enter or the button",
                    ),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

            submit_btn.click(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input, submit_btn],
            ).then(
                self.interact_with_agent,
                [stored_messages, chatbot, session_id, model_selector],
                [chatbot],
            ).then(
                lambda: (
                    gr.Textbox(
                        interactive=True,
                        placeholder="Enter your prompt here and press Shift+Enter or the button",
                    ),
                    gr.Button(interactive=True),
                ),
                None,
                [text_input, submit_btn],
            )

        return demo


# Add a name when initializing GradioUI
CustomGradioUI(create_agent).launch(
    server_name="0.0.0.0", share=False
)  # server_name="127.0.0.1", server_port=args.server_port, share=False
