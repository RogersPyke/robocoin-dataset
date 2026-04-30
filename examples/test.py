# import yaml
# from pathlib import Path
# import logging
# from robocoin_dataset.format_converter.tolerobot.lerobot_format_converter import LerobotFormatConverterFactory

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # 🔧 正确读取 YAML 文件内容
# config_path = Path("scripts/format_converters/tolerobot/configs/converter_config_ego_rrd_normal.yaml")
# if not config_path.exists():
#     raise FileNotFoundError(f"Config file not found: {config_path}")

# with open(config_path) as f:
#     config = yaml.safe_load(f)

# converter = LerobotFormatConverterFactory.create_converter(
#     dataset_path=Path("/home/liuyou/Documents/data/genrobot_ego_data_beta_v1.rrd"),
#     device_model="ego",
#     output_path=Path("/home/liuyou/Documents/data/test/"),
#     converter_config=config,  # ← 这里传入解析后的字典
#     converter_module_path="robocoin_dataset.format_converter.tolerobot.lerobot_format_converter_rrd",
#     converter_class_name="LerobotFormatConverterRrd",
#     repo_id="genrobot/lerobot-format-converter-rrd",
#     logger=logger,
# )
# for _ in converter.convert(is_test=True):
#     pass

# from openai import OpenAI

# client = OpenAI(
#     api_key=
#     base_url="https://api.bltcy.ai/v1" # 注意通常需要带上 /v1
# )

# response = client.chat.completions.create(
#     model="gemini-3.1-flash-lite-preview", # 或者 gemini-1.5-flash
#     messages=[{"role": "user", "content": "你好，请自我介绍"}]
# )
# print(response.choices[0].message.content)