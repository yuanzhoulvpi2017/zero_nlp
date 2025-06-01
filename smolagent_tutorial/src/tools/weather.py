import json
from smolagents.tools import Tool
from typing import List, Dict, Any
import pandas as pd
import requests


def format_weather_data(dict) -> str:
    s = ""
    for k, v in dict.items():
        s += f"{str(k)}:{str(v)};"
    return s


class AMapWeatherTool(Tool):
    name = "AMapWeatherTool"
    description = "get weather information from AMap API. Input should be a city name in Chinese, and the output will be a dictionary containing the weather information."
    inputs = {
        "query": {
            "type": "string",
            "description": "The name of the city in Chinese.",
        },
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__()

        city_id_df = pd.read_excel("data/AMap_adcode_citycode.xlsx")[
            ["中文名", "adcode"]
        ].to_dict(orient="records")
        self.city_id_dict = {
            k: v
            for k, v in zip(
                [i["中文名"] for i in city_id_df], [i["adcode"] for i in city_id_df]
            )
        }

    def forward(self, query: str) -> str:
        query = query.strip()
        city_id_list = [v for k, v in self.city_id_dict.items() if query in k]
        results = []
        for city_id in city_id_list:
            result = self.get_weather(city_id)
            results.extend(result)

        if len(results) > 0:
            final_result = "## Search Results\n\n" + "\n\n".join(
                [json.dumps(i, ensure_ascii=False) for i in results]
            )
            return final_result

        else:
            raise Exception("No results found! Try a less restrictive/shorter query.")

    def get_weather(self, city_id: str) -> List[dict]:
        try:
            url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={city_id}&key=389c43ce"

            response = requests.get(url)
            return json.loads(response.text).get("lives")
        except Exception as e:
            return []


if __name__ == "__main__":
    amp_tool = AMapWeatherTool()
    print(amp_tool.forward("台州"))
