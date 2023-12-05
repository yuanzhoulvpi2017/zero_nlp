from datasets import load_dataset

# 如果有能力直接下载这个数据的，可以运行这个命令
dataset = load_dataset("Anthropic/hh-rlhf")


# 将数据分别保存到train数据和test数据
dataset['train'].to_json('data/hh-rlhf2/train_hhrlhf.json')
dataset['test'].to_json('data/hh-rlhf2/test_hhrlhf.json')
