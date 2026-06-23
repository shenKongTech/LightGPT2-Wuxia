import os
import sys
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_device, generate_stream
from src.data import BpeTokenizer
from src.model import GPTModel


def main():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "base_config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    ckpt_path = cfg["path"]["checkpoint_path"]
    bpe_model = cfg["path"]["bpe_model_prefix"]

    # 检查文件
    if not os.path.exists(ckpt_path):
        print(f"错误：未找到模型权重 {ckpt_path}，请先运行训练！")
        return
    if not os.path.exists(f"{bpe_model}.model"):
        print(f"错误：未找到分词器文件，请先运行训练！")
        return

    # 加载分词器
    print("正在加载分词器...")
    tokenizer = BpeTokenizer(
        model_prefix=bpe_model,
        vocab_size=cfg["model"]["vocab_size"]
    )

    # 加载模型
    print("正在加载模型权重...")
    model = GPTModel(cfg["model"]).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"模型加载完成，已训练至第 {ckpt['epoch'] + 1} 轮")

    # 交互式对话
    print("\n" + "=" * 50)
    print("    武侠GPT 流式对话模式（输入 quit 退出）")
    print("=" * 50)

    while True:
        prompt = input("\n请输入故事开头：").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            print("退出对话模式。")
            break
        if not prompt:
            continue
        generate_stream(model, prompt, tokenizer, max_new=300, temperature=0.85, device=device)


if __name__ == "__main__":
    main()

