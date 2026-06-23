import os
import sys
import yaml

# 把项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_device
from src.data import BpeTokenizer, load_and_split_data, create_dataloader
from src.model import GPTModel
from src.training import Trainer


def main():
    # 加载配置
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "base_config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device()

    # 1. 加载数据集
    print("\n正在加载数据集...")
    train_text, val_text = load_and_split_data(
        cfg["data"]["novel_files"],
        train_ratio=cfg["data"]["train_ratio"]
    )
    print(f"训练集字符数: {len(train_text)}, 验证集字符数: {len(val_text)}")

    # 2. 初始化分词器
    all_text = train_text + val_text
    tokenizer = BpeTokenizer(
        model_prefix=cfg["path"]["bpe_model_prefix"],
        vocab_size=cfg["model"]["vocab_size"],
        train_corpus=all_text
    )
    print(f"词表实际大小: {tokenizer.vocab_len}")

    # 3. 构建数据加载器
    train_loader = create_dataloader(
        train_text, tokenizer,
        batch_size=cfg["training"]["batch_size"],
        max_len=cfg["model"]["context_length"],
        stride=cfg["training"]["stride"]
    )
    val_loader = create_dataloader(
        val_text, tokenizer,
        batch_size=cfg["training"]["batch_size"],
        max_len=cfg["model"]["context_length"],
        stride=cfg["training"]["stride"],
        shuffle=False
    )
    print(f"训练批次总数: {len(train_loader)}, 验证批次总数: {len(val_loader)}")

    # 4. 初始化模型
    model = GPTModel(cfg["model"])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f}M")

    # 5. 初始化训练器并启动训练
    trainer = Trainer(model, tokenizer, train_loader, val_loader, cfg, device)
    trainer.load_checkpoint()
    trainer.train()


if __name__ == "__main__":
    main()

