import os
import signal
import torch
import torch.nn.functional as F
from typing import Optional, List
from torch.optim import AdamW
from torch.amp import GradScaler

from ..model import GPTModel
from ..data import BpeTokenizer
from .scheduler import get_lr_scheduler
from ..utils.generation import generate_stream


class Trainer:
    """GPT 模型训练器，封装完整训练、评估、断点管理逻辑"""

    def __init__(
        self,
        model: GPTModel,
        tokenizer: BpeTokenizer,
        train_loader,
        val_loader,
        cfg: dict,
        device: torch.device
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"]
        )

        total_steps = len(train_loader) * cfg["training"]["num_epochs"]
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            total_steps,
            warmup_ratio=cfg["training"]["warmup_ratio"],
            min_lr_ratio=cfg["training"]["min_lr_ratio"]
        )

        # 混合精度训练
        self.use_amp = cfg["training"]["use_amp"] and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp) if self.use_amp else None

        # 训练状态
        self.start_epoch = 0
        self.global_step = 0
        self.train_loss_hist: List[float] = []
        self.val_loss_hist: List[float] = []
        self.token_hist: List[int] = []
        self.tokens_seen = 0
        self.grad_accum_steps = cfg["training"]["grad_accum_steps"]

        # 退出标记
        self.should_exit = [False]
        self._register_exit_signal()

    def _register_exit_signal(self):
        """注册退出信号处理，Ctrl+C/终止进程时自动保存断点"""
        def exit_handler(signum, frame):
            print("\n⚠️  检测到退出信号，正在保存断点，请稍候...")
            self.should_exit[0] = True

        signal.signal(signal.SIGINT, exit_handler)
        # Windows 不支持 SIGTERM，做兼容处理
        try:
            signal.signal(signal.SIGTERM, exit_handler)
        except AttributeError:
            pass

    def _calc_loss_batch(self, inp: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        inp, tgt = inp.to(self.device), tgt.to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp
        ):
            logits = self.model(inp)
            loss = F.cross_entropy(logits.flatten(0, 1), tgt.flatten(0, 1))
        return loss

    def _evaluate(self, num_batches: Optional[int] = None) -> float:
        self.model.eval()
        total_loss = 0.0
        total_steps = len(self.val_loader) if num_batches is None else min(num_batches, len(self.val_loader))
        if total_steps == 0:
            return float("nan")

        with torch.no_grad():
            for i, (inp, tgt) in enumerate(self.val_loader):
                if i >= total_steps:
                    break
                loss = self._calc_loss_batch(inp, tgt)
                total_loss += loss.item()

        self.model.train()
        return total_loss / total_steps

    def save_checkpoint(self):
        """保存完整训练断点"""
        save_path = self.cfg["path"]["checkpoint_path"]
        torch.save({
            "epoch": self.start_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_loss_hist,
            "val_losses": self.val_loss_hist,
            "token_track": self.token_hist
        }, save_path)

    def load_checkpoint(self) -> bool:
        """加载断点，返回是否加载成功"""
        save_path = self.cfg["path"]["checkpoint_path"]
        if not os.path.exists(save_path):
            return False

        print(f"\n发现断点文件 {save_path}，正在恢复训练...")
        try:
            ckpt = torch.load(save_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.start_epoch = ckpt["epoch"]
            self.global_step = ckpt["global_step"]
            self.train_loss_hist = ckpt.get("train_losses", [])
            self.val_loss_hist = ckpt.get("val_losses", [])
            self.token_hist = ckpt.get("token_track", [])
            self.tokens_seen = self.token_hist[-1] if self.token_hist else 0
            print(f"已恢复至第 {self.start_epoch} 轮，全局步数 {self.global_step}")
            return True
        except Exception as e:
            print(f"断点加载失败，将从头开始训练：{e}")
            return False

    def train(self):
        """启动完整训练流程"""
        cfg = self.cfg["training"]
        input("\n按回车键开始训练（中途Ctrl+C会自动保存断点）：")

        try:
            for epoch in range(self.start_epoch, cfg["num_epochs"]):
                if self.should_exit[0]:
                    break
                self.model.train()

                for batch_idx, (inp_batch, tgt_batch) in enumerate(self.train_loader):
                    if self.should_exit[0]:
                        break

                    loss = self._calc_loss_batch(inp_batch, tgt_batch)
                    loss = loss / self.grad_accum_steps

                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # 梯度累积步更新
                    if (batch_idx + 1) % self.grad_accum_steps == 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                cfg["grad_clip_norm"]
                            )
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                cfg["grad_clip_norm"]
                            )
                            self.optimizer.step()

                        self.scheduler.step()
                        self.optimizer.zero_grad()

                    self.tokens_seen += inp_batch.numel()
                    self.global_step += 1

                    # 定期评估 + 保存
                    if self.global_step % cfg["eval_freq"] == 0:
                        tr_loss = self._evaluate(cfg["eval_iter"])
                        va_loss = self._evaluate(cfg["eval_iter"])
                        self.train_loss_hist.append(tr_loss)
                        self.val_loss_hist.append(va_loss)
                        self.token_hist.append(self.tokens_seen)
                        cur_lr = self.optimizer.param_groups[0]["lr"]

                        print(
                            f"【Epoch {epoch + 1:2d} Step {self.global_step:06d}】"
                            f" Train Loss: {tr_loss:.3f} | Val Loss: {va_loss:.3f} | LR: {cur_lr:.6f}"
                        )
                        self.save_checkpoint()

                # 每轮结束生成样例
                if not self.should_exit[0]:
                    print(f"\n===== Epoch {epoch + 1} 生成样例 =====")
                    generate_stream(
                        self.model,
                        "乔峰大喝一声，身形一晃",
                        self.tokenizer,
                        max_new=120,
                        device=self.device
                    )

        except Exception as e:
            print(f"\n❌ 训练异常：{e}")
            print("正在保存当前断点...")
            self.save_checkpoint()
            print("断点已保存。")
            return

        # 退出前最终保存
        if self.should_exit[0]:
            self.save_checkpoint()
            print("✅ 断点保存完成，下次启动可自动续训。")
        else:
            print("\n🎉 全部训练完成！")
            self.save_checkpoint()

