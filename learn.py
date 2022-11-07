"""
PyTorch CRNN + CTC sequence recognition model for Devanagari and English word crops.
"""
import os
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utility import CTC_BLANK_INDEX, encode_text, unicodes


class _ResidualConvBlock(nn.Module):
    """Two-layer `3x3` residual block with batch normalization (`momentum=0.2`)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch, momentum=0.2)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch, momentum=0.2)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch, momentum=0.2),
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class _DevanagariCRNNNet(nn.Module):
    """
    Residual CNN + BiLSTM network mapping `(B, 1, 32, W)` to `(B, W // 4, num_classes)`.

    Unlike Latin CRNNs that pool height all the way to `H = 1`, this backbone stops
    vertical pooling at `H = 4` (`32 -> 16 -> 8 -> 4`) and flattens `C * H = 512`
    features per horizontal step before a linear projection to `256`. Keeping 4
    vertical bands preserves spatial separation between upper matras (`े`, `ै`, `ं`),
    the middle consonant body, and lower subscript matras (`ु`, `ू`, `ृ`, `्`).
    """

    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        # (B, 1, 32, W) -> (B, 64, 16, W/2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        # (B, 64, 16, W/2) -> (B, 128, 8, W/4)
        self.stage2 = nn.Sequential(
            _ResidualConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        # (B, 128, 8, W/4) -> (B, 128, 4, W/4)
        self.stage3 = nn.Sequential(
            _ResidualConvBlock(128, 128),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            _ResidualConvBlock(128, 128),
        )
        self.vertical_proj = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )

        self.blstm1 = nn.LSTM(
            input_size=256,
            hidden_size=256,
            bidirectional=True,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(512)
        self.blstm2 = nn.LSTM(
            input_size=512,
            hidden_size=256,
            bidirectional=True,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(512)
        self.fc = nn.Linear(512, num_classes)

        # Initialize blank logit bias slightly negative so early CTC steps do not
        # collapse to all-blank predictions.
        nn.init.constant_(self.fc.bias, 0.0)
        with torch.no_grad():
            self.fc.bias[num_classes - 1] = -2.0

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_contiguous() or x.stride(1) != x.size(2) * x.size(3):
            x = x.clone(memory_format=torch.contiguous_format)
        feat = self.stem(x)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        B, C, H, T = feat.shape
        seq = (
            feat.permute(0, 3, 1, 2)
            .clone(memory_format=torch.contiguous_format)
            .view(B, T, C * H)
        )
        seq = self.vertical_proj(seq)
        out1, _ = self.blstm1(seq)
        out1 = self.ln1(out1)
        out2, _ = self.blstm2(out1)
        out2 = self.ln2(out2)
        return self.fc(out2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward_logits(x), dim=-1)


class CRNN:
    """Wrapper around `_DevanagariCRNNNet` providing CTC training, inference, and checkpoint I/O."""

    def __init__(self, xShape: Tuple[int, int, int] = (32, 192, 1)) -> None:
        self.__xShape = xShape
        self.__numClasses = len(unicodes) + 1
        self.__device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                else "cpu"
            )
        )
        self.__net = _DevanagariCRNNNet(
            num_classes=self.__numClasses, in_channels=xShape[2]
        ).to(self.__device)
        self.__predictionModelSet = True

    @property
    def device(self) -> torch.device:
        return self.__device

    def train(
        self,
        X: np.ndarray,
        Y: Union[Sequence[str], Sequence[Sequence[int]], np.ndarray],
        modelPath: Optional[str] = None,
        epochs: int = 25,
        batch_size: int = 128,
        learning_rate: float = 2.5e-3,
    ) -> List[float]:
        """Train the CRNN with `nn.CTCLoss` and return per-epoch mean loss history."""
        if modelPath and os.path.isfile(modelPath):
            self.loadWeights(modelPath)

        self.__net.train()
        optimizer = torch.optim.AdamW(
            self.__net.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        ctc_loss_fn = nn.CTCLoss(
            blank=CTC_BLANK_INDEX, reduction="mean", zero_infinity=True
        )

        x_arr = np.asarray(X, dtype=np.float32)
        if x_arr.ndim == 4 and x_arr.shape[-1] == 1:
            x_arr = x_arr[..., 0]
        x_tensor = (
            torch.from_numpy(np.ascontiguousarray(x_arr))
            .unsqueeze(1)
            .to(self.__device)
            .clone(memory_format=torch.contiguous_format)
        )

        encoded_targets: List[List[int]] = []
        for item in Y:
            if isinstance(item, str):
                encoded_targets.append(encode_text(item))
            else:
                encoded_targets.append([int(v) for v in item if int(v) >= 0])

        n_samples = len(x_tensor)
        loss_history: List[float] = []
        rng = np.random.default_rng(42)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=learning_rate * 0.05
        )

        for _epoch in range(epochs):
            perm = rng.permutation(n_samples)
            epoch_loss = 0.0
            steps = 0
            for start in range(0, n_samples, batch_size):
                batch_idx = perm[start : start + batch_size]
                xb = x_tensor[batch_idx]
                tb_list = [encoded_targets[i] for i in batch_idx]
                bsz = len(xb)

                if bsz > 8 and rng.random() < 0.5:
                    shift = int(rng.integers(-2, 3))
                    if shift != 0:
                        xb = torch.roll(xb, shifts=shift, dims=-1)

                flat_targets = [idx for seq in tb_list for idx in seq]
                target_lengths = torch.tensor(
                    [len(seq) for seq in tb_list],
                    dtype=torch.long,
                    device=self.__device,
                )
                targets_t = torch.tensor(
                    flat_targets, dtype=torch.long, device=self.__device
                )

                optimizer.zero_grad()
                logits = self.__net.forward_logits(xb)
                input_lengths = torch.full(
                    (bsz,),
                    logits.size(1),
                    dtype=torch.long,
                    device=self.__device,
                )

                # PyTorch's MPS backend does not natively implement `aten::_ctc_loss`,
                # so compute the CTC loss reduction on CPU and pass gradients back to MPS.
                if self.__device.type == "mps":
                    logits_cpu = logits.detach().cpu().requires_grad_(True)
                    log_probs = (
                        F.log_softmax(logits_cpu, dim=-1)
                        .permute(1, 0, 2)
                        .contiguous()
                    )
                    loss = ctc_loss_fn(
                        log_probs,
                        targets_t.cpu(),
                        input_lengths.cpu(),
                        target_lengths.cpu(),
                    )
                    loss.backward()
                    logits.backward(
                        logits_cpu.grad.contiguous().to(self.__device)
                    )
                else:
                    log_probs = (
                        F.log_softmax(logits, dim=-1)
                        .permute(1, 0, 2)
                        .contiguous()
                    )
                    loss = ctc_loss_fn(
                        log_probs, targets_t, input_lengths, target_lengths
                    )
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.__net.parameters(), max_norm=5.0
                )
                optimizer.step()
                epoch_loss += float(loss.item())
                steps += 1

            scheduler.step()
            loss_history.append(epoch_loss / max(1, steps))

        self.__net.eval()
        return loss_history

    def loadWeights(self, modelPath: str) -> None:
        """Load state dict from `modelPath` onto the active device."""
        if not modelPath or not os.path.isfile(modelPath):
            return
        state = torch.load(modelPath, map_location=self.__device)
        self.__net.load_state_dict(state, strict=False)
        self.__net.eval()
        self.__predictionModelSet = True

    def predict(self, img: np.ndarray) -> np.ndarray:
        """
        Run forward pass on preprocessed grayscale image tensor `img`
        (`(N, H, W, 1)` or `(H, W)`) and return framewise softmax probabilities.
        """
        if not self.__predictionModelSet:
            raise RuntimeError("Model weights need to be loaded before prediction")
        x_arr = np.asarray(img, dtype=np.float32)
        if x_arr.ndim == 2:
            x_arr = x_arr[None, :, :]
        elif x_arr.ndim == 3 and x_arr.shape[-1] == 1:
            x_arr = x_arr[None, :, :, 0]
        elif x_arr.ndim == 4 and x_arr.shape[-1] == 1:
            x_arr = x_arr[..., 0]

        self.__net.eval()
        n_total = len(x_arr)
        chunks = []
        with torch.no_grad():
            for start in range(0, n_total, 256):
                part = np.ascontiguousarray(x_arr[start : start + 256])
                xb = (
                    torch.from_numpy(part)
                    .unsqueeze(1)
                    .to(self.__device)
                    .clone(memory_format=torch.contiguous_format)
                )
                probs = self.__net(xb)
                chunks.append(probs.cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def validate(
        self, X_val: np.ndarray, Y_val: Sequence[str]
    ) -> Tuple[float, List[str]]:
        """Evaluate exact-match word accuracy on `(X_val, Y_val)`."""
        from recognizer import postProcess

        preds = self.predict(X_val)
        decoded = [postProcess(preds[i : i + 1]) for i in range(len(X_val))]
        matches = sum(1 for p, t in zip(decoded, Y_val) if p == t)
        acc = matches / max(1, len(Y_val))
        return acc, decoded

    def save_model(self, model_name: str) -> str:
        """Save model weights to `<model_name>.pt`."""
        out_path = (
            model_name
            if model_name.endswith((".pt", ".pth"))
            else model_name + ".pt"
        )
        cpu_state = {
            k: v.detach().cpu() for k, v in self.__net.state_dict().items()
        }
        torch.save(cpu_state, out_path)
        return out_path

    save = save_model
