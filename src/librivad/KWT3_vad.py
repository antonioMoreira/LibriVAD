# ViT Code adapted from: https://github.com/lucidrains/vit-pytorch
# Original author: [Author: lucidrains, github:vit-pytorch]


import datetime
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.fft
import torch.optim as optim
from einops import rearrange
from einops.layers.torch import Rearrange
from joblib import Parallel, delayed
from numpy.typing import NDArray
from torch import einsum, nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from librivad.auc_Vit import auc_acc, split_data_into_sequences

np.random.seed(0)
torch.manual_seed(42)

device: torch.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

num_workers: int
pin_memory: bool
if device.type == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm: nn.LayerNorm = nn.LayerNorm(dim)
        self.fn: nn.Module = fn

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        return self.fn(self.norm(x), **kwargs)


class PostNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm: nn.LayerNorm = nn.LayerNorm(dim)
        self.fn: nn.Module = fn

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        return self.norm(self.fn(x, **kwargs))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim: int = dim_head * heads
        project_out: bool = not (heads == 1 and dim_head == dim)

        self.heads: int = heads
        self.scale: float = dim_head**-0.5

        self.attend: nn.Softmax = nn.Softmax(dim=-1)
        self.to_qkv: nn.Linear = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out: nn.Module = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        _b, _n, _, h = *x.shape, self.heads
        qkv: tuple[torch.Tensor, ...] = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in qkv)

        dots: torch.Tensor = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn: torch.Tensor = self.attend(dots)

        out: torch.Tensor = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        pre_norm: bool | str = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList([])

        P_Norm: type[PreNorm | PostNorm] = PreNorm if pre_norm else PostNorm

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        P_Norm(
                            dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                        ),
                        P_Norm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        for layer in self.layers:
            assert isinstance(layer, nn.ModuleList)
            attn: nn.Module = layer[0]
            ff: nn.Module = layer[1]
            x = attn(x) + x
            x = ff(x) + x
        return x


class KWT(nn.Module):
    def __init__(
        self,
        input_res: Sequence[int],
        patch_res: Sequence[int],
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        channels: int,
        dim_head: int,
        dropout: float,
        emb_dropout: float,
        pre_norm: bool | str = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.num_patches: int = int(input_res[0] / patch_res[0] * input_res[1] / patch_res[1])

        self.patch_dim: int = channels * patch_res[0] * patch_res[1]

        self.to_patch_embedding: nn.Sequential = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_res[0], p2=patch_res[1]),
            nn.Linear(self.patch_dim, dim),
        )

        self.pos_embedding: nn.Parameter = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.dropout: nn.Dropout = nn.Dropout(emb_dropout)

        self.transformer: Transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, pre_norm, dropout
        )

        self.to_latent: nn.Identity = nn.Identity()

        self.mlp_head: nn.Sequential = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [512, 100, 39]  # ty: ignore[invalid-mutable-override]
        x = x.permute(0, 2, 1)  # > [512, 39, 100]
        x = x.unsqueeze(1)  # > [512, 1, 39, 100]
        x = self.to_patch_embedding(x)  # > [512, 100, 192]
        _b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)  # > [512, 100, 192]
        x = self.to_latent(x)
        x = x.reshape(-1, x.shape[2])  # > [51200, 192])
        return self.mlp_head(x)


class WarmupCosineAnnealingLR(LambdaLR):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_max: int,
        warmup_epochs: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Initialize the WarmupCosineAnnealingLR scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer for which to
                schedule the learning rate.
            T_max (int): Total number of epochs for the cosine annealing schedule.
            warmup_epochs (int): Number of epochs for the linear warmup phase.
            eta_min (float): Minimum learning rate after annealing.
            last_epoch (int): The index of the last epoch when resuming training.
        """
        self.T_max: int = T_max
        self.warmup_epochs: int = warmup_epochs
        self.eta_min: float = eta_min

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing
                progress: float = (epoch - warmup_epochs) / (self.T_max - self.warmup_epochs)
                progress_tensor: torch.Tensor = torch.tensor(
                    progress, dtype=torch.float32
                )  # Convert to tensor with float32 dtype
                return float(
                    self.eta_min
                    + 0.5
                    * (1 - self.eta_min)
                    * (1 + torch.cos(progress_tensor * torch.pi)).item()
                )

        super().__init__(optimizer, lr_lambda, last_epoch)


def myfun(fe: str) -> tuple[NDArray[Any] | str, NDArray[Any] | str]:
    try:
        feat_tensor: torch.Tensor = torch.load(
            fe.split(",")[2].strip("\n") + ".pt", weights_only=True
        )
        label_tensor: torch.Tensor = torch.load(
            fe.split(",")[2].strip("\n") + ".lab.pt", weights_only=True
        )
        feat: NDArray[Any] = feat_tensor.detach().cpu().numpy()  # [98, 39]
        label: NDArray[Any] = label_tensor.detach().cpu().numpy()
        return (feat, label)
    except Exception:
        return ("None", "None")


def readBatchScp(
    batch_scp: Sequence[str] | NDArray[Any],
    num_cores: int,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    feat_list: list[NDArray[Any]] = []
    y_train_list: list[NDArray[Any]] = []
    X_batch: list[tuple[NDArray[Any] | str, NDArray[Any] | str]] = list(
        Parallel(n_jobs=num_cores)(
            delayed(myfun)(str(batch_scp[i])) for i in range(len(batch_scp))
        )
    )
    for t1, t2 in X_batch:
        try:
            t1_array: NDArray[np.float64] = np.array(t1, dtype=float)  # Ensure elements are numeric
            t2_value: NDArray[Any] | str = t2  # Label is numeric or None
            if (
                t2_value is not None
                and isinstance(t1, np.ndarray)
                and isinstance(t2_value, np.ndarray)
                and not np.any(np.isnan(t1_array))
                and not np.any(np.isinf(t1_array))
            ):
                X_sequences, y_sequences, _t3, _t3_seq = split_data_into_sequences(
                    t1, t2_value, sequence_length
                )
                feat_list.append(X_sequences)
                y_train_list.append(y_sequences)
        except Exception:
            _feat_array: NDArray[np.float64] = np.array(
                feat_list, dtype=float
            )  # Ensure elements are convertible to float
            _y_train_array: NDArray[np.float64] = np.array(y_train_list, dtype=float)

    feat_concat: NDArray[Any] = np.concatenate(feat_list, axis=0)
    y_train_concat: NDArray[Any] = np.concatenate(y_train_list, axis=0)

    if len(feat_concat) != len(y_train_concat):
        raise ValueError("Mismatch between feature and label batch lengths")

    feat_tensor: torch.Tensor = torch.from_numpy(feat_concat)
    y_train_tensor: torch.Tensor = torch.from_numpy(y_train_concat)

    return feat_tensor, y_train_tensor


class CustomDataset_train(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, time_series: torch.Tensor, labels: torch.Tensor) -> None:
        """
        This class creates a torch dataset.

        """
        self.time_series: torch.Tensor = time_series
        self.labels: torch.Tensor = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        time_serie: torch.Tensor = self.time_series[index]  # .clone().detach()
        label: torch.Tensor = self.labels[index]  # label

        return (time_serie, label)


def number_of_correct(pred: torch.Tensor, target: torch.Tensor) -> int | float:
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def predict(outputs: torch.Tensor) -> torch.Tensor:
    probabilities: torch.Tensor = torch.softmax(outputs, dim=1)
    predictions: torch.Tensor = torch.argmax(probabilities, dim=1)
    return predictions


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss: float | NDArray[Any],
    checkpoint_dir: str = "checkpoints",
    max_checkpoints: int = 5,
) -> None:
    # Ensure the checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Create checkpoint filename
    checkpoint_filename: str = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pth"

    # Save the checkpoint
    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, checkpoint_filename)
    print("Checkpoint saved at", checkpoint_filename)

    # Get list of all checkpoint files in the directory
    checkpoint_files: list[str] = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")],
        key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)),
    )

    # Remove older checkpoints if the number exceeds the max limit
    if len(checkpoint_files) > max_checkpoints:
        for old_checkpoint in checkpoint_files[:-max_checkpoints]:
            os.remove(os.path.join(checkpoint_dir, old_checkpoint))
            print("Removed old checkpoint:", old_checkpoint)


def load_checkpoint[M: nn.Module, O: optim.Optimizer](
    model: M,
    optimizer: O,
    checkpoint_dir: str = "checkpoints",
) -> tuple[M, O, int, Any]:
    checkpoint_files: list[str] = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")],
        key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)),
    )

    if not checkpoint_files:
        print("No checkpoint found, starting from scratch.")
        return model, optimizer, 0, None

    latest_checkpoint: str = checkpoint_files[-1]
    checkpoint_path: str = os.path.join(checkpoint_dir, latest_checkpoint)

    checkpoint: dict[str, Any] = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch: int = int(checkpoint["epoch"])
    loss: Any = checkpoint["loss"]

    print("Checkpoint loaded from", checkpoint_path, ": Resuming from epoch", epoch)
    return model, optimizer, epoch, loss


def main() -> None:
    # Hyperparameters
    mlp_dim: int = 768  # projected output dimension in FF layer
    num_classes: int = 2  # number of class
    dropout: float = 0.1
    batch_size: int = 32
    depth: int = 12  # number of encoder layer in transformer
    heads: int = 3  # number of head in Multi-head attension
    dim: int = 192  # linear transformation nn.Linear(..., dim) of patch dimension (embedding)
    _pool: str = "cls"
    channels: int = 1
    dim_head: int = dim // heads  # multi-head dimension
    emb_dropout: float = 0.1
    pre_norm: str = "False"

    learning_rate: float = 0.001
    weight_decay: float = 0.1
    _label_smoothing: float = 0.1
    _T_max: int = 1000  # Total number of iterations for one cycle
    _eta_min: int = 0  # Minimum learning rate
    sequence_length: int = 100
    input_feat: list[int] = [39, sequence_length]
    patch_size: list[int] = [39, 1]
    num_cores: int = 10

    model: KWT = KWT(
        input_feat,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels,
        dim_head,
        dropout,
        emb_dropout,
        pre_norm,
    )
    model.to(device)
    # print(model)

    # Define loss function and optimizer
    optimizer: optim.AdamW = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # reduce the learning after 20 epochs by a factor of

    scheduler: WarmupCosineAnnealingLR = WarmupCosineAnnealingLR(
        optimizer, T_max=100, warmup_epochs=10, eta_min=0.0001
    )
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    with open("phase.txt") as phase:
        content: str = phase.read()
    Trn: str = content.rstrip()

    with open("dir.txt") as fpath:
        content = fpath.read()
    model_dir: str = content.rstrip()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    n_epoch: int = int(np.loadtxt("epoch.txt"))

    if str(Trn) == "Train":
        trnScp: NDArray[np.str_] = np.genfromtxt("DNN.trn.scp", dtype="str")
        print("number of training files: ", trnScp.shape[0])
        uBatch_size: int = 10000
        fname: str = model_dir + "/training_loss.txt"
        start_epoch: int = 0
        _total_epochs: int = n_epoch
        model, optimizer, start_epoch, _loss = load_checkpoint(model, optimizer, model_dir)
        mode_str: str = "w" if start_epoch == 0 else "a+"
        with open(fname, mode_str) as fp:
            for epoch in range(0, n_epoch):
                model.train()
                # shuffle the data
                idx: NDArray[np.intp] = np.arange(0, len(trnScp))
                np.random.shuffle(idx)
                et: datetime.datetime = datetime.datetime.now()
                trainingLoss: list[NDArray[Any]] = []

                for i in range(0, len(trnScp), uBatch_size):  # utterance at a time (say)/batch
                    newidx: NDArray[np.intp] = idx[i : i + uBatch_size]
                    # get files for the index of current batch
                    batch_scp: NDArray[np.str_] = trnScp[newidx]
                    X_batch, y_train = readBatchScp(batch_scp, num_cores, sequence_length)
                    train_dataset: CustomDataset_train = CustomDataset_train(X_batch, y_train)
                    trainloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
                        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
                    )
                    for batch, labels in trainloader:
                        optimizer.zero_grad()
                        if batch.shape[0] > 1:
                            labels_np: NDArray[np.intp] = np.array(labels, int)
                            labels_dev: torch.Tensor = torch.from_numpy(labels_np).to(device)
                            batch_dev: torch.Tensor = batch.to(device)
                            output: torch.Tensor = model(batch_dev)
                            loss_val: torch.Tensor = criterion(
                                output, labels_dev.reshape(-1, 1).squeeze(1)
                            )  # ([51200]
                            loss_val.backward()
                            optimizer.step()
                            trainingLoss.append(loss_val.detach().cpu().numpy())
                scheduler.step()
                with open(os.path.join(f"{model_dir}/model_epoch_{epoch + 1}.model"), "wb") as m_fp:
                    torch.save(model.state_dict(), m_fp)
                # Save checkpoint after each epoch
                save_checkpoint(
                    epoch + 1, model, optimizer, float(np.mean(trainingLoss)), model_dir, 1
                )
                # Dev set
                model.eval()
                noises: list[str] = [
                    "SSN_noise",
                    "Domestic_noise",
                    "Nature_noise",
                    "Office_noise",
                    "Public_noise",
                    "Street_noise",
                    "Transport_noise",
                    "Babble_noise",
                    "City_noise",
                ]
                snr: list[str] = ["-5", "0", "5", "10", "15", "20"]
                devscp_list: list[NDArray[np.str_]] = []
                for nx in noises:
                    for db in snr:
                        tmp: NDArray[np.str_] = np.genfromtxt(
                            f"dev-clean/{nx}_{db}.lst", dtype="str"
                        )
                        devscp_list.append(tmp)
                tmp = np.genfromtxt("dev-clean/clean_clean.lst", dtype="str")
                devscp_list.append(tmp)
                devscp: Any = np.asmatrix(np.squeeze(np.concatenate(devscp_list, axis=0)))
                devLoss: list[NDArray[Any]] = []
                for i in range(0, len(devscp), uBatch_size):  # utterance at a time (say)/batch
                    newidx = idx[i : i + uBatch_size]
                    # get files for the index of current batch
                    batch_scp = trnScp[newidx]
                    X_batch, y_train = readBatchScp(batch_scp, num_cores, sequence_length)
                    _dev_dataset: CustomDataset_train = CustomDataset_train(X_batch, y_train)
                    devloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
                        train_dataset, batch_size=batch_size, shuffle=False, num_workers=0
                    )
                    for batch, labels in devloader:
                        optimizer.zero_grad()
                        if batch.shape[0] > 1:
                            labels_np = np.array(labels, int)
                            labels_dev = torch.from_numpy(labels_np).to(device)
                            batch_dev = batch.to(device)
                            output = model(batch_dev)
                            devloss: torch.Tensor = criterion(
                                output, labels_dev.reshape(-1, 1).squeeze(1)
                            )  # ([51200]
                            devLoss.append(devloss.detach().cpu().numpy())
                print(
                    f"{et} epoch  {epoch} --bsize {batch_size} -- "
                    f"--train-loss {np.mean(trainingLoss):.6f} --dev-loss {np.mean(devLoss):.6f}"
                )
                fp.write(
                    f"{et} epoch  {epoch} --bsize {batch_size} -- "
                    f"--train-loss {np.mean(trainingLoss):.6f} --dev-loss {np.mean(devLoss):.6f}\n"
                )
                fp.flush()

    elif str(Trn) == "Test":
        # eval res file-name
        with open("EvalFile.txt") as fpath1:
            content = fpath1.read()
        EvalRes: str = content.rstrip()

        print(f"model --> {model_dir}")
        PATH: str = f"{model_dir}/model_epoch_{n_epoch}.model"
        try:
            model.load_state_dict(torch.load(PATH))
        except Exception:
            model.load_state_dict(torch.load(PATH, map_location="cpu"))
            model.eval()

        noises = [
            "SSN_noise",
            "Domestic_noise",
            "Nature_noise",
            "Office_noise",
            "Public_noise",
            "Street_noise",
            "Transport_noise",
            "Babble_noise",
            "City_noise",
        ]
        snr = ["-5", "0", "5", "10", "15", "20"]
        header_list: list[list[str]] = []
        temp_score_list: list[list[float]] = []
        tmp_acc_list: list[list[float]] = []
        print("Evalset scoring...")
        finename: str = f"{model_dir}/{EvalRes}.testdata"
        with open(f"{model_dir}/{EvalRes}.testdata.frm", "w") as finenameFrm:
            # Eval-data(noise dependent)
            for nx in noises:
                for db in snr:
                    header_list.append([f"{nx},{db}"])
                    tstscp: NDArray[np.str_] = np.genfromtxt(
                        f"test-clean/{nx}_{db}.lst", dtype="str"
                    )
                    muc1, ACC1, _ndataP, _thr, _fpr, _tpr, score, _label = auc_acc(
                        tstscp, model, device, sequence_length, "True"
                    )
                    finenameFrm.write(f"{nx} {db}  {score.shape[0]}\n")
                    finenameFrm.flush()
                    temp_score_list.append([muc1])
                    tmp_acc_list.append([ACC1])
            # clean data
            tstscp = np.genfromtxt("test-clean/clean_clean.lst", dtype="str")
            muc1, ACC1, _ndataP, _thr, _fpr, _tpr, score, _label = auc_acc(
                tstscp, model, device, sequence_length, "True"
            )
            temp_score_list.append([muc1])
            tmp_acc_list.append([ACC1])
            header_list.append(["clean,clean"])

            temp_score_mat: Any = np.asmatrix(np.squeeze(np.concatenate(temp_score_list, axis=0))).T
            tmp_acc_mat: Any = np.asmatrix(np.squeeze(np.concatenate(tmp_acc_list, axis=0))).T

            header_arr: NDArray[Any] = np.squeeze(np.concatenate(header_list, axis=0))
            text_array: NDArray[Any] = header_arr.reshape(-1, 1).astype(object)
            combine: NDArray[Any] = np.hstack([text_array, temp_score_mat])
            np.savetxt(f"{finename}.auc.csv", combine, delimiter=",", fmt="%s")

            combine = np.hstack([text_array, tmp_acc_mat])
            np.savetxt(f"{finename}.acc.csv", combine, delimiter=",", fmt="%s")

    #     print('Devset scoring...')
    # development-set
    #     header = []
    #     temp_score =[]
    #     tmp_acc = []
    #     finename = model_dir + '/' + str(EvalRes)+ '.devdata'
    #     finenameFrm = open(model_dir + '/' + str(EvalRes) + '.devdata.frm', 'w')
    #     for nx in noises:
    #         for db in snr:
    #             header.append([str(nx) + ',' + str(db)])
    #             tstscp = np.genfromtxt(
    #                 'dev-clean/' + str(nx) + '_' + str(db) + '.lst', dtype='str'
    #             )
    #             muc1, ACC1, ndataP, thr, fpr, tpr, score, label = auc_acc(
    #                 tstscp, model, device, sequence_length, 'True'
    #             )
    #             finenameFrm.write("%s %s  %d\n" %(nx,db,score.shape[0]))
    #             finenameFrm.flush()
    #             temp_score.append([muc1])
    #             tmp_acc.append([ACC1])
    # clean data
    #     tstscp = np.genfromtxt('dev-clean/clean_clean.lst', dtype='str')
    #     muc1, ACC1, ndataP, thr, fpr, tpr, score, label = auc_acc(
    #         tstscp, model, device, sequence_length, 'True'
    #     )
    #     temp_score.append([muc1])
    #     tmp_acc.append([ACC1])
    #     header.append(['clean'+','+'clean'])

    #     temp_score = np.asmatrix(np.squeeze(np.concatenate(temp_score, axis=0 ))).T
    #     tmp_acc = np.asmatrix(np.squeeze(np.concatenate(tmp_acc, axis=0 ))).T

    #     header = np.squeeze(np.concatenate(header, axis=0 ))
    #     text_array = header.reshape(-1, 1).astype(object)
    #     combine = np.hstack([text_array, temp_score])
    #     np.savetxt(str(finename)+'.auc.csv', combine, delimiter=',', fmt='%s')

    #     combine = np.hstack([text_array, tmp_acc])
    #     np.savetxt(str(finename)+'.acc.csv', combine, delimiter=',', fmt='%s')
    #     finenameFrm.close()
    elif str(Trn) == "Test_ind":
        # eval res file-name
        with open("EvalFile.txt") as fpath1:
            content = fpath1.read()
        EvalRes = model_dir + "/" + content.rstrip()

        print(f"model --> {model_dir}")
        PATH = f"{model_dir}/model_epoch_{n_epoch}.model"
        try:
            model.load_state_dict(torch.load(PATH))
        except Exception:
            model.load_state_dict(torch.load(PATH, map_location="cpu"))
        model.eval()

        noises = [
            "SSN_noise",
            "Domestic_noise",
            "Nature_noise",
            "Office_noise",
            "Public_noise",
            "Street_noise",
            "Transport_noise",
            "Babble_noise",
            "City_noise",
        ]
        snr = ["-5", "0", "5", "10", "15", "20"]
        tstscp_list: list[NDArray[np.str_]] = []
        for db in snr:
            for nx in noises:  # all noises
                tstscp_list.append(np.genfromtxt(f"test-clean/{nx}_{db}.lst", dtype="str"))
        tstscp = np.squeeze(np.concatenate(tstscp_list, axis=0))
        print(tstscp)

        finename = f"{EvalRes}.testdata"
        # no clean
        muc1, ACC1, _ndataP, _xyz, _fpr, _tpr, score, label = auc_acc(
            tstscp, model, device, sequence_length, "True"
        )  # thr)
        np.savetxt(f"{finename}.score", score, fmt="%.4f")
        np.savetxt(f"{finename}.label", label, fmt="%.4f")

    elif str(Trn) == "Test_voices":
        # eval res file-name
        with open("EvalFile.txt") as fpath1:
            content = fpath1.read()
        EvalRes = content.rstrip()

        print(f"model --> {model_dir}")
        PATH = f"{model_dir}/model_epoch_{n_epoch}.model"
        try:
            model.load_state_dict(torch.load(PATH))
        except Exception:
            model.load_state_dict(torch.load(PATH, map_location="cpu"))
        model.eval()

        finename = f"{model_dir}/{EvalRes}.voices"
        with open(finename, "w") as fp:
            room: list[str] = ["rm1", "rm2", "rm3", "rm4"]
            mic: list[str] = ["mc01", "mc05"]
            dist: list[str] = ["clo", "far"]
            distract: list[str] = ["babb", "musi", "none", "tele"]
            for rm in room:
                for mik in mic:
                    for distant in dist:
                        for distr in distract:
                            cond_name: str = f"{rm}_{mik}_{distant}_{distr}"
                            file_path: str = f"list_voices/{cond_name}.lst"
                            try:
                                print([cond_name])
                                tstscp = np.genfromtxt(file_path, dtype="str")
                                muc1, ACC1, _ndataP, _xyz, _fpr, _tpr, _score, _label = auc_acc(
                                    tstscp, model, device, sequence_length, "True"
                                )  # thr)
                                fp.write(f"{[cond_name]}, {len(tstscp)}, muc, {muc1:f}\n")
                                fp.write(f"{[cond_name]}, {len(tstscp)}, acc {ACC1:f}\n")
                                fp.flush()
                            except Exception:
                                if not os.path.exists(file_path):
                                    print("File does not exist:", str(file_path))
                                elif os.path.getsize(file_path) == 0:
                                    print("File exists but is empty:", str(file_path))


if __name__ == "__main__":
    main()
