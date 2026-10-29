from pathlib import Path

import torch
import torch.optim as optim
from torch import nn

from librivad.vad.kwt3_vad import (
    KWT,
    Attention,
    CustomDataset_train,
    FeedForward,
    PostNorm,
    PreNorm,
    Transformer,
    WarmupCosineAnnealingLR,
    count_parameters,
    load_checkpoint,
    myfun,
    number_of_correct,
    predict,
    readBatchScp,
    save_checkpoint,
)


def test_kwt_building_blocks() -> None:
    x = torch.randn(2, 10, 16)
    ff = FeedForward(dim=16, hidden_dim=32, dropout=0.0)
    pre = PreNorm(dim=16, fn=ff)
    post = PostNorm(dim=16, fn=ff)
    assert pre(x).shape == (2, 10, 16)
    assert post(x).shape == (2, 10, 16)

    attn = Attention(dim=16, heads=2, dim_head=8, dropout=0.0)
    assert attn(x).shape == (2, 10, 16)

    transformer = Transformer(
        dim=16, depth=2, heads=2, dim_head=8, mlp_dim=32, pre_norm=True, dropout=0.0
    )
    assert transformer(x).shape == (2, 10, 16)


def test_kwt_model_forward_and_helpers() -> None:
    model = KWT(
        input_res=[39, 10],
        patch_res=[39, 1],
        num_classes=2,
        dim=16,
        depth=2,
        heads=2,
        mlp_dim=32,
        channels=1,
        dim_head=8,
        dropout=0.1,
        emb_dropout=0.1,
        pre_norm=False,
    )
    assert count_parameters(model) > 0

    batch = torch.randn(3, 10, 39)
    logits = model(batch)
    assert logits.shape == (30, 2)

    preds = predict(logits)
    assert preds.shape == (30,)

    targets = preds.clone()
    assert number_of_correct(preds, targets) == 30


def test_warmup_cosine_scheduler() -> None:
    layer = nn.Linear(4, 2)
    optimizer = optim.AdamW(layer.parameters(), lr=0.01)
    scheduler = WarmupCosineAnnealingLR(optimizer, T_max=10, warmup_epochs=2, eta_min=0.001)

    initial_lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    after_warmup_lr = optimizer.param_groups[0]["lr"]
    assert after_warmup_lr >= initial_lr

    for _ in range(5):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] <= after_warmup_lr


def test_dataset_and_checkpoints(tmp_path: Path) -> None:
    ts = torch.randn(5, 10, 39)
    labels = torch.randint(0, 2, (5, 10))
    ds = CustomDataset_train(ts, labels)
    assert len(ds) == 5
    item_x, item_y = ds[0]
    assert item_x.shape == (10, 39)
    assert item_y.shape == (10,)

    model = nn.Linear(4, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    ckpt_dir = str(tmp_path / "ckpts")
    os_ckpt_path = tmp_path / "ckpts"
    os_ckpt_path.mkdir(parents=True, exist_ok=True)

    _, _, epoch0, loss0 = load_checkpoint(model, optimizer, ckpt_dir)
    assert epoch0 == 0
    assert loss0 is None

    save_checkpoint(1, model, optimizer, 0.5, checkpoint_dir=ckpt_dir, max_checkpoints=2)
    save_checkpoint(2, model, optimizer, 0.4, checkpoint_dir=ckpt_dir, max_checkpoints=2)
    save_checkpoint(3, model, optimizer, 0.3, checkpoint_dir=ckpt_dir, max_checkpoints=2)

    pth_files = list(os_ckpt_path.glob("*.pth"))
    assert len(pth_files) == 2

    _, _, loaded_epoch, loaded_loss = load_checkpoint(model, optimizer, ckpt_dir)
    assert loaded_epoch == 3
    assert loaded_loss == 0.3


def test_myfun_and_read_batch_scp(tmp_path: Path) -> None:
    prefix = tmp_path / "utt1"
    feat = torch.randn(20, 39, dtype=torch.float32)
    lab = torch.ones((20, 1), dtype=torch.float32)
    torch.save(feat, f"{prefix}.pt")
    torch.save(lab, f"{prefix}.lab.pt")

    line = f"utt1.wav,utt1.npy,{prefix}"
    f_out, l_out = myfun(line)
    assert not isinstance(f_out, str)
    assert not isinstance(l_out, str)
    assert f_out.shape == (20, 39)
    assert l_out.shape == (20, 1)

    bad_f, bad_l = myfun("missing.wav,missing.npy,/nonexistent/path")
    assert bad_f == "None" and bad_l == "None"

    batch_feat, batch_lab = readBatchScp([line], num_cores=1, sequence_length=10)
    assert batch_feat.shape == (2, 10, 39)
    assert batch_lab.shape == (2, 10)
