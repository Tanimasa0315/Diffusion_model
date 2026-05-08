# Diffusion Model

PyTorch で DDPM / DDIM ベースの画像生成モデルを学習・サンプリングするためのプロジェクトです。
現在は CIFAR-10 を中心に、クラス条件付き U-Net、classifier-free guidance、EMA モデルを使った生成を扱います。

## 主な内容

- CIFAR-10 / MNIST / FashionMNIST の読み込み
- 時刻埋め込み付き U-Net によるノイズ予測
- 条件付き生成と classifier-free guidance
- DDPM / DDIM サンプリング
- EMA による重みの平滑化
- 学習済みモデル、チェックポイント、設定ファイルの保存

## セットアップ

Python 環境を用意して、依存パッケージをインストールします。

```powershell
python -m pip install -r requirements.txt
```

GPU を使う場合は、環境に合った CUDA 対応版の PyTorch が入っているか確認してください。

## 使い方

### 学習

`diffusion_model_train.ipynb` を上から順に実行します。

主な設定は notebook 内で変更できます。

- `dataset_name`: `MNIST`, `FashionMNIST`, `CIFAR10`
- `epochs`: 学習エポック数
- `batch_size`: バッチサイズ
- `lr`: 学習率
- `num_timesteps`: 拡散ステップ数
- `beta_schedule_type`: `linear` または `cosine`
- `gamma`: classifier-free guidance の強さ

学習結果は `models/<モデル名>/` 以下に保存されます。

```text
models/
  CIFAR10_SimpleU-net_YYYYMMDD/
    checkpoints/
    config/
    weights/
```

### サンプリング

`diffusion_model_sampling.ipynb` を実行します。

保存済みモデルを読み込み、DDIM サンプリングで画像を生成します。通常の重み `model_final.pth` と EMA 重み `model_ema_final.pth` を切り替えて使えます。

## ディレクトリ構成

```text
src/
  dataset.py                    Dataset と transform の作成
  diffusion_mnist.py            古い DDPM / DDIM 実装
  diffusion_mnist_normalize.py  [-1, 1] 正規化向けの Diffuser 実装
  ema.py                        EMA モデル管理
  model_utils.py                モデル・Diffuser・設定の読み込み
  plot.py                       生成画像と loss の描画
  simple_unet.py                U-Net モデル定義

diffusion_model_train.ipynb     学習用 notebook
diffusion_model_sampling.ipynb  生成用 notebook
requirements.txt                必要パッケージ
```

## 現在の代表的な設定

最新の CIFAR-10 モデルでは、おおよそ以下の設定を使っています。

- Dataset: CIFAR-10
- Model: `CondSimpleUnet`
- Image size: `32 x 32`
- Channels: `3`
- Diffusion timesteps: `500`
- Beta schedule: `cosine`
- Guidance scale `gamma`: `3.0`
- Batch size: `128`
- Epochs: `10`
- EMA decay: `0.999`

## メモ

- データセットは `torchvision.datasets` から自動ダウンロードされます。
- 学習済み重みやチェックポイントはファイルサイズが大きくなりやすいです。
- `diffusion_mnist.py` には古い実装が残っており、現在は主に `diffusion_mnist_normalize.py` を使います。
