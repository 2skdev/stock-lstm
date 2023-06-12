# https://qiita.com/seiji1997/items/bee6b75b4b461ee3a7b2

import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import pandas_ta as ta
import torch
import torch.nn.functional as F
import yfinance as yf
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

if __name__ == '__main__':
  # データを取得
  yf.pdr_override()
  end = datetime.date.today() + datetime.timedelta(days=1)
  start = end - datetime.timedelta(days=3000)
  # symbol = '^N225'
  symbol = '7751.T'
  df = pdr.get_data_yahoo(symbol, start, end)
  df.info()

  # Closeのみを残す
  df = df[['Close']]

  # 25日MAを計算
  df['25MA'] = df['Close'].rolling(window=25, min_periods=0).mean()

  # MAを正規化
  scaler = StandardScaler()
  ma = df['25MA'].values.reshape(-1, 1)
  ma_std = scaler.fit_transform(ma)

  # データセットを作る
  data = [] # 入力
  label = [] # 正解
  for i in range(len(ma_std) - 25):
    data.append(ma_std[i: i + 25])
    label.append(ma_std[i + 25])

  data = np.array(data)
  label = np.array(label)
  print(f'Data size  : {data.shape}')
  print(f'Label size : {label.shape}')

  # 訓練データとテストデータに分割
  test_len = 252
  train_len = data.shape[0] - test_len

  train_data = data[:train_len]
  train_label = label[:train_len]
  test_data = data[train_len:]
  test_label = label[train_len:]
  print(f'Train data size  : {train_data.shape}')
  print(f'Train label size : {train_label.shape}')
  print(f'Test data size   : {test_data.shape}')
  print(f'Test label size  : {test_label.shape}')

  # Tensorに変換
  train_x = torch.Tensor(train_data)
  test_x = torch.Tensor(test_data)
  train_y = torch.Tensor(train_label)
  test_y = torch.Tensor(test_label)
  train_dataset = TensorDataset(train_x, train_y)
  test_dataset = TensorDataset(test_x, test_y)

  train_batch = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=2)
  test_batch = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False, num_workers=2)

  # モデル構築
  class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
      super(Net, self).__init__()
      self.lstm = nn.LSTM(D_in, H, batch_first=True, num_layers=1)
      self.linear = nn.Linear(H, D_out)
    
    def forward(self, x):
      output, (hidden, cell) = self.lstm(x)
      output = self.linear(output[:, -1, :])
      return output

  # デバイス
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Device: {device}')

  # ハイパーパラメータ
  D_in = 1 # 入力次元
  H = 200 # 隠れ層次元
  D_out = 1 # 出力次元
  epoch = 100 # 学習回数

  # ネットワーク
  if True:
    net = torch.load('net_50.pth')
    i_start = 50
    is_train = False
  else:
    net = Net(D_in, H, D_out).to(device)
    i_start = 0
    is_train = True

  # 損失関数、最適化関数
  criterion = nn.MSELoss() # 平均二乗誤差
  optimizer = optim.Adam(net.parameters())

  # 学習の実行
  train_loss_list = [] # 学習損失
  test_loss_list = [] # 評価損失

  def train():
    # 損失の初期化
    train_loss = 0

    net.train()
    for data, label in train_batch:
      data = data.to(device)
      label = label.to(device)

      # 勾配を初期化
      optimizer.zero_grad()
      # 予測値を計算(順伝播)
      y_pred = net(data)
      # 損失を計算
      loss = criterion(y_pred, label)
      # 勾配の計算(逆伝播)
      loss.backward()
      # パラメータの更新
      optimizer.step()
      # ミニバッチごとの損失を蓄積
      train_loss += loss.item()
    
    return train_loss

  def test():
    # 損失の初期化
    test_loss = 0

    # 結果
    pred_ma = []
    true_ma = []

    net.eval()
    with torch.no_grad():
      for data, label in test_batch:
        data = data.to(device)
        label = label.to(device)

        # 予測値を計算(順伝播)
        y_pred = net(data)
        # 損失を計算
        loss = criterion(y_pred, label)
        # ミニバッチごとの損失を蓄積
        test_loss += loss.item()

        # 結果の保存
        pred_ma.append(y_pred.view(-1).tolist())
        true_ma.append(label.view(-1).tolist())

    return test_loss, pred_ma, true_ma

  def plot(i, pred_ma, true_ma):
    fig, axes = plt.subplots(1, 2)
    
    # 損失
    axes[0].set_title('Train and Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].plot(range(i_start + 1, i + 1), train_loss_list, color='blue', linestyle='-', label='Train_Loss')
    axes[0].plot(range(i_start + 1, i + 1), test_loss_list, color='red', linestyle='--', label='Test_Loss')
    axes[0].legend()

    # 予測
    pred_ma = scaler.inverse_transform([[e for l in pred_ma for e in l]])
    true_ma = scaler.inverse_transform([[e for l in true_ma for e in l]])
    date = df.index[-1 * test_len:]
    test_close = df['Close'][-1 * test_len:].values.reshape(-1)
    axes[1].set_title('Prediction')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Stock Price')
    axes[1].plot(date, test_close, color='black', linestyle='-', label='close')
    axes[1].plot(date, true_ma[0], color='dodgerblue', linestyle='--', label='true_25MA')
    axes[1].plot(date, pred_ma[0], color='red', linestyle=':', label='predicted_25MA')
    axes[1].legend()

    # 表示
    plt.savefig(f'{i + 1}.png')

  if is_train == True:
    for i in range(i_start, epoch):
      print('-' * 50)
      print(f'Epoch: {i + 1}/{epoch}')

      # 開始時間
      start_time = time.time()

      ### 学習
      train_loss = train()

      # 平均損失を計算
      batch_train_loss = train_loss / len(train_batch)

      ### 評価
      test_loss, pred_ma, true_ma = test()

      # 平均損失を計算
      batch_test_loss = test_loss / len(test_batch)

      ### グラフ化
      plot(i, pred_ma, true_ma)

      # モデルを保存
      if (i + 1) % 10 == 0:
        torch.save(net, f'net_{i + 1}.pth')

      # 終了時間
      end_time = time.time()

      # エポックごとに損失を表示
      print(f'End {end_time - start_time: .2f}s. Train loss: {batch_train_loss:.2E} Test loss: {batch_test_loss:.2E}')

      # 損失をリストに追加
      train_loss_list.append(batch_train_loss)
      test_loss_list.append(batch_test_loss)
  else:
    ### 評価
    _, pred_ma, true_ma = test()

    # 平均絶対誤差を計算
    mae = mean_absolute_error(true_ma[0], pred_ma[0])
    print(f'MAE: {mae:.3f}')

    plot(0, pred_ma, true_ma)
    plt.show()
