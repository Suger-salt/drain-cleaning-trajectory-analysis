### 作業環境
os：ubuntu 24.02 LTS


### lerobotの動作確認コード



`$ lerobot-find-port`

  接続してるポート番号を確認できる。接続している教師用アームや追従用アームの一旦抜いて、enterを押せば \ttyACM0 or \ttyACM1 が得られる。
  間違えるとキャリブレーションまた必要になるから注意必要やな。


`$ lerobot-find-cameras opencv`
  接続されているカメラの情報やindexが見れる


`$ nvidia-smi`
  nvidiaのバージョン確認


`$conda list torch`
`$pip show lerobot`
  torchのバージョン確認



` sudo chmod a+rw /dev/ttyACM0 /dev/ttyACM1`
  一旦のUSBポートの権限付与

















