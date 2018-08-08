PyTorchを使ったCycleGANの実装
=============================

今回の課題として実装した論文は `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks <https://arxiv.org/abs/1703.10593>`_ (ICCV 2017)である。

このプロジェクトに置いてある実装は２つある。一つはトロント大学 「CSC321 Winter 2018」という授業の課題４ [1]_ の説明に沿って実装した生成ネットワークと識別ネットワークを大幅に軽くした簡単バージョンのCycleGANで（simplified）、３２ｘ３２の絵文字で構成したデータセットで実験を行った。もう一つは元論文で提案したモデルの完全バージョンを実装したもので（cyclegan）、論文で引用された「facades」という２５６ｘ２５６の画像で構成したデータセットを使って実験を行った。

====================
各ディレクトリの説明
====================

++++++++++++
simplified
++++++++++++

トロント大の授業課題に沿って実装した軽いバージョンCycleGAN。もともとGANの実装を触ったことがなかったのと、論文で提案したモデルはかなり重く、実験で使われたデータセット「facades」を研究室のGTX 1080Tiに200エポック回すだけで８時間以上かかるので、最初は簡易バージョンのCycleGANを実装した。使われたデータセットはAppleやWindowsの絵文字各２０００個あまりを含まって、つまり実験の内容はApple風とWindows風の絵文字のスタイル変換。

* モデルのアーキテクチャ [1]_

 * 生成ネットワーク

  .. image:: samples/assets/simplified_generator.png

 * 識別ネットワーク

  .. image:: samples/assets/simplified_discriminator.png

* 実行方法

  80000エポックを回す。画像のサイズが小さいので、トレーニングはGTX 1080Tiで一時間あまりに完了できる。

  ``cd simplified``

  ``python cycle_gan.py --use_cycle_consistency_loss --train_iters=80000``

* 実験結果

 * Apple -> Windows

    .. image:: samples/simplified/sample-080000-X-Y.png

 * Windows -> Apple

    .. image:: samples/simplified/sample-080000-Y-X.png

 * Apple -> Windows -> Apple

    .. image:: samples/simplified/sample-080000-reconstructed-X.png

 * Windows -> Apple -> Windows

    .. image:: samples/simplified/sample-080000-reconstructed-Y.png

++++++++
cyclegan
++++++++

元論文に沿って実装したフルバージョン。実装したときに、元論文だけじゃなくて、[2]_ と [3]_ のソースコードも参照した。公式実装のなかに論文で言及しなかった変更があるので、それらの部分はソースコードに準じて実装した。

* 実行方法

 これらのコマンドにより「facades]のデータセットで200エポックの学習を行う。トレーニングを開始する前に生成ネットワークと識別ネットワークのアーキテクチャやパラメータを出力する。

  ``cd data && ./download_cyclegan_dataset.sh facades``

  ``cd ../cyclegan && python train.py``

* 実験結果

 .. image:: samples/full-spec/some.png 

 上から下までの順番は下記のようになる。

 * facade
 * facade -> label
 * label
 * label -> facade
 * facade -> label -> facade
 * label -> facade -> label

++++
data
++++

 フルバージョンのCycleGANに使うデータ。``download_cyclegan_dataset.sh`` は論文著者の実装 [2]_ からコピーしたデータセットをダウンロードする用のスクリプトである。``./download_cyclegan_dataset.sh`` で実行すれば利用できるデータセットの名前を出力してくれる。``./download_cyclegan_dataset.sh [dataset_name]`` で実行すれば自動的にデータセットをダウンロードし、前処理を行ってくれる。

+++++++
samples
+++++++

 実験結果やモデルのアーキテクチャの画像。

================
参考になったもの
================
.. [1] `CSC321 Winter 2018 Assignment 4 <https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf>`_

.. [2] `元論文と一緒に配布されたPyTorchによる実装 <https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix>`_

.. [3] `いろんな種類のGANを実装したレポジトリ <https://github.com/eriklindernoren/PyTorch-GAN>`_

---------------
