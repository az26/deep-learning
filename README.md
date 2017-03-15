# deep-learning
）ローカルリポジトリを作成する
$ git init

2）ローカルリポジトリにファイルの変更点を追加（インデックスに追加）
$ git add ファイル名

3）ローカルリポジトリにインデックスに追加したファイルを登録
$ git commit -m "変更点などのコメント"

4）追加したインデックス（ファイルの変更点など）をGitHubに作成
$ git remote add origin リポジトリのURI

5）ローカルリポジトリのファイルをGitHubのリポジトリに送信
$ git push origin master

1）新しいブランチを作る
$ git branch ブランチ名

2）今あるブランチを確認する
$ git branch

3）ブランチを移動する
$ git checkout ブランチ名

4）ブランチを結合（マージ）する
※$ git checkoutで，結合したいブランチに移動して…
$ git merge 取り込むブランチ名
