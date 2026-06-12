# TouchDesigner 組み込み手順 — Muse2 EEG → V/A 連続予測

学習済みモデル `../preprocessed_data_neurosense/va_window_model.joblib` を **TD内のPython（td-ml venv）で直接ロード**し、Script CHOP で valence/arousal(0〜1) を連続出力する。OSC往復なし。

## 0. 前提
- TD の Python は `td-ml` venv（lightgbm / scipy / joblib / scikit-learn が入っていること）。
  TDPyEnvManager で `td-ml` を active にしておく。
- このフォルダ（`va_prediction_model/touchdesigner/`）のフルパスを控える。以後 `<TD_DIR>` と書く。

## 1. 全体グラフ
```
Muse2 ─(Mind Monitor / WiFi / OSC /muse/eeg 256Hz)→ [oscin1] → [rename1] → [VA_Infer] → [null_va] → ビジュアル
```

## 2. Muse → TD（取り込み）
1. **`oscin1` (OSC In CHOP)** を作成
   - `Network Port` = **5005**
   - `Time Slice` = **On**（フレーム間の全サンプルを出力＝256Hzを取りこぼさない。必須）
2. Mind Monitor 側：PCのIPを宛先、Port **5005**、**Raw EEG（/muse/eeg）有効**で送信（同一WiFi）。
3. **`rename1` (Rename CHOP)** を `oscin1` の後に
   - `From` = `*`、`To` = `TP9 AF7 AF8 TP10`（**この順が絶対**。モデルのch順）。

> Mind Monitor が使えない場合：`muse-lsl`(BLE→LSL)＋小さなPython OSCブリッジで `/muse/eeg` を5005へ送る。下流は同じ。

## 3. 推論部品 `VA_Infer`（Base COMP）
`rename1` の後に **Base COMP** を作り名前を `VA_Infer` に。中に以下を作る。

### 3-1. COMP内のOperator
| 名前 | 種類 | 設定/配線 |
|---|---|---|
| `in1` | **In CHOP** | COMPの入力（= `rename1`）を受ける |
| `ext_va_infer` | **Text DAT** | `va_infer_ext.py` の中身を貼る（またはSync DATで参照） |
| `callbacks` | **Text DAT** | `va_infer_scriptCHOP_callback.py` の中身を貼る |
| `predict` | **Script CHOP** | 入力 = `in1`。`Callbacks DAT` = `callbacks` |
| `out1` | **Out CHOP** | `predict` を受けてCOMP外へ出す |

配線：`in1 → predict → out1`

### 3-2. COMP のパラメータ設定（VA_Infer を選択して Component Editor / 右クリック Customize）
- **Parent Shortcut** = `ps`
- **Extension 1** = `op('ext_va_infer').module.ExtVAInfer(me)`、**Promote Extension** = On
- カスタムPar（任意だが推奨。`Settings` ページ等に追加）：
  | Par名 | 型 | 既定 | 用途 |
  |---|---|---|---|
  | `Modelpath` | File | `<...>/va_window_model.joblib` | 空なら自動で `../preprocessed_data_neurosense/va_window_model.joblib` |
  | `Codedir` | Folder | `<TD_DIR>` | DATに `__file__` が無い環境用の保険（va_features.py の場所） |
  | `Calsec` | Float | `60` | キャリブ秒数 |
  | `Ema` | Float | `0.3` | 出力平滑（小=滑らか, 大=反応的） |
  | `Calibrate` | Pulse | — | 押すとキャリブ開始（下のParExecで `ps.StartCalibration()`） |
  | `Status` | String(read) | — | 状態表示（モデル/キャリブ） |

> `Calibrate`(Pulse) を押したら走らせるには、VA_Infer に **Parameter Execute DAT** を置き
> `onPulse(par)` 内で `if par.name=='Calibrate': parent.ps.StartCalibration()` とする。

### 3-3. コードの置き場所
- `va_features.py` は **この `<TD_DIR>` フォルダに置いたまま**にする（拡張が `sys.path` に追加して import）。
- `va_infer_ext.py` → `ext_va_infer` DAT、`va_infer_scriptCHOP_callback.py` → `callbacks` DAT。
  （Text DAT に貼るか、外部ファイル参照のSync DATでもよい）

## 4. 出力を使う
- `VA_Infer` の後に **`null_va` (Null CHOP)** を置く。
- ビジュアル側から `op('null_va')['valence']`、`op('null_va')['arousal']`（各 0〜1）でパラメータ/シェーダを駆動。

## 5. 使い方の流れ
1. Muse装着 → Mind Monitor送信開始 → `oscin1` に4chが来ているか確認。
2. TD起動直後はバッファが4秒たまるまで V/A ≒ 0.5。数秒で動き出す。
3. 装着して落ち着いたら `Calibrate` を1回押す（`Calsec` 秒、安静）。本人基準でz化され安定する。

## 6. データの流れ（内部）
`oscin1`(256Hz, time-slice) → `rename1` → `in1` → `predict.onCook`：
新着4chサンプルを `ps.Push()` でリングバッファへ → `ps.Predict()` が
末尾4秒を 256→128Hz間引き → `va_features.window_features`(61次元) →
z化（キャリブ or global） → `0.5*(SVR+LightGBM)` → clip[0,1] → EMA平滑 →
`valence`/`arousal` を出力 → `out1` → `null_va`。

## 7. トラブルシュート
- **値が常に0.5付近で動かない**：バッファ未充填（4秒待つ）／`oscin1` にサンプルが来ていない（ポート・WiFi・Mind MonitorのRaw EEG設定）。
- **ch順がおかしい/予測が変**：`rename1` の順序が `TP9 AF7 AF8 TP10` か確認。Mind Monitorの `/muse/eeg` は通常この順。
- **import エラー（va_features / sklearn / lightgbm）**：TDのPythonが `td-ml` venv になっているか（TDPyEnvManagerでactive化）。`Codedir` が `va_features.py` の場所を指しているか。
- **周波数特徴がずれる**：`oscin1` の `Time Slice`=On で256Hzが保たれているか（フレーム最新値だけだとレートが崩れる）。
- **モデルが見つからない**：`Modelpath` を絶対パスで指定。

## 8. 既知の精度の注意（現行モデル）
窓単位モデルは弱め（被験者非依存 valence CCC≈0.17 / arousal≈0.08）で、出力は中央寄り（regression-to-mean）。
相対的な上下の動きは使えるが絶対値は粗い。改善はリポジトリ計画 Part C（出力の分散マッチング等）で対応予定。
