#ToughDesignerのCHOP用の処理
#入力EEG → VAを返す
import numpy as np

def onSetupParameters(scriptOp):
    return

def onCook(scriptOp):
    scriptOp.clear()
    scriptOp.isTimeSlice = False
    scriptOp.rate = 60
    ps = parent.ps                      

    # 入力の4ch新着サンプルをバッファへ（無入力なら追加しない）
    # ※ Script CHOP に numInputs は無い。接続入力数は len(scriptOp.inputs) で見る
    if len(scriptOp.inputs) > 0:
        arr = scriptOp.inputs[0].numpyArray()   # shape (numChans, numSamples)
        if arr is not None and arr.size:
            ps.Push(arr)

    # 現在の V/A（4秒未充填なら [0.5,0.5] 付近が返る）
    va = ps.Predict()
    v = float(va[0]); a = float(va[1])

    scriptOp.numSamples = 1
    scriptOp.appendChan("valence").copyNumpyArray(np.array([v], dtype=np.float32))
    scriptOp.appendChan("arousal").copyNumpyArray(np.array([a], dtype=np.float32))
