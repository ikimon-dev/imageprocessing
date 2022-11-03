
import cv2
import numpy as np
# Webカメラを選択する。いくつか繋がっている場合は0の部分を変える。
cap = cv2.VideoCapture(0)
# 画像の解像度を指定
pixel = 0.178  # mm/pixel
# 長さを測る高さを指定
analysis_row = 360

while True:
    # Webカメラの画像を読み込む
    ret, frame = cap.read()

    # モノクロ化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # エッジ検出
    gray_edge = cv2.Canny(gray, 50, 150)

    # 二値化データの作成
    ret, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    # 輪郭抽出
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    max_cnt = max(contours, key=lambda x: cv2.contourArea(x))
    # 輪郭内を塗りつぶし
    pic_thresh = cv2.drawContours(gray, [max_cnt], -1, 255, -1)
    # 指定行のデータを週出
    bright_data = pic_thresh[analysis_row, :]
    # 長さを算出
    num_pix = np.count_nonzero(bright_data)
    length = round(num_pix*pixel, 1)
    print(str(length) + "mm")

    # 画像を出力
    cv2.imshow('Camera', gray_edge)
    # ESC keyが押されたらループを抜け終了する.
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()  # Webカメラを解放
cv2.destroyAllWindows()  # 画面を閉じる
