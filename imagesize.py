# https://qiita.com/okm_6880/items/0f964484d570208d715d
import cv2
import numpy as np

# パラメータ
blur = 11  # ぼかし
x_dis, y_dis = (145, 110)  # ARマーカー間の実寸
size = 3  # 表示画像サイズ＝ARマーカー間の実寸×size
th = 130  # 閾値の初期値

cap = cv2.VideoCapture(2)  # カメラ番号取得
aruco = cv2.aruco


def nothing(x):
    pass


cv2.namedWindow('binary')
cv2.createTrackbar('threshold', 'binary', th, 256, nothing)

while True:
    try:
        ret, img = cap.read()  # 戻り値 = ,カメラ画像
        p_dict = aruco.getPredefinedDictionary(
            aruco.DICT_4X4_50)  # ArUcoマーカーのdict取得(50ピクセル)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            img, p_dict)  # カメラ画像からArUcoマーカー検出

        # 時計回りで左上から順に表示画像の座標をmに格納
        m = np.empty((4, 2))  # [x,y]の配列を4点分
        corners2 = [np.empty((1, 4, 2))]*4
        for i, c in zip(ids.ravel(), corners):
            corners2[i] = c.copy()
        m[0] = corners2[0][0][2]  # マーカー0の右下
        m[1] = corners2[1][0][3]  # マーカー1の左下
        m[2] = corners2[2][0][0]  # マーカー2の左上
        m[3] = corners2[3][0][1]  # マーカー3の右上

        width, height = (x_dis*size, y_dis*size)  # 変形後画像サイズ
        x_ratio = width/x_dis
        y_ratio = height/y_dis

        marker_coordinates = np.float32(m)
        true_coordinates = np.float32(
            [[0, 0], [width, 0], [width, height], [0, height]])
        mat = cv2.getPerspectiveTransform(
            marker_coordinates, true_coordinates)  # 画像サイズを任意の大きさに合わせる
        img_trans = cv2.warpPerspective(img, mat, (width, height))
        tmp = img_trans.copy()

        # グレースケール変換
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', tmp)

        # ぼかし処理
        tmp = cv2.GaussianBlur(tmp, (blur, blur), 0)
        # cv2.imshow('blur', tmp)

        # 二値化処理
        gamma = cv2.getTrackbarPos('threshold', 'binary')
        th = cv2.getTrackbarPos('threshold', 'binary')
        _, tmp = cv2.threshold(tmp, th, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow('binary', tmp)

        # 輪郭検出
        contours, hierarchy = cv2.findContours(
            tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        n, img_label, data, center = cv2.connectedComponentsWithStats(tmp)

        x, y = box[0]
        center = int(center[0][0]), int(center[0][1])
        angle = rect[2]
        scale = 1.0
        mat = cv2.getRotationMatrix2D(center, angle, scale)
        img_trans = cv2.warpAffine(img_trans, mat, (width, height))  # アフィン変換

        color_lower = np.array([0, 0, 0])  # 抽出する色の下限(BGR形式)
        color_upper = np.array([0, 0, 0])  # 抽出する色の上限(BGR形式)
        img_mask = cv2.inRange(img_trans, color_lower,
                               color_upper)  # 範囲からマスク画像を作成
        img_trans = cv2.bitwise_not(
            img_trans, img_trans, mask=img_mask)  # 元画像とマスク画像の演算(背景を白くする)
        img_trans_mesure = img_trans.copy()
        img_trans = cv2.cvtColor(img_trans, cv2.COLOR_BGR2GRAY)

        # ぼかし処理
        img_trans = cv2.GaussianBlur(img_trans, (blur, blur), 0)

        # 二値化処理
        gamma = cv2.getTrackbarPos('threshold', 'binary')
        th = cv2.getTrackbarPos('threshold', 'binary')
        _, img_trans = cv2.threshold(img_trans, th, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(
            img_trans, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        img_trans_mesure = cv2.rectangle(
            img_trans_mesure, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.putText(img_trans_mesure, "width={:.1f}mm".format(
            w/x_ratio), (int(0), int(30)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.putText(img_trans_mesure, "hight={:.1f}mm".format(
            h/y_ratio), (int(0), int(50)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.imshow('raw', img)
        cv2.imshow('image', img_trans_mesure)
        print(w/x_ratio, h/y_ratio)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    # ARマーカーが隠れた場合のエラーを無視する
    except ValueError:
        print("ValueError")
    except IndexError:
        print("IndexError")
    except AttributeError:
        print("AttributeError")

cv2.destroyAllWindows()
