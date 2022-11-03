import cv2
import numpy as np


def nothing(x):
    pass


if __name__ == '__main__':

    x_dis, y_dis = (237, 152)  # ARマーカー間の実寸
    size = 3  # 表示画像サイズ＝ARマーカー間の実寸×size
    th = 130  # 閾値の初期値

    cap = cv2.VideoCapture(2)
    aruco = cv2.aruco

    cv2.namedWindow('image_gray')
    cv2.createTrackbar('threshold', 'image_gray', th, 256, nothing)

    while True:

        try:
            # 1フレームずつ取得する。
            ret, frame = cap.read()
            # フレームが取得できなかった場合は、画面を閉じる
            if not ret:
                break
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

            # グレースケール画像へ変換
            gray = cv2.cvtColor(img_trans, cv2.COLOR_BGR2GRAY)

            # 2値化
            th = cv2.getTrackbarPos('threshold', 'image_gray')

            retval, bw = cv2.threshold(
                gray, th, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            _, bw = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV)

            # 輪郭を抽出
            #   contours : [領域][Point No][0][x=0, y=1]
            #   cv2.CHAIN_APPROX_NONE: 中間点も保持する
            #   cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない
            contours, hierarchy = cv2.findContours(
                bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            # 矩形検出された数（デフォルトで0を指定）
            detect_count = 0

            # 各輪郭に対する処理
            for i in range(0, len(contours)):

                # 輪郭の領域を計算
                area = cv2.contourArea(contours[i])

                # ノイズ（小さすぎる領域）と全体の輪郭（大きすぎる領域）を除外
                if area < 1e4 or 1e6 < area:
                    continue

                # 外接矩形
                if len(contours[i]) > 0:
                    rect = contours[i]
                    x, y, w, h = cv2.boundingRect(rect)
                    cv2.rectangle(img_trans, (x, y),
                                  (x + w, y + h), (0, 255, 0), 2)

                    detect_count = detect_count + 1

            # ウィンドウに出力
            cv2.imshow("Frame", frame)
            cv2.imshow("image_trans", img_trans)

            cv2.imshow("image_gray", bw)

            key = cv2.waitKey(1)
            # Escキーを入力されたら画面を閉じる
            if key == 27:
                break

            # ARマーカーが隠れた場合のエラーを無視する
        except ValueError:
            print("ValueError")
        except IndexError:
            print("IndexError")
        except AttributeError:
            print("AttributeError")

    cap.release()
    cv2.destroyAllWindows()
