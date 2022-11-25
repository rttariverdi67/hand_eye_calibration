import cv2, os
import matplotlib.pyplot as plt
import numpy as np

def ruco_pose(color_dir, camera='azure'):

    assert camera in ['azure', 's20'], print('camera must be one of ["azure", "s20"]')

    if camera=='azure':
        camera_params = np.load('azure.npy', allow_pickle=True).tolist()
        camera_matrix = camera_params['intrinsics']
        dist_coeff = camera_params['dist_coeff']
    else :
        camera_params = np.load('S20_camera_params.npy', allow_pickle=True).tolist()
        camera_matrix = camera_params['intrinsics']
        dist_coeff = camera_params['dist_coeff']

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
    # Note: Pattern generated using the following link
    # https://calib.io/pages/camera-calibration-pattern-generator
    board = cv2.aruco.CharucoBoard_create(14, 9, 0.03975, 0.02975, aruco_dict)
    # (columns:squaresX, rows:squaresY, Checker Width: squareLength, markerLength, markers type)

    all_corners, all_ids, T_arucos = [], [], []
    frame = cv2.imread(color_dir)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
    if len(corners) > 0:
        ret, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if ret > 0:
            all_corners.append(c_corners)
            all_ids.append(c_ids)
    imsize = gray.shape

    all_corners = [x for x in all_corners if len(x) >= 4]
    all_ids = [x for x in all_ids if len(x) >= 4]
    ret, _, dist_coeff_, rvec, tvec = cv2.aruco.calibrateCameraCharuco(
        all_corners, all_ids, board, imsize, None, None)

    ret, p_rvec, p_tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners=c_corners,
                                                                charucoIds=c_ids,
                                                                board=board,
                                                                cameraMatrix=camera_matrix,
                                                                distCoeffs=dist_coeff, 
                                                                rvec=0*rvec[0], 
                                                                tvec=0*tvec[0],
                                                                )
    


    return ret, p_rvec, p_tvec, frame, camera_matrix, dist_coeff




def plot_pose(color_dir, save_fig=False, camera='azure'):
    ret, p_rvec, p_tvec, frame, camera_matrix, dist_coeff = ruco_pose(color_dir, camera=camera)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    assert ret == True, print('No marker detected, check imput image!')

    ax = cv2.drawFrameAxes(frame,
                            camera_matrix,
                            dist_coeff,
                            p_rvec,
                            p_tvec,
                            0.1)
    
    if save_fig:
        plt.figure(figsize=(12,8)) 
        plt.imshow(ax) 
        plt.savefig(f'figs/{os.path.basename(color_dir)}')                      
        plt.show()
        plt.figure().clear()
        plt.close()

    return ax, p_rvec, p_tvec




# if __name__=='__main__':
