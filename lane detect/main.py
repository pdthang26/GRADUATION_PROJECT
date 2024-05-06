import pickle
import time
import cv2
import numpy as np

def undistort(img, cal_dir='camera_calib/cal_pickle.p'):
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def pipeline(img, s_thresh=(180, 255), sx_thresh=(60, 255)):
    img = undistort(img)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float64)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    h_channel = hls[:, :, 0]
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

def perspective_warp(img,
                     dst_size=(1080, 720),
                     src=np.float32([(0, 0.6), (0.25, 0.32), (0.75, 0.32), (1.05, 0.6)]),
                     dst=np.float32([(0, 0.72), (0, 0), (1.05, 0), (1.05, 0.72)])):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    dst = dst * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def inv_perspective_warp(img,
                         dst_size=(1080, 720),
                         src=np.float32([(0, 0.72), (0, 0), (1.05, 0), (1.05, 0.72)]),
                         dst=np.float32([(0, 0.6), (0.25, 0.32), (0.75, 0.32), (1.05, 0.6)])):
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    dst = dst * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def get_hist(img):
    hist = np.sum(img[img.shape[0] // 2:, :], axis=0)
    return hist


def sliding_window(img, nwindows=12, margin=70, minpix=100, draw_windows=True):
    out_img = np.dstack((img, img, img)) * 255
    histogram = get_hist(img)
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    window_height = int(img.shape[0] / nwindows)
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        if draw_windows:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 0, 255), 3)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (255, 0, 0), 3)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

    return out_img, (left_fitx, right_fitx),(left_fit, right_fit), ploty


def get_curve(img, leftx, rightx):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 6 / 720 # meter per pixel
    xm_per_pix = 4 / 930 # meter per pixel
    # Fit new polynomials to left (x,y) and right (x,y) pixel points in world space
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = leftx[0]*ploty**2 + leftx[1]*ploty + leftx[2]
    right_fitx = rightx[0]*ploty**2 + rightx[1]*ploty + rightx[2]
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Evaluation point
    y_eval = np.max(ploty)

    # Calculate the left and right curvatures
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    
    # Average curvature of the lane
    average_cur = (left_curverad + right_curverad)/2

    car_pos = img.shape[1] / 2
    l_fit_x_int = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
    center = (car_pos - lane_center_position) * xm_per_pix/10
    return average_cur, center


def draw_lanes(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    color_img = np.zeros_like(img)
    offset = 400

    if left_fit is not None and right_fit is not None:
        # Both lanes detected
        left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
        points = np.hstack((left, right))
        cv2.fillPoly(color_img, np.int_(points), (0, 255, 0))
    elif right_fit is not None and left_fit is None:
        # Only right lane detected, offset its position to simulate the left lane
        right_lane_offset = right_fit - offset
        left = np.array([np.transpose(np.vstack([right_lane_offset, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
        points = np.hstack((left, right))
        cv2.fillPoly(color_img, np.int_(points), (0, 255, 0))

    inv_perspective = inv_perspective_warp(color_img)
    inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.3, 0)
    return inv_perspective

def vid_pipeline(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ = pipeline(img)
    img_ = perspective_warp(img_)
    out_img, curve_x,curves, ploty = sliding_window(img_, draw_windows=False)
    curverad,lane_curve = get_curve(img, curves[0], curves[1])
    img = draw_lanes(img, curve_x[0], curve_x[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Lane Curvature: {:.0f} m'.format(curverad), (450, 50), font, 0.7, (0, 255, 255), 2)
    cv2.putText(img, 'Vehicle offset: {:.4f} m'.format(lane_curve), (450, 80), font, 0.7, (0, 255, 255), 2)
    return out_img,img, gray

video_path = 'test (3).mp4'
cap = cv2.VideoCapture(video_path)

while True:
    start_time = time.time()
    ret, img = cap.read()
    if ret:
        img = cv2.resize(img, (1080, 720), interpolation=cv2.INTER_AREA)
        out,kp, detection_gray = vid_pipeline(img)
        cv2.putText(kp, f'Processing Time: {round((time.time() - start_time), 3)} Seconds', (450, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Lane Detection',kp)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
