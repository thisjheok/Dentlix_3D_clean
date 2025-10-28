import time
import os
import cv2
import numpy as np
import torch
from fontTools.misc.psLib import endofthingRE
from ultralytics import YOLO
from ultralytics.engine.results import Results
from collections import defaultdict, Counter
import threading
import queue

sum_execution_time = 0
sum_yolo_time = 0
sum_guide_time = 0



def get_seg_center(x_list: np.ndarray, y_list: np.ndarray) -> tuple[float, float]:
    # 빈 배열이나 점이 3개 미만인 경우 간단한 중심점 계산
    if len(x_list) < 3:
        return float(np.mean(x_list)), float(np.mean(y_list))

    # 벡터화된 계산
    n = len(x_list)
    x2 = np.roll(x_list, -1)
    y2 = np.roll(y_list, -1)

    # Cross product 계산 최적화
    cross_product = x_list * y2 - y_list * x2
    area = 0.5 * np.abs(np.sum(cross_product))

    # 영역이 0에 가까우면 단순 평균 사용 (수치적 안정성)
    if area < 1e-10:
        return float(np.mean(x_list)), float(np.mean(y_list))

    # 중심점 계산 최적화
    inv_6_area = 1.0 / (6.0 * area)
    center_x = np.abs(np.sum((x_list + x2) * cross_product)) * inv_6_area
    center_y = np.abs(np.sum((y_list + y2) * cross_product)) * inv_6_area

    return float(center_x), float(center_y)

    # 치아 순서 탐지


def find_path(d, edge, start_p):
    visit = np.zeros(len(edge))
    path = list()
    while np.any(visit == 0):
        path.append(start_p)
        visit[start_p] = 1

        if np.all(visit == 1):
            break

        if (np.sum(visit) == 1) | ((visit[edge[start_p, 1]] == 0) & (visit[edge[start_p, 2]] == 0)):
            if d[edge[start_p, 0], edge[start_p, 1]] < d[edge[start_p, 1], edge[start_p, 2]]:
                start_p = edge[start_p, 1]
            else:
                start_p = edge[start_p, 2]
        else:
            if visit[edge[start_p, 1]] == 0:
                start_p = edge[start_p, 1]
            else:
                start_p = edge[start_p, 2]

    return path


def arrangement_for_id(union_track, remove_union_idx=None):
    if not union_track:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    if remove_union_idx is None:
        remove_union_idx = []

    union_len = len(union_track)
    mask_center_list_x = np.zeros(union_len, dtype=np.float32)
    mask_center_list_y = np.zeros(union_len, dtype=np.float32)
    tooth_id_list = np.zeros(union_len, dtype=np.int32)
    bounding_box_id_list = np.zeros(union_len, dtype=np.int32)
    b_conf_list = np.zeros(union_len, dtype=np.float32)

    # 벡터화된 처리를 위한 데이터 수집
    valid_tracks = []
    result_indices = []

    for i, track in enumerate(union_track):
        if len(remove_union_idx) != 0 and i < len(remove_union_idx) and remove_union_idx[i]:
            continue
        valid_tracks.append(track)
        result_indices.append(track.get_result_idx())

    # 한번에 모든 데이터 추출
    if valid_tracks:
        # 벡터화된 좌표 추출
        centers = [track.get_xy() for track in valid_tracks]
        tooth_ids = [track.get_tooth_id() for track in valid_tracks]
        box_ids = [track.get_box_id() for track in valid_tracks]
        box_confs = [track.get_box_conf() for track in valid_tracks]

        # NumPy 배열로 한번에 할당
        centers_array = np.array(centers)
        mask_center_list_x[result_indices] = centers_array[:, 0]
        mask_center_list_y[result_indices] = centers_array[:, 1]
        tooth_id_list[result_indices] = tooth_ids
        bounding_box_id_list[result_indices] = box_ids
        b_conf_list[result_indices] = box_confs

    return mask_center_list_x, mask_center_list_y, tooth_id_list, bounding_box_id_list, b_conf_list


def read_label_file(file_path):
    """
    Read a label file and return a list of boxes.
    Each box is represented as [class, x_center, y_center, width, height].
    """
    uu = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26]
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            box = [uu[int(parts[0])]] + [float(part) for part in parts[1:]]
            boxes.append(box)
    return boxes


def find_label_file(directory, frame_num):
    import re
    """
    Find the label file in the directory that corresponds to the given frame number.
    """
    pattern = re.compile(rf"frame_{frame_num:04d}_jpg\.rf\..+\.txt")
    for file_name in os.listdir(directory):
        if pattern.match(file_name):
            return os.path.join(directory, file_name)
    return None


def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
        box1 : list of floats : [x_center, y_center, width, height]
        box2 : list of floats : [x_center, y_center, width, height]

    Returns:
        float : IoU value
    """
    # Convert to (x1, y1, x2, y2)
    box1 = [box1[0] - box1[2] * 0.5, box1[1] - box1[3] * 0.5,
            box1[0] + box1[2] * 0.5, box1[1] + box1[3] * 0.5]
    box2 = [box2[0] - box2[2] * 0.5, box2[1] - box2[3] * 0.5,
            box2[0] + box2[2] * 0.5, box2[1] + box2[3] * 0.5]

    # Calculate intersection
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area


def evaluate_detections(detected_boxes, ground_truth_boxes):
    """
    Evaluate detection boxes against ground truth boxes.
    Parameters:
    detected_boxes : list of lists : [[class, x_center, y_center, width, height], ...]
    ground_truth_boxes : list of lists : [[class, x_center, y_center, width, height], ...]
    Returns:
    dict : evaluation results including precision and recall
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    unmatched_detections = 0

    matched_gt_boxes = set()
    for det_box in detected_boxes:
        best_iou = 0
        best_gt_box = None
        for gt_box in ground_truth_boxes:
            current_iou = iou(det_box[1:], gt_box[1:])
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_box = gt_box

        if best_iou >= 0.6:
            if det_box[0] == best_gt_box[0]:
                true_positive += 1
                matched_gt_boxes.add(tuple(best_gt_box))
            else:
                false_positive += 1
        else:
            unmatched_detections += 1

    false_negative = len(ground_truth_boxes) - len(matched_gt_boxes)
    total_detected = true_positive + false_positive

    precision = true_positive / total_detected if total_detected > 0 else 0
    recall = true_positive / len(ground_truth_boxes) if len(ground_truth_boxes) > 0 else 0

    return {
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'unmatched_detections': unmatched_detections,
        'precision': precision,
        'recall': recall
    }

def extract_contours(image):
    

    # 외곽선 찾기
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 원본 이미지에 외곽선 그리기 (외곽선은 흰색으로 표시)
    contour_image = np.zeros_like(image)  # 빈 이미지 생성
    cv2.drawContours(contour_image, contours, -1, 255, 1)

    return contour_image



def extract_edges_img(image, mask_layer):
    # start = time.time()

    # kernel = np.ones((15, 15), np.uint8)
    # mask_layer = cv2.morphologyEx(mask_layer, cv2.MORPH_DILATE, kernel, iterations=1)

    # 255 값을 가진 인덱스 찾기
    indices = np.where(mask_layer == 255)

    # 최소값과 최대값 찾기
    y_min, y_max = np.min(indices[0]), np.max(indices[0])
    x_min, x_max = np.min(indices[1]), np.max(indices[1])

    # 해당 영역을 잘라내기
    image = image[y_min-5:y_max+6, x_min-5:x_max+6]
    mask_layer2 = mask_layer[y_min-5:y_max+6, x_min-5:x_max+6]

    # 이미지를 BGR에서 HSV로 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # H, S, V 채널 각각 추출
    h_channel, s_channel, v_channel = cv2.split(hsv_image)

    h_channel[h_channel < 10] = 255

    # H와 S 채널을 정규화 (0~1 사이로 변환)
    h_norm = h_channel / 179.0
    s_norm = s_channel / 255.0
    # v_norm = v_channel / 255.0

    # 방법 2: H와 S의 곱
    hs_product = (h_norm * s_norm).astype(np.float32)

    # 히스토그램 계산
    hs_hist = cv2.calcHist([hs_product], [0], mask_layer2, [256], [0, 1])

    # 최빈값(Mode) 찾기
    mode_index = np.argmax(hs_hist)
    mode_value = mode_index / 256  # 히스토그램 구간을 256으로 나눴으므로


    # Thresholding 범위 설정 (예: 최빈값을 기준으로 ±10%)
    range_percentage = 0.25  # 10%, 팬텀일 때는 60%
    lower_bound = max(mode_value - range_percentage, 0)
    upper_bound = min(mode_value + range_percentage, 1)

    # print(f"Thresholding 범위: {lower_bound} ~ {upper_bound}")

    mask = np.logical_and(hs_product >= lower_bound, hs_product <= upper_bound)
    mask = mask.astype(np.uint8) * 255  # 0-255로 변환

    mask[mask_layer2==0] = 0

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=1)

    # 연결 성분 분석을 통해 모든 성분에 라벨 부여
    num_labels, labels_im = cv2.connectedComponents(mask)

    # 각 연결 성분의 크기 계산
    component_sizes = np.bincount(labels_im.flatten())[1:]  # 첫 번째 성분은 배경이므로 제외


    # 가장 큰 연결 성분의 라벨 찾기
    largest_component_label = np.argmax(component_sizes) + 1

    largest_component_size = np.max(component_sizes)
    component_idx = np.array(np.where(component_sizes > (largest_component_size * 0.25)))
    component_labels = component_idx + 1
    # print(component_idx, component_labels)

    # 가장 큰 연결 성분만 남기기
    largest_component_mask = np.zeros_like(mask)
    # largest_component_mask[labels_im == largest_component_label] = 255
    for label in component_labels[0]:
        largest_component_mask[labels_im == label] = 255

    largest_component_mask = cv2.morphologyEx(largest_component_mask, cv2.MORPH_DILATE, kernel, iterations=1)
    largest_component_mask = cv2.morphologyEx(largest_component_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    largest_component_mask = cv2.morphologyEx(largest_component_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 가우시안 블러로 스무딩
    largest_component_mask = cv2.GaussianBlur(largest_component_mask, (11, 11), 4)

    # 임계값 적용으로 이진화 복원
    _, largest_component_mask = cv2.threshold(largest_component_mask, 127, 255, cv2.THRESH_BINARY)

    # # 내부 구멍 채우기 - 방법 1: Flood Fill 사용
    # filled_mask = largest_component_mask.copy()

    # # Flood Fill을 사용해 구멍을 채우기
    # h, w = filled_mask.shape[:2]
    # flood_fill_mask = np.zeros((h+2, w+2), np.uint8)  # Flood Fill 마스크는 원본보다 2픽셀 더 커야 함
    # cv2.floodFill(filled_mask, flood_fill_mask, (0, 0), 255)  # 외부에서 시작해 채움

    # # 채운 결과를 반전하여 내부 구멍을 채운 결과 생성
    # filled_mask = cv2.bitwise_not(filled_mask)
    # filled_mask = filled_mask | largest_component_mask  # 채운 결과와 원본 결합

    # filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_DILATE, kernel, iterations=1)


    # 결과를 원본 이미지 크기로 복원
    full_mask = np.zeros_like(mask_layer)
    full_mask[y_min-5:y_max+6, x_min-5:x_max+6] = largest_component_mask


    return full_mask


def compare_hu_moments(hu_moments1, hu_moments2):
    distance = np.linalg.norm(hu_moments1 - hu_moments2, axis=1)
    return distance


def sampling(indeces):
    num_samples = int(len(indeces[0])/2)
    indices_to_keep = np.random.choice(len(indeces[0]), num_samples, replace=False)
    return indices_to_keep


def center_point_cloud(points):
    """
    포인트 클라우드의 중심점을 원점으로 이동시킵니다.
    """
    centroid = np.mean(points, axis=1, keepdims=True)
    centered_points = points - centroid
    return centered_points, centroid

def icp(d1, d2, d1_area, d2_area, max_iterate = 10):
    # np.random.seed(42)

    # 포인트 클라우드의 중심을 맞추고 중심점을 저장합니다.
    d1, centroid1 = center_point_cloud(d1)
    d2, centroid2 = center_point_cloud(d2)

    # d1과 d2의 샘플 수 확인
    num_samples_d1 = d1.shape[1]
    num_samples_d2 = d2.shape[1]

    if num_samples_d1 > num_samples_d2:
        indices_to_keep = np.random.choice(num_samples_d1, num_samples_d2, replace=False)
        d1 = d1[:, indices_to_keep]
    else:
        indices_to_keep = np.random.choice(num_samples_d2, num_samples_d1, replace=False)
        d2 = d2[:, indices_to_keep]

    src = np.array([d1.T], copy=True).astype(np.float32)
    dst = np.array([d2.T], copy=True).astype(np.float32)

    knn = cv2.ml.KNearest_create()
    responses = np.array(range(len(d2[0]))).astype(np.float32)
    knn.train(src[0], cv2.ml.ROW_SAMPLE, responses)

    scale_factor = np.sqrt(d1_area / d2_area)

    Tr = np.array([[scale_factor, 0, 0],
                   [0, scale_factor,  0],
                   [0,         0,          1]])

    dst = cv2.transform(dst, Tr[0:2])

    D = 100
    # inliers = None
    # src_eval = None
    # dst_eval = None
    for i in range(max_iterate):
        ret, results, neighbours, dist = knn.findNearest(dst[0], 1)

        indeces = results.astype(np.int32).T
        sample_idx = sampling(indeces)
        indeces = indeces[0][sample_idx]


        T, inliers = cv2.estimateAffinePartial2D(dst[0, sample_idx], src[0, indeces], True, method=cv2.LMEDS)
        # T, inliers = cv2.estimateAffine2D(dst[0, sample_idx], src[0, indeces], True, method=cv2.LMEDS)
        # T, inliers = cv2.estimateAffinePartial2D(dst[0, sample_idx], src[0, indeces], True, method=cv2.RANSAC, maxIters=5000, ransacReprojThreshold=10)
        # T, inliers = cv2.estimateAffine2D(dst[0, sample_idx], src[0, indeces], True, method=cv2.RANSAC, ransacReprojThreshold=10)



        dst = cv2.transform(dst, T)
        Tr = np.dot(np.vstack((T,[0,0,1])), Tr)


        if D < np.mean(dist):
            # src_eval = src[0, indeces][inliers.reshape(-1,)]
            # dst_eval = dst[0, sample_idx][inliers.reshape(-1,)]
            break
        else:
            D = np.mean(dist)




    # 최종 변환 행렬을 원래 위치로 되돌리기 위해 중심점을 적용합니다.
    Tr[0:2, 2] += centroid1.flatten() - np.dot(Tr[0:2, 0:2], centroid2.flatten())

    return Tr[0:2], D

def create_directory_with_suffix(base_path, dir_name):
    # 디렉토리 경로 생성
    directory_path = os.path.join(base_path, dir_name)

    # 디렉토리가 이미 존재할 경우 숫자를 붙여서 새로운 디렉토리 이름 생성
    suffix = 1
    new_directory_path = directory_path

    # 이미 같은 이름의 디렉토리가 있으면 숫자를 붙여서 새 이름으로 변경
    while os.path.exists(new_directory_path):
        new_directory_path = f"{directory_path}_{suffix}"
        suffix += 1

    # 최종적으로 만들어진 디렉토리 경로에 디렉토리 생성
    os.makedirs(new_directory_path)

    return new_directory_path


class ToothTracker:
    def __init__(self, is_up: bool, tooth: list, tooth_num: int, implant_num: int, loaded_data: np.lib.npyio.NpzFile):
        self.u_num = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
        self.l_num = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
        self.do_init = True
        self.isUp = is_up
        self.tooth = tooth
        self.tooth_num = tooth_num
        self.implant_num = implant_num
        self.max_tooth_num = 16
        self.show_m = True
        self.show_b = True
        self.show_d = False
        self.show_n = True
        self.guding_2D = False
        self.guding_3D = False
        self.alpha = 0.5
        self.max_id_num = 20000000
        self.tooth_id = []
        # self.id_num_matching: np.ndarray = np.empty([])
        # self.id_count: np.ndarray = np.empty(0)
        # self.tracked_id: np.ndarray = np.empty(0)
        # self.exist_tooth: np.ndarray = np.empty(0, dtype=bool)
        self.current_tid_bid = {}
        self.current_bid_tid = {}
        self.most_common_box_avg = {}
        self.make_infor()
        self.test_frame_count = 60  # 초기셋팅으로 사용할 프레임 수
        self.norm_old_sep = 30  # norm과 old를 구분할 프레임 수
        self.count_tid_bid = defaultdict(lambda: defaultdict(lambda: {"count": 0, "x_sum": 0, "y_sum": 0}))
        self.count_bid_tid = defaultdict(Counter)
        self.current_result = Results(orig_img=np.array([]), path='', names={})
        # 평가변수
        self.total_true_positive = 0
        self.total_false_positive = 0
        self.total_false_negative = 0
        self.total_unmatched_detections = 0

        # 3D 모델 관련 데이터
        self.pros = loaded_data['pros']
        self.views = loaded_data['views']
        self.contour_pros = loaded_data['contour_pros']
        self.contour_length = loaded_data['contour_length']
        self.view_parameters = loaded_data['view_parameters']
        self.views_hu_moments = loaded_data['hu_moments']
        self.view_areas = loaded_data['view_areas']
        self.tranformed_image = None
        self.img_idx = None
        self.implant_x = None
        self.implant_y = None
        self.most_sim_idx = None

    def set_option(self, show_m=True, show_b=True, show_d=False, show_n=True, show_p=False,
                   guding_2D=False, guding_3D=False, alpha=0.5,
                   test_frame_count=60, norm_old_sep=30):

        self.show_m = show_m
        self.show_b = show_b
        self.show_d = show_d
        self.show_n = show_n
        self.show_p = show_p
        self.guding_2D = guding_2D
        self.guding_3D = guding_3D
        self.alpha = alpha
        self.test_frame_count = test_frame_count
        self.norm_old_sep = norm_old_sep

    def make_infor(self):  # 초기 변수 셋팅
        for i in range(self.max_tooth_num):  # tooth_id Setting
            if self.tooth[i] == 0:
                continue
            else:
                if self.isUp:
                    self.tooth_id.append(self.u_num[i])
                else:
                    self.tooth_id.append(self.l_num[i])
        # self.current_tid_bid=[0] * self.tooth_num
        self.id_count = np.zeros(self.max_id_num)
        self.id_num_matching = np.zeros(self.max_id_num)
        self.exist_tooth = np.zeros(self.tooth_num, dtype=bool)
        self.tracked_id_map = np.zeros(self.max_id_num, dtype=bool)  # track 중인 id
        # self.tracked_id = []

    class TrackObject:  # Track 객체를 관리할 클래스
        def __init__(self, box_id, box_conf, tooth_id, result_idx, center_x, center_y, is_new):
            self.__box_id = box_id
            self.__box_conf = box_conf
            self.__tooth_id = tooth_id
            self.__center_x = center_x
            self.__center_y = center_y
            self.__is_new = is_new
            self.__result_idx = result_idx  # prediction 결과의 순서

        def get_box_conf(self):
            return self.__box_conf

        def get_result_idx(self):
            return self.__result_idx

        def get_box_id(self):
            return self.__box_id

        def get_tooth_id(self):
            return self.__tooth_id

        def get_xy(self):
            return self.__center_x, self.__center_y

        def get_is_new(self):
            return self.__is_new

        def set_tooth_id(self, tooth_id):
            self.__tooth_id = tooth_id

    def preprocess(self):  # 전처리 (최적화됨)
        # 조기 반환으로 빈 결과 처리
        if len(self.current_result.boxes.cls) == 0:
            return np.array([], dtype=int), [], np.array([])

        start = time.time()

        # 상/하악 필터링 최적화
        cls_target = 1 if self.isUp else 0
        self.current_result = self.current_result[self.current_result.boxes.cls == cls_target]

        # 조기 반환으로 빈 결과 처리
        if len(self.current_result.boxes.cls) == 0:
            return np.array([], dtype=int), [], np.array([])

        # 치아 개수 제한 최적화 - 불필요한 정렬 제거
        if len(self.current_result.boxes.cls) > self.tooth_num:
            # topk 사용으로 부분 정렬만 수행
            top_conf_threshold = torch.topk(self.current_result.boxes.conf, self.tooth_num)[0][-1]
            self.current_result = self.current_result[self.current_result.boxes.conf >= top_conf_threshold]

        # 한번에 모든 데이터 추출 (GPU->CPU 전송 최소화)
        box_conf_list = self.current_result.boxes.conf.detach().cpu().numpy()
        box_id_list = self.current_result.boxes.id.detach().cpu().numpy().astype(np.int32)
        mask_list = self.current_result.masks.xy

        # 벡터화된 카운트 업데이트
        np.add.at(self.id_count, box_id_list, 1)

        # 사라진 ID 처리 최적화
        tracked_indices = np.nonzero(self.tracked_id_map)[0]  # where보다 빠름
        unique_elements = np.setdiff1d(tracked_indices, box_id_list, assume_unique=True)

        if unique_elements.size > 0:
            # 벡터화된 업데이트
            self.tracked_id_map[unique_elements] = False
            self.id_count[unique_elements] = 0
            self.id_num_matching[unique_elements] = 0

            # isin 대신 searchsorted 사용 (더 빠름)
            tooth_id_array = np.array(self.tooth_id)
            missing_teeth = self.id_num_matching[unique_elements]
            valid_mask = missing_teeth != 0
            if np.any(valid_mask):
                missing_teeth_valid = missing_teeth[valid_mask]
                element_indices = np.searchsorted(tooth_id_array, missing_teeth_valid)
                # 범위 확인
                valid_indices = (element_indices < len(tooth_id_array)) & (tooth_id_array[element_indices] == missing_teeth_valid)
                if np.any(valid_indices):
                    self.exist_tooth[element_indices[valid_indices]] = False

        end = time.time()
        elapsed_time = (end - start) * 1000
        print(f"elapsed time in preprocess(): {elapsed_time:.3f} ms")

        return box_id_list, mask_list, box_conf_list

        # new_track이랑 old_track이랑 분류

    # tracked_id List에 Id가 없는 Box면 New 있으면 Old
    def separate_detection(self, box_id_list, mask_list, box_conf_list):
        # 조기 반환으로 빈 마스크 처리
        if len(mask_list) == 0:
            return [], []

        start = time.time()
        new_track = []
        old_track = []

        # 벡터화된 중심점 계산을 위한 배치 처리
        centers = []
        for mask in mask_list:
            mask_array = np.array(mask)
            if len(mask_array) > 0:
                x_list = mask_array[:, 0]
                y_list = mask_array[:, 1]
                center_x, center_y = get_seg_center(x_list, y_list)
                centers.append((center_x, center_y))
            else:
                centers.append((0, 0))  # 빈 마스크 처리

        # 추적 상태를 한번에 확인
        is_tracked = self.tracked_id_map[box_id_list]

        # TrackObject 생성 최적화
        for i, ((center_x, center_y), is_old) in enumerate(zip(centers, is_tracked)):
            track_obj = self.TrackObject(
                box_id_list[i],
                box_conf_list[i],
                self.id_num_matching[box_id_list[i]],
                i,
                center_x,
                center_y,
                not is_old  # is_new = not is_old
            )

            if is_old:
                old_track.append(track_obj)
            else:
                new_track.append(track_obj)

        end = time.time()
        elapsed_time = (end - start) * 1000
        print(f"elapsed time in separate detection(): {elapsed_time:.3f} ms")
        return new_track, old_track

    # 새로 탐지되어, 통계를 내는 중인 Box는 Norm 이미 통계가 끝난(30Frame) Box면 Old / Old를 다시 Norm과 Old로 나누는 과정임.
    def split_box_into_norm_and_old(self, old_track):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        Parameters:
            old_track : list of TrackObject : 현재 추적 중인 치아 객체
        Returns:
            norm : 30 프레임동안 새롭게 통계를 내야하는 치아 객체
            old_track : 통계를 낸 치아 객체
        """
        if not old_track:
            return [], []

        old = []
        norm = []

        # 치아 ID 인덱스 매핑 캐시 (반복 계산 방지)
        tooth_id_to_index = {tid: idx for idx, tid in enumerate(self.tooth_id)}

        for track in old_track:
            box_id = track.get_box_id()
            tooth_id = track.get_tooth_id()
            count = self.id_count[box_id]

            if count < self.norm_old_sep:  # Norm 단계
                if tooth_id in tooth_id_to_index:
                    self.exist_tooth[tooth_id_to_index[tooth_id]] = False
                self.count_bid_tid[box_id][tooth_id] += 1
                norm.append(track)

            elif count == self.norm_old_sep:  # 임계점 - 가장 적합한 치아 ID 선택
                most_common_tid_list = self.count_bid_tid[box_id].most_common()

                # 첫 번째로 사용 가능한 치아 ID 찾기
                for tid_count in most_common_tid_list:
                    most_common_tid = tid_count[0]
                    if most_common_tid in tooth_id_to_index:
                        tid_index = tooth_id_to_index[most_common_tid]
                        if not self.exist_tooth[tid_index]:
                            track.set_tooth_id(most_common_tid)
                            self.exist_tooth[tid_index] = True
                            old.append(track)
                            break

            else:  # Old 단계
                if tooth_id in tooth_id_to_index:
                    self.exist_tooth[tooth_id_to_index[tooth_id]] = True
                old.append(track)

        return norm, old

    # 맨 처음 Tooth_id에 대한 Box_id를 통계내는 과정에서 쓰는 치아번호 식별
    def set_unino_track_1(self, new_track, old_track):
        start = time.time()
        new_len = len(new_track)
        old_len = len(old_track)
        # Visualize the results on the frame
        if new_len != 0:  # 새로운 Track이 감지됐다면
            union_track = new_track
            union_track.extend(old_track)  # union_track = new + old Track, new_len>=idx면 old이고 아니면 new이다.
            path = self.cal_path(union_track)  # 탐지된 치아의 순서 확인

            if old_len == 0:  # 만약 old가 없다면
                union_track = self.all_new_set(path, union_track)  # 새로 다시 셋팅
            else:
                most_left_old = -1
                most_left_old_idx = -1
                for i, num in enumerate(path):  # 순서상 old중 가장 왼쪽에 있는 치아 찾기
                    if num >= new_len:
                        most_left_old = num  # old중 가장 왼쪽에 있는 치아가 path에 들어있는 번호
                        most_left_old_idx = i  # old중 가장 왼쪽에 있는 치아의 path에서의 idx
                        break
                most_left_old_tooth_id = union_track[most_left_old].get_tooth_id()
                most_left_old_tooth_idx = self.tooth_id.index(
                    most_left_old_tooth_id)  # old중 가장 왼쪽에 있는 치아 왼쪽에 있는 객체가 치아번호 배열에서 몇번쨰 인덱스인지
                if most_left_old_idx > most_left_old_tooth_idx:  # Path에서의 idx가 치아번호 배열에서의 idx보다 크면 탐지 오류이다.
                    # ex) 18 17 16 15 14 13/ path에서의 4번 idx(왼쪽에 3개의 치아번호를 할당해줘야함) 치아번호배열에서 2번idx(할당할 치아번호가 2개 밖에 안남음)
                    union_track = self.all_new_set(path, union_track)  # 새로 처음부터 다 치아번호할당
                else:  # 찾은 번호를 기준으로 new를 set
                    last_old_tooth_idx = most_left_old_tooth_idx
                    last_old_idx = most_left_old_idx
                    for i, num in enumerate(path):
                        if i < most_left_old_idx:  # old중 가장 왼쪽에 있는 치아 보다 왼쪽에 있는 new들의 번호를 할당해줌
                            union_track[num].set_tooth_id(
                                self.tooth_id[most_left_old_tooth_idx - most_left_old_idx + i])
                            self.exist_tooth[most_left_old_tooth_idx - most_left_old_idx + i] = True
                            self.id_num_matching[union_track[num].get_box_id()] = union_track[num].get_tooth_id()
                            if not self.tracked_id_map[union_track[num].get_box_id()]:
                                self.tracked_id_map[union_track[num].get_box_id()] = True

                        elif i > most_left_old_idx:  # old중 가장 왼쪽에 있는 치아 보다 오른쪽에 있는 치아에 관하여
                            if num >= new_len:  # old면(uninon_track은 new + old니까 new_len보다 크거나 같은 idx의 uninon_track은 old임)
                                last_old_tooth_idx = self.tooth_id.index(union_track[num].get_tooth_id())
                                last_old_idx = i  # 기준이 될 변수 업데이트
                            else:  # new면
                                if last_old_tooth_idx - last_old_idx + i >= self.tooth_num or self.exist_tooth[
                                    last_old_tooth_idx - last_old_idx + i]:
                                    union_track = self.all_new_set(path,
                                                                   union_track)  # 치아번호를 할당하려했는데 tooth_num을 넘어가는 위치에서 탐지된 치아거나 이미 할당된 치아번호면 처음부터 다시 Setting
                                    break
                                else:  # 정상적으로 할당이 가능한 경우
                                    union_track[num].set_tooth_id(self.tooth_id[last_old_tooth_idx - last_old_idx + i])
                                    self.exist_tooth[last_old_tooth_idx - last_old_idx + i] = True
                                    self.id_num_matching[union_track[num].get_box_id()] = union_track[
                                        num].get_tooth_id()
                                    if not self.tracked_id_map[union_track[num].get_box_id()]:
                                        self.tracked_id_map[union_track[num].get_box_id()] = True
                                        # self.tracked_id = np.append(self.tracked_id, union_track[num].get_box_id())
                        else:
                            continue
        else:  # 새로운 Track이 없다면
            union_track = old_track

        end = time.time()
        elapsed = (end - start) * 1000
        print(f"elapsed time in set_union_track_1() {elapsed:.3f} ms")
        return union_track

    # 통계를 낸 이후에 치아번호를 식별하는 과정
    def set_union_track_2(self, new_track, old_track):
        start = time.time()
        norm_track, old_track = self.split_box_into_norm_and_old(old_track)
        new_track.extend(norm_track)
        new_len = len(new_track)
        old_len = len(old_track)

        remove_union_idx = [False] * (new_len + old_len)  # 오탐이라고 판단하여 무시할 치아 index를 기록

        if new_len != 0:  # 새로운 Track이 감지됐다면
            union_track = new_track + old_track
            if len(union_track) < 3:  # 3이하면 path계산불가
                return -1, -1
            if old_len == 0:  # old가 하나도 없는 경우
                return -2, -2

            path = self.cal_path(union_track)
            exist_tooth_num, most_left_old, most_left_old_idx = self.extract_exist_tooth_num(union_track, path, new_len)
            most_left_old_tooth_id = union_track[most_left_old].get_tooth_id()
            most_left_old_tooth_idx = self.tooth_id.index(most_left_old_tooth_id)

            last_old_tooth_idx, last_old_idx, tooth_standard = self.calculate_tooth_standard(
                most_left_old_tooth_idx, most_left_old_idx
            )

            if old_len >= 3:
                transform_matrix = self.calculate_transform_matrix(old_track)

                split_old_new = self.split_tracks_and_points(union_track, path, new_len, transform_matrix)
                split_tooth_num = self.split_tooth_num(exist_tooth_num)

                remove_union_idx = self.assign_tooth_ids(
                    split_old_new, split_tooth_num, path, union_track, remove_union_idx
                )
            else:
                remove_union_idx = self.simple_assign_tooth_ids(
                    union_track, path, last_old_idx, last_old_tooth_idx, tooth_standard, new_len, remove_union_idx
                )
        else:  # 새로운 Track이 없다면
            union_track = old_track
        end = time.time()
        elapsed = (end - start) * 1000
        print(f"elapsed time in set_union_track_2() {elapsed:.3f} ms")

        return remove_union_idx, union_track

    def calculate_transform_matrix(self, old_track):
        src_points, dst_points = [], []

        for track in old_track:
            tooth_id = track.get_tooth_id()
            x, y = track.get_xy()

            if tooth_id in self.most_common_box_avg:
                avg_x, avg_y = self.most_common_box_avg[tooth_id]
                src_points.append([x, y])
                dst_points.append([avg_x, avg_y])

        src_points = np.array(src_points, dtype='float32')
        dst_points = np.array(dst_points, dtype='float32')

        # Affine 변환 행렬 계산
        transform_matrix, inliers = cv2.estimateAffinePartial2D(src_points, dst_points)
        # print(transform_matrix)
        return transform_matrix

    def split_tracks_and_points(self, union_track, path, new_len, transform_matrix):
        split_old_new = []
        current_group = {}

        for i, num in enumerate(path):
            if num >= new_len:
                split_old_new.append(current_group)
                current_group = {}
            else:
                x, y = union_track[num].get_xy()
                point = np.array([[x, y]], dtype='float32')  # cv2.transform 함수는 (N, 2) 또는 (N, 3) 형식의 배열을 입력으로 받음
                transformed_point = cv2.transform(np.array([point]), transform_matrix)  # Affine 변환 적용
                x_prime, y_prime = transformed_point[0][0]
                current_group[i] = (x_prime, y_prime)

        split_old_new.append(current_group)
        return split_old_new

    def split_tooth_num(self, exist_tooth_num):
        split_tooth_num = []
        current_group = {}

        for id in self.tooth_id:
            if id in exist_tooth_num:
                split_tooth_num.append(current_group)
                current_group = {}
            else:
                x, y = self.most_common_box_avg[id]
                current_group[id] = (x, y)

        split_tooth_num.append(current_group)
        # print(split_tooth_num)
        return split_tooth_num

    def assign_tooth_ids(self, split_old_new, split_tooth_num, path, union_track, remove_union_idx):
        from scipy.spatial import distance

        for idx in range(len(split_tooth_num)):
            old_dict = split_old_new[idx]
            tooth_dict = split_tooth_num[idx]

            old_keys = sorted(old_dict.keys())
            tooth_keys = sorted(tooth_dict.keys(), key=lambda x: self.tooth_id.index(x))
            old_index_map = {key: idx for idx, key in enumerate(old_keys)}
            tooth_index_map = {key: idx for idx, key in enumerate(tooth_keys)}
            # print("old,tooooot", old_keys, tooth_keys)

            if not old_dict:
                continue
            if not tooth_dict:
                for o_k in old_keys:
                    remove_union_idx[o_k] = True
                continue

            old_coords = [old_dict[key] for key in old_keys]
            tooth_coords = [tooth_dict[key] for key in tooth_keys]
            dist_matrix = distance.cdist(old_coords, tooth_coords)
            t_d_len = len(tooth_dict)
            o_d_len = len(old_dict)
            start_idx = 0
            end_idx = start_idx + t_d_len - o_d_len

            for i_path in old_keys:
                if end_idx < start_idx:
                    o_d_len -= 1
                    end_idx = start_idx + t_d_len - o_d_len
                    remove_union_idx[path[i_path]] = True
                    continue
                else:
                    tooth_match_list = [tooth_keys[i] for i in range(start_idx, end_idx + 1)]
                    # print("tooth_match_list", tooth_match_list)

                    specific_indices = [tooth_index_map[key] for key in tooth_match_list]
                    old_idx = old_index_map[i_path]
                    # print(specific_indices, old_idx)
                    specific_distances = dist_matrix[old_idx, specific_indices]
                    min_specific_idx = np.argmin(specific_distances)
                    min_distance = specific_distances[min_specific_idx]
                    selected_tooth_key = tooth_match_list[min_specific_idx]
                    start_idx = start_idx + min_specific_idx + 1
                    t_d_len -= min_specific_idx + 1
                    o_d_len -= 1
                    end_idx = start_idx + t_d_len - o_d_len
                    union_track[path[i_path]].set_tooth_id(selected_tooth_key)
                    self.id_num_matching[union_track[path[i_path]].get_box_id()] = union_track[
                        path[i_path]].get_tooth_id()
                    if not self.tracked_id_map[union_track[path[i_path]].get_box_id()]:
                        self.tracked_id_map[union_track[path[i_path]].get_box_id()] = True

        return remove_union_idx

    def simple_assign_tooth_ids(self, union_track, path, last_old_idx, last_old_tooth_idx, tooth_standard, new_len,
                                remove_union_idx):
        for i, num in enumerate(path):
            if i < last_old_idx - last_old_tooth_idx:
                remove_union_idx[num] = True
            elif i < last_old_idx:
                union_track[num].set_tooth_id(self.tooth_id[tooth_standard + i])
                self.id_num_matching[union_track[num].get_box_id()] = union_track[num].get_tooth_id()
                if not self.tracked_id_map[union_track[num].get_box_id()]:
                    self.tracked_id_map[union_track[num].get_box_id()] = True
                    # self.tracked_id = np.append(self.tracked_id, union_track[num].get_box_id())
            elif i > last_old_idx:
                if num >= new_len:  # old면
                    last_old_tooth_idx = self.tooth_id.index(union_track[num].get_tooth_id())
                    last_old_idx = i
                    tooth_standard = last_old_tooth_idx - last_old_idx
                else:
                    if tooth_standard + i < self.tooth_num and not self.exist_tooth[tooth_standard + i]:
                        union_track[num].set_tooth_id(self.tooth_id[tooth_standard + i])
                        self.id_num_matching[union_track[num].get_box_id()] = union_track[num].get_tooth_id()
                        if not self.tracked_id_map[union_track[num].get_box_id()]:
                            self.tracked_id_map[union_track[num].get_box_id()] = True
                            # self.tracked_id = np.append(self.tracked_id, union_track[num].get_box_id())
                    elif tooth_standard + i < self.tooth_num and self.exist_tooth[tooth_standard + i]:
                        remove_union_idx[num] = True
                    elif tooth_standard + i >= self.tooth_num:
                        remove_union_idx[num] = True
            else:
                continue

        return remove_union_idx

    def extract_exist_tooth_num(self, union_track, path, new_len):
        exist_tooth_num = []
        most_left_old = -1
        most_left_old_idx = -1

        for i, num in enumerate(path):
            if num >= new_len:
                if most_left_old_idx == -1:
                    most_left_old = num  # 가장 왼쪽에 있는 old의 path에 들어있는 번호
                    most_left_old_idx = i  # 가장 왼쪽에 있는 old의 path에서의 idx
                if union_track[num].get_tooth_id() not in exist_tooth_num:
                    exist_tooth_num.append(union_track[num].get_tooth_id())
                # print(i, num, new_len, union_track[num].get_tooth_id(), union_track[num].get_box_id())

        # print("exist_tooth_num", exist_tooth_num)
        return exist_tooth_num, most_left_old, most_left_old_idx

    def calculate_tooth_standard(self, most_left_old_tooth_idx, most_left_old_idx):
        last_old_tooth_idx = most_left_old_tooth_idx
        last_old_idx = most_left_old_idx
        tooth_standard = last_old_tooth_idx - last_old_idx

        return last_old_tooth_idx, last_old_idx, tooth_standard

    # 처음 60Frame동안 tooth_id별로 box_id가 몇번씩 할당됐었는지 기록
    def countTidBid(self, tooth_id_list, bounding_box_id_list, cenx_list, ceny_list):
        for tooth_id, box_id, x, y in zip(tooth_id_list, bounding_box_id_list, cenx_list, ceny_list):
            self.count_tid_bid[tooth_id][box_id]["count"] += 1
            self.count_tid_bid[tooth_id][box_id]["x_sum"] += x
            self.count_tid_bid[tooth_id][box_id]["y_sum"] += y

    def initCurrentTidBid(self):  # 통계를 다 낸 이후, 각 치아번호 별로 가장 많이 나왔던 box_id를 찾는다.
        # print(self.count_tid_bid.items())
        for tooth_num, box_counter in self.count_tid_bid.items():
            self.current_tid_bid[tooth_num] = max(box_counter.items(), key=lambda item: item[1]['count'])[0]
            most_common_box_id = self.current_tid_bid[tooth_num]
            data = self.count_tid_bid[tooth_num][most_common_box_id]
            if data['count'] > 0:
                avg_x = data['x_sum'] / data['count']
                avg_y = data['y_sum'] / data['count']
                self.most_common_box_avg[tooth_num] = (avg_x, avg_y)
            else:
                self.most_common_box_avg[tooth_num] = None

    def init_tooth_id(self):  # 통계를 다 낸 이후에, 각 치아 번호 별로 가장 많이 나왔던 box_id를 할당하고, 통계에 사용한 변수들을 초기화
        # print("start init", self.id_count)
        self.exist_tooth = np.zeros(self.tooth_num, dtype=bool)
        self.id_num_matching = np.zeros(self.max_id_num, dtype=int)
        self.tracked_id_map = np.zeros(self.max_id_num, dtype=bool)
        self.id_count = np.zeros(self.max_id_num, dtype=int)

        for tooth_num, box_id in self.current_tid_bid.items():
            self.id_num_matching[box_id] = tooth_num
            self.id_count[box_id] = self.norm_old_sep + 3
            self.exist_tooth[self.tooth_id.index(tooth_num)] = True
            if not self.tracked_id_map[box_id]:
                self.tracked_id_map[box_id] = True
                # self.tracked_id = np.append(self.tracked_id, box_id)
        # 초기화
        self.do_init = False
        self.current_tid_bid = {}
        self.count_tid_bid = defaultdict(lambda: defaultdict(lambda: {"count": 0, "x_sum": 0, "y_sum": 0}))

    def frame_capture_worker(self, cap, frame_queue, stop_event):
        """Pipeline 1: Frame capture worker"""
        frame_counter = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video total frames: {total_frames}")

        while not stop_event.is_set():
            try:
                ret, frame = cap.read()

                if not ret:
                    break  # 루프 종료

                if not frame_queue.full():
                    # if frame_counter > 1000:
                    frame_queue.put((frame.copy(), frame_counter))
                    frame_counter += 1
                else:
                    print(f"Frame queue full, continue {frame_counter}")
                    continue

            except Exception as e:
                print(f"Frame capture error at frame {frame_counter}: {e}")
                import traceback
                traceback.print_exc()
                stop_event.set()  # 에러 발생시에도 종료 신호
                break

        print(f"Frame capture worker ended at frame {frame_counter}")

    def inference_worker(self, model, frame_queue, inference_queue, stop_event):
        """Pipeline 2: Model inference worker"""
        while not stop_event.is_set():
            try:
                frame_data = frame_queue.get(timeout=0.1)
                if frame_data is None:
                    print(f"Video ended at frame {frame_counter}/{total_frames}, stopping all workers...")
                    stop_event.set()  # 동영상 끝났을 때 모든 스레드 종료 신호
                    break

                frame, frame_counter = frame_data
                start_time = time.time()

                results = model.track(frame, persist=True, tracker="botsort.yaml", imgsz=[1088,1920], rect = True,
                                    show_boxes=False, iou=0.72, stream=False,conf=0.7, stream_buffer=False, half=True)

                end_time = time.time()
                yolo_time = (end_time - start_time) * 1000

                inference_queue.put((frame, results[0], frame_counter, yolo_time))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Inference worker error: {e}")
                import traceback
                traceback.print_exc()
                stop_event.set()  # 에러 발생시 모든 스레드 종료
                break

    def tooth_identification_worker(self, inference_queue, identification_queue, stop_event):
        """Pipeline 3: Tooth identification worker (최적화됨)"""
        do_init_count = 0

        # 성능 모니터링 변수
        preprocess_times = []
        separate_times = []
        total_frames = 0

        while not stop_event.is_set():
            try:
                inference_data = inference_queue.get(timeout=0.1)
                if inference_data is None:
                    continue

                frame, result, frame_counter, yolo_time = inference_data
                start_time = time.time()

                # Process tooth identification
                self.current_result = result

                if len(self.current_result.boxes.cls) == 0:
                    do_init_count = 0
                    self.do_init = True
                    identification_time = (time.time() - start_time) * 1000
                    identification_queue.put((frame, None, None, None, None, frame_counter, yolo_time, identification_time))
                else:
                    frame = self.current_result.orig_img

                    # 세부 시간 측정
                    preprocess_start = time.time()
                    box_id_list, mask_list, box_conf_list = self.preprocess()
                    preprocess_time = (time.time() - preprocess_start) * 1000
                    preprocess_times.append(preprocess_time)

                    new_track, old_track = self.separate_detection(box_id_list, mask_list, box_conf_list)


                    # Process tooth identification logic
                    if self.do_init:
                        if len(self.current_result.boxes.cls) >= 3:
                            union_track = self.set_unino_track_1(new_track, old_track)
                            mask_center_list_x, mask_center_list_y, tooth_id_list, bounding_box_id_list, b_conf_list = arrangement_for_id(union_track)
                            self.countTidBid(tooth_id_list, bounding_box_id_list, mask_center_list_x, mask_center_list_y)
                            do_init_count += 1

                            if do_init_count == self.test_frame_count:
                                self.initCurrentTidBid()
                                self.init_tooth_id()
                                do_init_count = 0
                        else:
                            mask_list, mask_center_list_x, mask_center_list_y, tooth_id_list = None, None, None, None
                    else:
                        remove_union_idx, union_track = self.set_union_track_2(new_track, old_track)
                        if union_track == -1:
                            mask_list, mask_center_list_x, mask_center_list_y, tooth_id_list = None, None, None, None
                        elif union_track == -2:
                            do_init_count = 0
                            self.do_init = True
                            mask_list, mask_center_list_x, mask_center_list_y, tooth_id_list = None, None, None, None
                        else:
                            mask_center_list_x, mask_center_list_y, tooth_id_list, bounding_box_id_list, b_conf_list = arrangement_for_id(union_track, remove_union_idx)

                    end_time = time.time()
                    identification_time = (end_time - start_time) * 1000
                    total_frames += 1

                    identification_queue.put((frame, mask_list, mask_center_list_x, mask_center_list_y, tooth_id_list, frame_counter, yolo_time, identification_time))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Tooth identification worker error: {e}")
                continue

    def plot_worker(self, identification_queue, plot_queue, stop_event):
        """Pipeline 4: Plot worker"""
        while not stop_event.is_set():
            try:
                identification_data = identification_queue.get(timeout=0.1)
                if identification_data is None:
                    # print(f"Video ended at frame {frame_counter}/{total_frames}, stopping all workers...")
                    # stop_event.set()  # 동영상 끝났을 때 모든 스레드 종료 신호
                    continue
                
                frame, mask_list, mask_center_list_x, mask_center_list_y, tooth_id_list, frame_counter, yolo_time, identification_time = identification_data
                start_time = time.time()
                
                # Execute plot function if we have data
                if mask_list is not None:
                    plotted_frame = self.plot(frame, frame_counter, mask_list, mask_center_list_x, mask_center_list_y, tooth_id_list)
                else:
                    plotted_frame = frame
                

                
                # Add frame number
                text = f'Frame: {frame_counter}'
                cv2.putText(plotted_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                end_time = time.time()
                plot_time = (end_time - start_time) * 1000
                
                plot_queue.put((plotted_frame, frame_counter, yolo_time, identification_time, plot_time))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Plot worker error: {e}")
                continue

    # Load the YOLOv8 model
    def run(self, model_path, video_path, output_path):
        global sum_execution_time, sum_yolo_time, sum_guide_time
        all_start_tieme = time.time()

        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("웹캠을 열 수 없습니다.")
            exit()
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 4-Pipeline setup
        frame_queue = queue.Queue(maxsize=10000)
        inference_queue = queue.Queue(maxsize=10000)
        identification_queue = queue.Queue(maxsize=10000)
        plot_queue = queue.Queue(maxsize=10000)
        stop_event = threading.Event()
        
        # Start all 4 pipeline threads
        threads = []
        
        # Pipeline 1: Frame capture
        frame_thread = threading.Thread(
            target=self.frame_capture_worker,
            args=(cap, frame_queue, stop_event)
        )
        frame_thread.daemon = True
        frame_thread.start()
        threads.append(frame_thread)
        
        # Pipeline 2: Inference
        inference_thread = threading.Thread(
            target=self.inference_worker,
            args=(model, frame_queue, inference_queue, stop_event)
        )
        inference_thread.daemon = True
        inference_thread.start()
        threads.append(inference_thread)
        
        # Pipeline 3: Tooth identification
        identification_thread = threading.Thread(
            target=self.tooth_identification_worker,
            args=(inference_queue, identification_queue, stop_event)
        )
        identification_thread.daemon = True
        identification_thread.start()
        threads.append(identification_thread)
        
        # Pipeline 4: Plot
        plot_thread = threading.Thread(
            target=self.plot_worker,
            args=(identification_queue, plot_queue, stop_event)
        )
        plot_thread.daemon = True
        plot_thread.start()
        threads.append(plot_thread)
        
        # Statistics
        total_frames = 0
        sum_yolo_time = 0
        sum_identification_time = 0
        sum_plot_time = 0
        
        # Main loop - just consume pipeline output and display
        try:
            while not stop_event.is_set():
                # if total_frames >= 500:  # Frame limit
                #     break

                try:
                    # Get completed frame from plot pipeline
                    plotted_frame, frame_counter, yolo_time, identification_time, plot_time = plot_queue.get(timeout=0.1)

                    # Update statistics
                    total_frames += 1
                    sum_yolo_time += yolo_time
                    sum_identification_time += identification_time
                    sum_plot_time += plot_time

                    # Write to video file
                    out.write(plotted_frame)

                    # Display frame
                    cv2.imshow("YOLO Object Tracking", plotted_frame)

                    # Check for exit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                except queue.Empty:
                    # No frames ready yet, check if stop_event is set
                    if stop_event.is_set():
                        print("Stop event detected, exiting main loop...")
                        break
                    continue
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        all_end_time = time.time()
        # Cleanup - stop all pipelines
        print("Shutting down pipelines...")
        stop_event.set()
        
        # Send sentinel values to all queues
        try:
            frame_queue.put(None, timeout=1)
            inference_queue.put(None, timeout=1) 
            identification_queue.put(None, timeout=1)
            plot_queue.put(None, timeout=1)
        except queue.Full:
            pass
        
        # Wait for all threads to finish
        for thread in threads:
            thread.join(timeout=2)
        
        # Print statistics
        if total_frames > 0:
            print(f"\n=== Pipeline Performance ===")
            print(f"Total frames processed: {total_frames}")
            print(f"Average YOLO time: {sum_yolo_time/total_frames:.1f} ms")
            print(f"Average Identification time: {sum_identification_time/total_frames:.1f} ms") 
            print(f"Average Plot time: {sum_plot_time/total_frames:.1f} ms")
            print(f"Total pipeline time: {(sum_yolo_time + sum_identification_time + sum_plot_time)/total_frames:.1f} ms")
            print(f"Overall FPS: {total_frames / (all_end_time - all_start_tieme):.2f} FPS")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        

    def save_frame_with_detections(self, orig_img, tooth_id_list, detections, output_image_path, output_txt_path):
        height, width, _ = orig_img.shape
        with open(output_txt_path, 'w') as f:
            for i in range(len(detections.cls)):
                tooth_id = self.u_num.index(tooth_id_list[i])
                cls = detections.cls[i]
                x1, y1, x2, y2 = detections.xyxy[i]
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height
                f.write(f"{int(tooth_id)} {x_center} {y_center} {bbox_width} {bbox_height}\n")

                ''' # Calculate the center of the bounding box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Put the tooth_id text at the center of the bounding box
                cv2.putText(orig_img, str(tooth_id), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        '''
        # Save image with bounding boxes and tooth_id text
        cv2.imwrite(output_image_path, orig_img)

    # plot 함수
    # def plot(self, frame, frame_counter, mask_list, mask_center_list_x, mask_center_list_y, tooth_id_list, created_dir):
    def plot(self, frame, frame_counter, mask_list, mask_center_list_x, mask_center_list_y, tooth_id_list):
        global sum_guide_time
        if self.show_m:
            mask_layer = np.zeros_like(frame)
            for mask in mask_list:
                mask = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask_layer, [mask], (0, 255, 0))
            frame = cv2.addWeighted(frame, 1, mask_layer, self.alpha, 0)
        if self.show_b:
            for i, d in enumerate(self.current_result.boxes):
                box: torch.Tensor = d.xyxy.squeeze().numpy()  # 가정: d.xyxy가 tensor 형태일 경우 .numpy()로 ndarray로 변환
                # box 좌표를 int로 변환 (OpenCV는 정수 좌표를 기대
                start_point = (int(box[0]), int(box[1]))  # x1, y1
                end_point = (int(box[2]), int(box[3]))  # x2, y2
                # 프레임에 박스를 그립니다.
                cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)  # 초록색 박스, 선 두께 2 
        if self.show_n:
            self.show_num(frame, mask_center_list_x, mask_center_list_y, tooth_id_list)
        if self.show_d:  # 점 표시
            for x, y in zip(mask_center_list_x, mask_center_list_y):
                center = (int(x), int(y))
                cv2.circle(frame, center, radius=4, color=(0, 0, 255), thickness=-1)
        if self.show_p:

            if frame_counter % 1 == 0:
                mask_layer = np.zeros_like(frame)
                for mask, tooth_id in zip(mask_list, tooth_id_list):
                    # if tooth_id != 16:
                    if (tooth_id == 34) or (tooth_id == 35) or (tooth_id == 36):
                        mask = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(mask_layer, [mask], (255, 255, 255))

                mask_layer1 = mask_layer[:, :, 0]

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                test_img = np.array(image)
                # extract_edges_img 결과를 컬러로 변환
                test_img2 = extract_edges_img(test_img, mask_layer1)
                edge_color = np.zeros_like(frame)  # BGR 형식으로 생성
                edge_pixels = test_img2 > 0
                edge_color[edge_pixels] = [0, 255, 0]  # 초록색 (BGR)

                # 블렌딩
                alpha = 0.5
                frame[edge_pixels] = (1 - alpha) * frame[edge_pixels] + alpha * edge_color[edge_pixels]

        if len(mask_center_list_x) >= 5:
            if self.guding_2D:  # 타원 표시

                # 2D 알고리즘
                pass

            if self.guding_3D:  # 가이딩 표시
                target_teeth = [34, 35, 36]
                img_idx = None

                # dir_path = created_dir + '/frame' + str(frame_counter)
                valid_tooth_count = 0
                if frame_counter%1 == 0:
                # if frame_counter%3 == 1:
                # if frame_counter > 380:
                    mask_layer = np.zeros_like(frame)
                    for mask, tooth_id in zip(mask_list, tooth_id_list):
                        # if tooth_id != 16:
                        if tooth_id in target_teeth:
                            mask = np.array(mask, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(mask_layer, [mask], (255, 255, 255))
                            valid_tooth_count += 1

                    if valid_tooth_count == 3:
                        mask_layer1 = mask_layer[:,:,0]

                        sTime = time.time()
                        # self.transformed_image, self.img_idx, self.most_sim_idx = self.guide_implant3D(mask_layer1, mask_center_list_x, mask_center_list_y, tooth_id_list, frame, dir_path)
                        # 여기 오류 왜 나는지 분석 필요
                        transformed_image, img_idx, most_sim_idx = self.guide_implant3D(mask_layer1, mask_center_list_x, mask_center_list_y, tooth_id_list, frame)
                        if transformed_image is not None:
                            (self.transformed_image, self.img_idx, self.most_sim_idx) = (transformed_image, img_idx, most_sim_idx)

                            mask = np.isin(tooth_id_list, target_teeth)  # Boolean 마스크 생성
                            if np.any(mask):  # 해당 치아가 하나라도 감지된 경우
                                self.implant_x = np.mean(mask_center_list_x[mask])
                                self.implant_y = np.mean(mask_center_list_y[mask])

                        eTime = time.time()

                        guid_time = (eTime-sTime)*1000
                        sum_guide_time += guid_time
                        print(f"Guide 실행 시간: {guid_time:.3f} ms")

                # Apply transformed image overlay if available
                if img_idx is not None:
                    alpha = 0.5
                    frame[self.img_idx[:, 0], self.img_idx[:, 1]] = (1 - alpha) * frame[self.img_idx[:, 0], self.img_idx[:, 1]] + \
                        alpha * self.transformed_image[self.img_idx[:, 0], self.img_idx[:, 1]]
                else:
                    # 이전 프레임 데이터가 있고, 현재 프레임에서 target_teeth가 감지된 경우에만 적용
                    if (self.img_idx is not None and self.transformed_image is not None and
                        self.implant_x is not None and self.implant_y is not None):

                        mask = np.isin(tooth_id_list, target_teeth)

                        if np.any(mask):
                            # 현재 프레임의 치아 중심 계산
                            temp_x = np.mean(mask_center_list_x[mask])
                            temp_y = np.mean(mask_center_list_y[mask])

                            # 이동 벡터 계산
                            offset_y = temp_y - self.implant_y
                            offset_x = temp_x - self.implant_x

                            print(f"이동 벡터: y={offset_y:.2f}, x={offset_x:.2f}")

                            # 이전 프레임의 img_idx를 현재 위치로 이동
                            img_idx = self.img_idx + np.round([offset_y, offset_x]).astype(int)

                            # 이미지 경계 체크
                            height, width = frame.shape[:2]
                            valid_mask = ((img_idx[:, 0] >= 0) & (img_idx[:, 0] < height) &
                                         (img_idx[:, 1] >= 0) & (img_idx[:, 1] < width))

                            if np.any(valid_mask):
                                valid_img_idx = img_idx[valid_mask]
                                valid_original_idx = self.img_idx[valid_mask]

                                alpha = 0.5
                                frame[valid_img_idx[:, 0], valid_img_idx[:, 1]] = \
                                    (1 - alpha) * frame[valid_img_idx[:, 0], valid_img_idx[:, 1]] + \
                                    alpha * self.transformed_image[valid_original_idx[:, 0], valid_original_idx[:, 1]]



        return frame

    # uninon_track으로 부터 정보를 뽑아낸다. mask중심점 리스트, tooth_id 리스트, box_id 리스트, box_conf리스트
    # 순서는 맨 처음 YOLO모델이 Detect한 객체 순서대로 정렬했다.

    # Union_track의 모든 객체들을 Path 순서에 맞게 처음부터 다시 치아번호를 부여
    def all_new_set(self, path, union_track):
        self.exist_tooth = np.zeros(self.tooth_num, dtype=bool)
        for i, num in enumerate(path):
            union_track[num].set_tooth_id(self.tooth_id[i])
            self.id_num_matching[union_track[num].get_box_id()] = self.tooth_id[i]
            self.exist_tooth[i] = True
            if union_track[num].get_is_new():
                if not self.tracked_id_map[union_track[num].get_box_id()]:
                    self.tracked_id_map[union_track[num].get_box_id()] = True
                    # self.tracked_id = np.append(self.tracked_id, union_track[num].get_box_id())
        return union_track

    # 치아 번호 PLOT
    def show_num(self, frame, mask_center_list_x, mask_center_list_y, tooth_id_list):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 0)
        thickness = 2
        positions = np.column_stack([mask_center_list_x, mask_center_list_y, tooth_id_list])
        # 각 좌표에 tooth_id 쓰기
        for x, y, tooth_id in positions:
            # (x, y) 좌표에 tooth_id 문자열을 적습니다. (정수 부분만)
            cv2.putText(frame, str(int(tooth_id)), (int(x), int(y)), font, font_scale, color, thickness)

    def cal_path(self, union_track: list[TrackObject]):
        get_xy_ufunc = np.frompyfunc(lambda obj: obj.get_xy(), 1, 2)

        center_x_array, center_y_array = get_xy_ufunc(union_track)
        center_x_list = np.array(center_x_array, dtype=float)
        center_y_list = np.array(center_y_array, dtype=float)

        dx = center_x_list.reshape(-1, 1) - center_x_list
        dy = center_y_list.reshape(-1, 1) - center_y_list
        d = np.sqrt((dx ** 2) + (dy ** 2))
        arg_d = np.argsort(d)

        degree = ((np.arctan2(dy, dx) * 180) / np.pi) % 360
        cond = (degree - degree[arg_d[:, 0], arg_d[:, 1]].reshape(-1, 1)) % 360
        d3 = d.copy()
        d[((cond > 270) | (cond < 90)) & (cond != 0)] = 10000 * np.abs(
            180 - cond[((cond > 270) | (cond < 90)) & (cond != 0)])
        d[arg_d[:, 0], arg_d[:, 0]] = d3[arg_d[:, 0], arg_d[:, 0]]
        arg_d2 = np.argsort(d)
        edge = arg_d2[:, 0:3]
        sort_d = np.sort(d)
        start_p = arg_d2[np.argsort(sort_d[:, 2])[-2:], 0]
        start_p = start_p[np.argmin(center_x_list[start_p])]

        return find_path(d, edge, start_p)



    # 임플란트 3D 가이딩
    # def guide_implant3D(self, mask_layer1, center_x_list: np.ndarray, center_y_list: np.ndarray, tooth_id_list: np.ndarray, frame, dir_path):  # 임플란트 가이딩
    def guide_implant3D(self, mask_layer1, center_x_list: np.ndarray, center_y_list: np.ndarray, tooth_id_list: np.ndarray, frame):  # 임플란트 가이딩
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        test_img = np.array(image)

        test_img2 = extract_edges_img(test_img, mask_layer1)
        test_img3 = extract_contours(test_img2)
        d1 = np.argwhere(test_img3>0).T
        d1_flipped = d1[::-1]  # x, y 순서로 변환
        
        test_img2_area = np.sum(test_img2==255)
        
        # os.makedirs(dir_path, exist_ok=True)
        
        # cv2.imwrite(dir_path + "/frame.jpg", test_img2)
        # cv2.imwrite(dir_path + "/frame.jpg", test_img3)
        
        min_dist = 1000
        
        if self.most_sim_idx is not None:
            prev_dist_hu = compare_hu_moments(self.views_hu_moments, self.views_hu_moments[self.most_sim_idx])
            indices = np.argsort(prev_dist_hu)
            
            rets = []
            dists = []
            d2s = []
            
            # with open(dir_path + '/output_using_prev.txt', 'w') as file:

            start = time.time()
            for i, idx in enumerate(indices[:5]):   # 110ms
                
                img_area = self.view_areas[idx]
                d2 = self.contour_pros[idx][:, :self.contour_length[idx]-1]
                # cv2.imwrite(dir_path + "/prev_view%d_%d.jpg" % (i, idx), self.views[idx])
                
                d2_flipped = d2[::-1]

                ret, dist = icp(d1_flipped, d2_flipped, test_img2_area, img_area)   # 66ms
                
                rets.append(ret)
                dists.append(dist)
                d2s.append(d2)
                
                # file.write("view%d_%d's dist: %d \n" % (i, idx, dist) )
                
                if dist < 30:
                    print("Using previous frame's most similar view!")
                    break
                
            index = np.argmin(np.array(dists))      # 0.015ms
            # file.write("Best - index: %d, dist: %d" % (index, dists[index]))
                
                
            min_dist = dists[index]
            end = time.time()
            print(f"Previous frame's most similar view ICP time: {(end-start)*1000:.3f} ms")
        
        if min_dist > 30:
            # 모멘트 계산
            moments = cv2.moments(test_img2)
            # Hu 모멘트 계산
            hu_moments = cv2.HuMoments(moments).flatten()
            log_hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10) 
            # print(log_hu_moments)

            
            distances = compare_hu_moments(self.views_hu_moments, log_hu_moments)
            # idx = distances.argmin()
            indices = np.argsort(distances)
            
            rets = []
            dists = []
            d2s = []
            
            # os.makedirs(dir_path, exist_ok=True)
            
            # # cv2.imwrite(dir_path + "/frame.jpg", test_img2)
            # cv2.imwrite(dir_path + "/frame.jpg", test_img3)
            
            # with open(dir_path + '/output.txt', 'w') as file:
                
            for i, idx in enumerate(indices[:100]):   # 110ms

                img_area = self.view_areas[idx]
                d2 = self.contour_pros[idx][:, :self.contour_length[idx]-1]
                # cv2.imwrite(dir_path + "/view%d_%d.jpg" % (i, idx), self.views[idx])

                d2_flipped = d2[::-1]

                ret, dist = icp(d1_flipped, d2_flipped, test_img2_area, img_area)   # 66ms

                rets.append(ret)
                dists.append(dist)
                d2s.append(d2)

                # file.write("view%d_%d's dist: %d \n" % (i, idx, dist) )

                # if i > 5:
                if dist < 30:
                    break
                
            index = np.argmin(np.array(dists))      # 0.015ms
            min_dist = dists[index]
            # file.write("Best - index: %d, dist: %d" % (index, dists[index]))

        print("min_dist is ",min_dist)

        if min_dist >30:
            return None, None, None

        start = time.time()
        H = rets[index]

        # d = d2s[index]
        # image = self.views[indices[index]]
        image = self.pros[indices[index]]

        transformed_image = cv2.warpAffine(image, H, (0, 0))

        R = transformed_image[:,:,0]
        G = transformed_image[:,:,1]
        B = transformed_image[:,:,2]
        img_idx = np.argwhere((G>0) | (R>0) | (B>0))

        # img_idx = np.argwhere(transformed_image>0)

        # # 이미지 인덱스에 해당하는 부분만 블렌딩
        # alpha = 0.5
        # test_img[img_idx[:, 0], img_idx[:, 1]] = (1 - alpha) * test_img[img_idx[:, 0], img_idx[:, 1]] + \
        #                                         alpha * transformed_image[img_idx[:, 0], img_idx[:, 1]]


        # # 프레임을 비디오 파일에 쓰기
        # frame = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)

        transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
        end = time.time()
        print(f"임플란트 뷰 변환 시간: {(end-start)*1000:.3f} ms")
        
        
        return transformed_image, img_idx, indices[index]

