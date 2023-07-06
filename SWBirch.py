import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale
from sklearn.cluster import Birch
import cv2
from memory_profiler import profile
from sklearn.cluster import Birch
from sklearn.datasets import make_blobs
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
from threading import stack_size
import time

n_decompositon = 1000  # divide the array 'reduced_distance' into 1000 parts along the axis=0
class NewBirch(Birch):
    @profile
    def predict(self, X):
        X = check_array(X, accept_sparse='csr')
        self._check_fit(X)
        '''
        try:
            reduced_distance = safe_sparse_dot(X, self.subcluster_centers_.T)  # the original code
            reduced_distance *= -2
            reduced_distance += self._subcluster_norms
            return self.subcluster_labels_[np.argmin(reduced_distance, axis=1)]
        except MemoryError:
        '''
        # assume that the matrix is dense
        argmin_list = np.array([], dtype=np.int)
        interval = int(np.ceil(X.shape[0] / n_decompositon))
        for index in range(0, n_decompositon - 1):
            lb = index * interval
            ub = (index + 1) * interval
            reduced_distance = safe_sparse_dot(X[lb:ub, :], self.subcluster_centers_.T)
            reduced_distance *= -2
            reduced_distance += self._subcluster_norms
            argmin_list = np.append(argmin_list, np.argmin(reduced_distance, axis=1))

        lb = (n_decompositon - 1) * interval
        reduced_distance = safe_sparse_dot(X[lb:X.shape[0], :], self.subcluster_centers_.T)
        reduced_distance *= -2
        reduced_distance += self._subcluster_norms
        argmin_list = np.append(argmin_list, np.argmin(reduced_distance, axis=1))

        return self.subcluster_labels_[argmin_list]
# 資料夾路徑
#input_folder = "D:\\testpython\\Experiment\\SKY LBP2"
input_folder = "D:\\testpython\\image\\GTA"
output_folder = "D:\\testpython\\Experiment\\GTA_BIRCH\\BIRCH_SW+GRAY_SKY_C6Output 0.075"
os.makedirs(output_folder, exist_ok=True)

image_files = os.listdir(input_folder)


for image_file in image_files:
    input_image_path = os.path.join(input_folder, image_file)
    img = cv2.imread(input_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img)
    stack_size(524288)
    size = stack_size()
    print(size)
    # Define window size
    window_size = (3, 3)

    # Get image size
    height, width= img.shape

    # Initialize vectorized
    vectorized = []

    # Sliding window processing
    for i in range(height):
        for j in range(width):
            # 確定窗口的邊界
            top = max(i - window_size[0] // 2, 0)
            bottom = min(i + window_size[0] // 2 + 1, height)
            left = max(j - window_size[1] // 2, 0)
            right = min(j + window_size[1] // 2 + 1, width)
                
            # 提取窗口區域
            window = img[top:bottom, left:right]
                
            # 將窗口區域轉換為一維向量
            window_vector = window.reshape((-1,))
                
            # 檢查窗口區域中是否有缺失值
            if window_vector.size < window_size[0] * window_size[1]:
                # 計算缺失值的數量
                num_missing = window_size[0] * window_size[1] - window_vector.size
                    
                # 將缺失的值用0填補
                missing_values = np.zeros(num_missing)
                    
                # 將缺失值添加到窗口區域向量中
                window_vector = np.concatenate((window_vector, missing_values))

            # Add window region vector to vectorized
            vectorized.append(window_vector)
    # Convert vectorized to a numpy array
    feature_img = np.array(vectorized)/255
    print('feature_img_array=',feature_img)
    print('feature_img=', feature_img.shape)
    print('img=', img.shape)
    # Parameters setting
    T = 0.075  # maximum radius of CF
    n = 2  # number of clusters (not essential)
    # BIRCH Clustering methodx
    #model = Birch(threshold=T, branching_factor=B, n_clusters=n)
    #model = BirchChunked(threshold=T, n_clusters=n)
    start_time = time.time()
    model = NewBirch(threshold=T, n_clusters=n)
    print('ready')
    model.fit(feature_img)
    print('finished')
    labels = model.fit_predict(feature_img)

    end_time = time.time()
    execution_time = end_time - start_time
    print('finished2')
    print("程式執行時間：", execution_time, "秒")
    #colors = np.random.randint(0, 255, (n, 3), dtype=np.uint8)
    colors = [(150, 150, 150), (0, 0, 0), (0, 195, 0), (255, 157, 0), (255, 176, 220), (255, 255, 0)]
    segmented_data = labels.reshape((height, width))
    segmented_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            label = segmented_data[i, j]
            segmented_image[i, j] = colors[label]
    save_path = os.path.join(output_folder, "segmented_" + image_file)
    cv2.imwrite(save_path, segmented_image)
