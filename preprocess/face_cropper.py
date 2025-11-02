import argparse
import logging
import os
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Process
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

mp.set_start_method('spawn', force=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FAN(object):
    def __init__(self, device='gpu'):
        import face_alignment
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        if out is None:
            return [0], 'kpt68'
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:, 0]);
            right = np.max(kpt[:, 0]);
            top = np.min(kpt[:, 1]);
            bottom = np.max(kpt[:, 1])
            bbox = [left, top, right, bottom]
            return bbox, 'kpt68'


class VideoCropper(object):
    def __init__(self,
                 root_dir: str = None,
                 output_dir: str = None,
                 target_size: int = 512,
                 scale: float = 1.5,
                 do_padding: bool = False,
                 num_workers: int = 8,
                 ):

        self.scale = scale
        self.do_padding = do_padding
        self.num_workers = num_workers
        self.target_size = target_size
        self.video_files = []
        self.cropped_files = []

        # instantiate face detector
        # self.face_detector = FAN()

        # update face bounding box
        self.min_left = np.inf
        self.min_top = np.inf
        self.max_right = -np.inf
        self.max_bottom = -np.inf

        # Common video file extensions
        video_extensions = {
            '.mp4', '.avi',
            # '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg', '.3gp', '.ts'
        }

        self.video_files = ["/root/autodl-tmp/datasets/tmp/V01_S0308_I00001235_P1618.mp4"]
        self.cropped_files = ["/root/autodl-tmp/datasets/tmp/V01_S0308_I00001235_P1618_cropped.mp4"]

        # # Walk through all folders and subfolders
        # for folder_path, subfolders, files in os.walk(root_dir):
        #     for file in files:
        #         # For example:
        #         # folder_path: /.../react-2025/listener/session22
        #         # file: Camera-2024-07-17-104338-104241.mp4
        #
        #         # Check if the file has a video extension
        #         file_ext = os.path.splitext(file)[1].lower()
        #         if file_ext in video_extensions:
        #             # Get full path and add to our list
        #             video_input_path = os.path.join(folder_path, file)
        #             self.video_files.append(video_input_path)
        #
        #             last_two_parts = Path(folder_path).parts[-2:]  # Get the last two elements
        #             video_output_dir = output_dir / Path(*last_two_parts)
        #             os.makedirs(video_output_dir, exist_ok=True)
        #             output_path = video_output_dir / file
        #             self.cropped_files.append(output_path)

    def get_face_bbox(self, image, face_detector=None):
        h, w, _ = image.shape
        if face_detector is None:
            face_detector = self.face_detector
        bbox, bbox_type = face_detector.run(image)
        if len(bbox) < 4:
            print('no face detected!')
            # left = 0;
            # right = h - 1;
            # top = 0;
            # bottom = w - 1
            left = -1;
            right = -1;
            top = -1;
            bottom = -1;
        else:
            left = bbox[0];
            right = bbox[2];
            top = bbox[1];
            bottom = bbox[3];

        return left, right, top, bottom, bbox_type

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        if type == 'kpt68':
            old_size = (right - left + bottom - top) / 2 * 1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        elif type == 'bbox':
            old_size = (right - left + bottom - top) / 2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.1])
        else:
            raise NotImplementedError
        return old_size, center

    def compute_stable_crop_region(self, all_left, all_right, all_top, all_bottom,
                                   frame_width, frame_height):
        """
        使用鲁棒统计方法计算稳定的裁剪区域
        保持裁剪框大小不变，但选择最优的中心位置
        """
        # 过滤掉无效检测（-1值）
        valid_indices = [i for i in range(len(all_left)) if all_left[i] > 0]
        if len(valid_indices) == 0:
            raise ValueError("No valid face detection found")

        valid_left = [all_left[i] for i in valid_indices]
        valid_right = [all_right[i] for i in valid_indices]
        valid_top = [all_top[i] for i in valid_indices]
        valid_bottom = [all_bottom[i] for i in valid_indices]

        # 计算每帧的人脸中心点
        center_x_list = [(l + r) / 2 for l, r in zip(valid_left, valid_right)]
        center_y_list = [(t + b) / 2 for t, b in zip(valid_top, valid_bottom)]

        # 计算人脸大小
        face_widths = [r - l for l, r in zip(valid_left, valid_right)]
        face_heights = [b - t for t, b in zip(valid_top, valid_bottom)]

        # 使用中位数而不是平均值来获得更稳定的人脸大小
        median_face_width = np.median(face_widths)
        median_face_height = np.median(face_heights)
        median_face_size = max(median_face_width, median_face_height)

        # 计算稳定的中心位置（使用trimmed mean去除极端值）
        stable_center_x = self.calculate_trimmed_mean(center_x_list, trim_percent=0.1)
        stable_center_y = self.calculate_trimmed_mean(center_y_list, trim_percent=0.1)

        # 应用缩放因子计算最终的裁剪框大小
        self.crop_size = int(median_face_size * self.scale)

        # 确保裁剪框不小于目标大小
        if self.crop_size < self.target_size:
            self.crop_size = self.target_size

        # 基于稳定中心计算裁剪边界
        left = int(stable_center_x - self.crop_size / 2)
        top = int(stable_center_y - self.crop_size / 2)
        right = left + self.crop_size
        bottom = top + self.crop_size

        # 边界检查和调整
        if left < 0:
            left = 0
            right = self.crop_size
        if top < 0:
            top = 0
            bottom = self.crop_size
        if right > frame_width:
            right = frame_width
            left = frame_width - self.crop_size
        if bottom > frame_height:
            bottom = frame_height
            top = frame_height - self.crop_size

        return left, top, right, bottom

    def calculate_trimmed_mean(self, values, trim_percent=0.1):
        """
        计算截尾均值，去除最高和最低的一定百分比的值
        这种方法比简单平均更能抵抗异常值的影响
        """
        if len(values) == 0:
            return 0

        sorted_values = sorted(values)
        trim_count = int(len(sorted_values) * trim_percent)

        # 如果数据点太少，至少去除最大和最小值各一个
        if trim_count == 0 and len(sorted_values) > 2:
            trim_count = 1

        if trim_count > 0:
            trimmed_values = sorted_values[trim_count:-trim_count]
        else:
            trimmed_values = sorted_values

        return sum(trimmed_values) / len(trimmed_values) if trimmed_values else sum(sorted_values) / len(sorted_values)

    def interpolate_missing_detections(self, all_left, all_right, all_top, all_bottom):
        """
        对缺失的检测结果进行智能插值
        """
        n_frames = len(all_left)

        # 找出所有有效检测的索引
        valid_indices = [i for i in range(n_frames) if all_left[i] > 0]

        if len(valid_indices) < 2:
            # 有效检测太少，无法插值
            return all_left, all_right, all_top, all_bottom

        # 对每个无效检测进行插值
        for i in range(n_frames):
            if all_left[i] <= 0:  # 无效检测
                # 找到前后最近的有效检测
                prev_valid = max([idx for idx in valid_indices if idx < i], default=None)
                next_valid = min([idx for idx in valid_indices if idx > i], default=None)

                if prev_valid is not None and next_valid is not None:
                    # 线性插值
                    alpha = (i - prev_valid) / (next_valid - prev_valid)
                    all_left[i] = all_left[prev_valid] * (1 - alpha) + all_left[next_valid] * alpha
                    all_right[i] = all_right[prev_valid] * (1 - alpha) + all_right[next_valid] * alpha
                    all_top[i] = all_top[prev_valid] * (1 - alpha) + all_top[next_valid] * alpha
                    all_bottom[i] = all_bottom[prev_valid] * (1 - alpha) + all_bottom[next_valid] * alpha
                elif prev_valid is not None:
                    # 使用前一个有效值
                    all_left[i] = all_left[prev_valid]
                    all_right[i] = all_right[prev_valid]
                    all_top[i] = all_top[prev_valid]
                    all_bottom[i] = all_bottom[prev_valid]
                elif next_valid is not None:
                    # 使用后一个有效值
                    all_left[i] = all_left[next_valid]
                    all_right[i] = all_right[next_valid]
                    all_top[i] = all_top[next_valid]
                    all_bottom[i] = all_bottom[next_valid]

        return all_left, all_right, all_top, all_bottom

    def face_cropping(self, input_paths, output_paths, record_queue):
        """
        #TODO Method 1:
        use the first frame of the video as the reference frame and crop the face from the reference frame.

        #Method 2:
        [1] calculate the width & height of all frames of the video (take the maximum value for now),
            and then do scaling
        [2] compute average coords (left, right, top, bottom) of all collected face bounding boxs,
            and then compute new center coordinates;
        """

        face_detector = FAN()
        for input_path, output_path in zip(input_paths, output_paths):

            try:
                # here raise a potential exception
                assert os.path.exists(input_path), (
                    "Input video file does not exist: {}".format(input_path))
                print(f"Input video file exist: {input_path}")

                cap = cv2.VideoCapture(input_path)
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                logger.info(f"num_frames: {num_frames}, "
                            f"frame_width: {frame_width}, "
                            f"frame_height: {frame_height}, "
                            f"fps: {fps}")

                # for dynamically save maximum face size
                max_size = 0

                left = self.min_left
                top = self.min_top
                right = self.max_right
                bottom = self.max_bottom

                prev_left, prev_top, prev_right, prev_bottom = left, top, right, bottom

                all_left = []
                all_right = []
                all_top = []
                all_bottom = []

                pbar = tqdm(total=num_frames, desc="Processing Frames",
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

                maximum_frame_count = 100
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    if frame_count > maximum_frame_count:
                        break

                    # frame_count += 1
                    left, right, top, bottom, bbox_type = self.get_face_bbox(frame, face_detector)

                    if left == -1 and prev_right > 0:
                        all_left.append(prev_left)
                        all_right.append(prev_right)
                        all_top.append(prev_top)
                        all_bottom.append(prev_bottom)
                        continue

                    # store all facial boundaries for average computation
                    # TODO how to optimize in consider 'outliers' in some frames
                    all_left.append(left)
                    all_right.append(right)
                    all_top.append(top)
                    all_bottom.append(bottom)

                    old_size, _ = self.bbox2point(left, right, top, bottom, bbox_type)
                    new_size = int(old_size * self.scale)
                    max_size = max(new_size, max_size)

                    pbar.update(1)

                cap.release()
                assert len(all_left) > 0, "No face detected in video: {}".format(input_path)

                self.crop_size = max_size
                # compute average of face boundaries
                left = int(sum(all_left) / len(all_left))
                top = int(sum(all_top) / len(all_top))
                right = int(sum(all_right) / len(all_right))
                bottom = int(sum(all_bottom) / len(all_bottom))

                if self.do_padding:
                    padding = 32  # TODO hyperparams
                    left = max(0, left - padding)
                    top = max(0, top - padding)
                    right = min(frame_width, right + padding)
                    bottom = min(frame_height, bottom + padding)

                # in case average bounding box size > cropping size
                width = right - left
                height = bottom - top
                max_side = max(width, height)
                if max_side > self.crop_size:
                    ratio = self.crop_size / max_side
                    left = int(left + (width - width * ratio) / 2)
                    top = int(top + (height - height * ratio) / 2)
                    right = int(left + self.crop_size)
                    bottom = int(top + self.crop_size)

                # compute the center coords
                center_x = (left + right) / 2
                center_y = (top + bottom) / 2
                # recompute the bounding box coords
                left = int(center_x - self.crop_size / 2)
                top = int(center_y - self.crop_size / 2)
                right = int(center_x + self.crop_size / 2)
                bottom = int(center_y + self.crop_size / 2)

                if left < 0:
                    left = 0
                    right = self.crop_size
                if top < 0:
                    top = 0
                    bottom = self.crop_size
                if right > frame_width:
                    right = frame_width
                    left = frame_width - self.crop_size
                if bottom > frame_height:
                    bottom = frame_height
                    top = frame_height - self.crop_size

                # TODO debug [Modification]
                # # 先进行插值处理缺失的检测
                # all_left, all_right, all_top, all_bottom = self.interpolate_missing_detections(
                #     all_left, all_right, all_top, all_bottom
                # )
                # # 使用新的稳定计算方法
                # left, top, right, bottom = self.compute_stable_crop_region(
                #     all_left, all_right, all_top, all_bottom,
                #     frame_width, frame_height
                # )

                # write video to /output_path
                self.video_write(input_path, output_path, left, right, top, bottom)
                logger.info(f"left: {left}, right: {right}, top: {top}, bottom: {bottom}")

            except Exception as e:
                error_type = type(e).__name__
                record_queue.put((input_path, e))
                logger.error(f"Error processing {input_path}: {error_type}: {e}")
                continue

    def video_write(self, input_path, output_path, left, right, top, bottom):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                              (self.target_size, self.target_size))

        # frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # here do cropping
            cropped_frame = frame[top:bottom, left:right]
            # if self.target_size != self.crop_size:
            cropped_frame = cv2.resize(cropped_frame, (self.target_size, self.target_size))
            out.write(cropped_frame)

            # frames.append(frame)
        cap.release()
        out.release()
        print(f"Finished processing {input_path} -> {output_path}")

        # here do cropping
        # cropped_frames = [f[top:bottom, left:right] for f in frames]
        # if self.target_size != self.crop_size:
        #     cropped_frames = [cv2.resize(frame, (self.target_size, self.target_size))
        #                       for frame in cropped_frames]

        # for frame in cropped_frames:
        #     out.write(frame)


def main(args):
    root_dir = args.root_dir
    output_dir = args.output_dir
    target_size = args.target_size
    scale = args.scale
    do_padding = args.do_padding
    num_workers = args.num_workers

    video_cropper = VideoCropper(
        root_dir=root_dir,
        output_dir=output_dir,
        target_size=target_size,
        scale=scale,
        do_padding=do_padding,
        num_workers=num_workers,
    )

    video_files = video_cropper.video_files
    cropped_files = video_cropper.cropped_files
    record_queue = Queue()

    video_cropper.face_cropping(
        input_paths=video_files,
        output_paths=cropped_files,
        record_queue=record_queue,
    )

    # video_batches = []
    # cropped_batches = []
    # bs = len(video_files) // num_workers
    # for i in range(num_workers):
    #     start_idx = i * bs
    #     video_batches.append(video_files[start_idx:start_idx + bs])
    #     cropped_batches.append(cropped_files[start_idx:start_idx + bs])
    #
    # processes = []
    # for i in range(num_workers):
    #     p = Process(target=video_cropper.face_cropping,
    #                 args=(video_batches[i], cropped_batches[i], record_queue))
    #     p.start()
    #     processes.append(p)
    #
    # for p in processes:
    #     p.join()
    # logger.info("All processing workers have completed")

    # logger.info("Saving error messages to .txt")
    # while True:
    #     try:
    #         record = record_queue.get(block=False)
    #         # save
    #         with open(error_file_path, "a") as f:
    #             f.write(f"{record[0]}: {record[1]}\n")
    #     except:
    #         break
    #     logger.error(f"Error processing {record[0]}: {record[1]}")


def check_frame_size():
    src_video_path = "/path/to/V01_S1881_I00000189_P2767_cropped.mp4"
    cap = cv2.VideoCapture(src_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(f"fps: {fps}, height: {height}, width: {width}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face cropping')
    # root_dir: path/to/react-2025
    parser.add_argument('--root_dir', type=str,
                        default='',
                        help="root directory")
    # output_dir: path/to/react-2025-cropped
    parser.add_argument('--output_dir', type=str,
                        default='',
                        help="output directory")
    parser.add_argument('--target_size', type=int,
                        default=512, help="target video frame size")
    parser.add_argument('--scale', type=float, default=1.5,
                        help="scale of face bounding box")
    parser.add_argument('--do_padding', type=bool, default=False,
                        help="whether to do padding")
    parser.add_argument('--num_workers', type=int, default=12,
                        help="number of workers for data loading")

    args = parser.parse_args()

    main(args)


    # def load_resume_files(self):
    #     resume_input_paths = []
    #     resume_output_paths = []
    #
    #     raw_video_list = []
    #     raw_video_dir = \
    #         "/lustre/projects/Research_Project-T127204/xk219/projects/datasets/source_data/react-2025"
    #     # cropped_video_list = []
    #     # cropped_video_dir = \
    #     #     "/lustre/projects/Research_Project-T127204/xk219/projects/datasets/source_data/react-2025-cropped"
    #
    #     for folder_path, subfolders, files in os.walk(raw_video_dir):
    #         if len(files) > 0:
    #             video_path = [os.path.join(folder_path, file) for file in files]
    #             raw_video_list.extend(video_path)
    #
    #     # for folder_path, subfolders, files in os.walk(cropped_video_dir):
    #     #     if len(files) > 0:
    #     #         video_path = [os.path.join(folder_path, file) for file in files]
    #     #         cropped_video_list.extend(video_path)
    #
    #     # for raw_video_path in raw_video_list:
    #     #     cropped_video_id = raw_video_path.replace("react-2025", "react-2025-cropped")
    #     #     if cropped_video_id not in cropped_video_list:
    #     #         resume_input_paths.append(raw_video_path)
    #     #         resume_output_paths.append(cropped_video_id)
    #
    #     video_processed_list = []
    #     log_path = "/lustre/projects/Research_Project-T127204/xk219/projects/Human-AI/master/MAFRG/scripts/face.out"
    #     with open(log_path) as file:
    #         for line in file:
    #             if "Finished processing" in line:
    #                 if "no face detected!" in line:
    #                     line = line.replace("no face detected!", "")
    #                 # list = line.strip().split(" ")
    #                 try:
    #                     _, _, raw_video_path, _, cropped_video_path = line.strip().split(" ")
    #                 except Exception as e:
    #                     error_type = type(e).__name__
    #                     print(f"Error processing {raw_video_path}: {error_type}: {e}")
    #                     # print(f"line content: {line.strip()}")
    #                 video_processed_list.append(raw_video_path)
    #     file.close()
    #
    #     for raw_video_path in raw_video_list:
    #         if raw_video_path not in video_processed_list:
    #             resume_input_paths.append(raw_video_path)
    #             resume_output_paths.append(raw_video_path.replace("react-2025", "react-2025-cropped"))
    #
    #     self.video_files = resume_input_paths
    #     self.cropped_files = resume_output_paths
