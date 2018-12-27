import simplejson as json
from collections import Counter
import numpy as np
from numpy import ma
from pykalman import KalmanFilter
import ast

class ProcessOpenPose2DData():
    def __init__(self, processFromStart=False, read_checkpoints=True):

        if processFromStart:
            with open('../openpose/output/jsonfiles.txt', 'r') as f:
                json_filenames = f.readlines()
            self.json_filenames = [filename[:-1] for filename in json_filenames]  # Remove \n character from filenames

            with open('../openpose/data/walking/video_filenames.txt', 'r') as f2:
                video_filenames = f2.readlines()
            self.video_filenames = [filename[:-5] for filename in video_filenames]  # Remove \n character from filenames

            self.files = [[json_file for json_file in json_filenames if json_file.startswith(file)] for file in
                          self.video_filenames]

            self.initial_process_of_data()
        else:
            with open('json_data.txt', 'r') as f:
                self.json_data = ast.literal_eval(f.read())

        self.keypoint_mapping = [
            "Nose",
            "Neck",
            "RShoulder",
            "RElbow",
            "RWrist",
            "LShoulder",
            "LElbow",
            "LWrist",
            "MidHip",
            "RHip",
            "RKnee",
            "RAnkle",
            "LHip",
            "LKnee",
            "LAnkle",
            "REye",
            "LEye",
            "REar",
            "LEar",
            "LBigToe",
            "LSmallToe",
            "LHeel",
            "RBigToe",
            "RSmallToe",
            "RHeel"]

        self.no_videos = len(self.json_data)
        if read_checkpoints:
            with open('keypoints.txt', 'r') as f:
                self.keypoints = ast.literal_eval(f.read())



    def initial_process_of_data(self):
        # Filter out json files that are empty, providing all following are empty
        self.json_data = [['../openpose/output/' + filename for filename in file_set] for file_set in self.files]
        non_empty_jsons = []
        for fileset in self.json_data:
            non_empty_set = []
            for file in fileset:
                with open(file, 'r') as f:
                    data = json.load(f)
                if data['people'] != []:
                    non_empty_set.append(file)
                f.close()
            non_empty_jsons.append(non_empty_set)
        self.json_data = non_empty_jsons

        # Are there any keypoints collected for json files with < 25 joints
        without_25 = []
        for fileset in self.json_data:
            for file in fileset:
                f = open(file, 'r')
                data = json.load(f)
                if len(data['people'][0]['pose_keypoints_2d']) != 75:
                    without_25.append(file, len(data['people'][0]['pose_keypoints_2d']))
                f.close()
        print('No. keypoint sets without 25 keypoints collected ' + str(len(without_25)))  # 0


    def check_keypoints(filename):
        f = open(filename, 'r')
        data = json.load(f)
        if data['people'] != []:
            pose_keypoints = data['people'][0]['pose_keypoints_2d']
        else:
            pose_keypoints = []
        return pose_keypoints

    def generate_empty_keypoints_dict(to_examine, to_examine_files):
        empty_keypoints = {}
        for i, fileset in enumerate(to_examine_files):
            curr_empty_keypoints = []
            for file_no, file in enumerate(fileset):
                with open(file, 'r') as f:
                    data = json.load(f)
                if data['people'] == []:
                    curr_empty_keypoints.append(file)
                f.close()
            empty_keypoints[to_examine[i]] = curr_empty_keypoints
        return empty_keypoints

    def remove_empty_keypoints(self):
        empty_keypoints = self.generate_empty_keypoints_dict()
        new_json_data = [[keypoint for keypoint in self.json_data[i] if keypoint not in empty_keypoints[i]] for i in
                         range(self.no_videos)]

    def check_keypoints_in_range(self, file_prefix, start_no, end_no):
        for i in range(start_no, end_no + 1):
            filename = file_prefix + str(i) + '_keypoints.json'
            self.check_keypoints(filename)

    def check_all_no_keypoints(self, file_prefix, start_no, end_no):
        for i in range(start_no, end_no):
            filename = file_prefix + str(start_no) + '_keypoints.json'
            f = open(filename, 'r')
            data = json.load(f)
            if data['people'] != []:
                print("false")
                return

    def get_frame_w_more_than_k_zeros(filename, k):
        f = open(filename, 'r')
        data = json.load(f)
        pose_keypoints = data['people'][0]['pose_keypoints_2d']
        counts = Counter(pose_keypoints)
        no_zeros = int(counts[0]/3)
        return no_zeros > k


    def find_first_na_data(self):
        is_continuous_na = []
        first_na_data = []  # Stores indexes of first occurrance of file per video set that has no keypoints data
        json_data = self.json_data
        for fileset in json_data:
            seen_na = False
            for i, file in enumerate(fileset):
                with open(file, 'r') as f:
                    data = json.load(f)
                if seen_na and data['people'] != []:
                    seen_na = False
                    f.close()
                    break
                if not (seen_na) and data['people'] == []:  # first empty keypoints json found, set in first_na_data
                    seen_na = True
                    first_na_data.append(i)
                f.close()
            is_continuous_na.append(seen_na)

        # Get rid of continuous NA keypoints
        no_videos = self.no_videos
        new_json_data = []
        for i in range(no_videos):
            curr_data = []
            if is_continuous_na[i]:  # get rid of first occurance of keypoints.json with no keypoints - last keypoints file for that fileset
                new_json_data.append([json_data[i][j] for j in range(first_na_data[i])])
            else:
                new_json_data.append(json_data[i])
        return first_na_data, new_json_data

    def remove_end_na_data(self):
        first_na_data, new_json_data = self.find_first_na_data(self.json_data).split()
        # json_data = self.json_data
        # to_examine = [i for i, fileset in enumerate(new_json_data) if json_data[i][first_na_data[i]] in fileset]
        to_examine_new = [37, 41, 100, 140, 149, 175, 221, 222, 228, 240, 241, 242, 246, 247, 266, 279, 281, 293, 294,
                          308, 312]

        new_json_data = [self.json_data[i][:first_na_data[i]] if (i in self.empty_keypoints and i not in to_examine_new) else self.json_data[i] for
            i in range(self.no_videos)]
        self.json_data = new_json_data

    def write_back(filename, data):
        with open(filename, 'w') as f:
            f.write(str(data))



    def delete_frames_w_more_than_half_zeros(self):
        frames_to_delete = [[keypoint for keypoint in output if self.check_keypoints(keypoint).count(0)>37] for output in self.json_data]
        new_json_data = [[keypoint for keypoint in self.json_data[i] if keypoint not in frames_to_delete[i]] for i in
                        range(self.no_videos)]
        keypoints = [[self.check_keypoints(keypoint_filename).flatten() for keypoint_filename in video_keypoints] for
                     video_keypoints in new_json_data]
        self.json_data = new_json_data

        self.write_back('json_data.txt', new_json_data) # Intermediary writeback


    def apply_kf_to_missing(self):
        # Treat remaining empty keypoints as full NAs we need to replace using kalman filter
        # keypoints = self.keypoints
        # keypoints = [[[item if item!=0 or isinstance(item, float) else np.ma for item in video_keypoints] for video_keypoints in video] for video in keypoints]
        # # anything with 0 =noise, 0.0 = actual angle
        # # self.write_back('keypoints.txt', keypoints)
        # self.keypoints = keypoints
        keypoints = self.keypoints
        keypoints = np.ma.asarray(keypoints)
        # Replace nans with ma.masked
        keypoints = [[[item if item != 0 or isinstance(item, float) else ma.masked for item in video_keypoints] for video_keypoints in video] for video in keypoints]
        kf = KalmanFilter.em(X=keypoints, n_iter=5)
        (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
        (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)
        return




processOpenPose = ProcessOpenPose2DData()
processOpenPose.apply_kf_to_missing()
print(processOpenPose.keypoints[0])

