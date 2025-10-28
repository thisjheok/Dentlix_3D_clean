# from vid_stream_run_v11_debug_human import ToothTracker    # debug 모드
from vid_stream_run_v11_human import ToothTracker      # 일반 모드
import numpy as np

'''
#0105 환자 사전정보
isUp=False
tooth=[0,1,1,0,1,1,1,1, 1,1,1,1,1,1,1,0] #0105 #존재하는 치아
tooth_num=13 #0105 #존재하는 치아 갯수
implant_left=46 #빠진 치아 왼쪽 id
implant_right=44 #빠진 치아 오른쪽 id
'''

# 조윤성 환자 사전정보
isUp=False #치아가 Upper, Lower 구분
tooth=[0,1,1,1,1,1,1,1, 1,1,1,1,1,1,0,0] # test1
tooth_num = 13 #0090 #존재하는 치아 갯수 # test1

# tooth=[0,0,1,0,1,1,1,1, 1,1,1,1,1,0,0,0] # test3,4
# tooth_num=10 #0090 #존재하는 치아 갯수 # test3,4


implant_num = 37

#model_path="/data/MIN/runs/segment/train_UpDown_agm/weights/best.pt"
# model_path="/home/ubuntu/JW(IESW)/implant3/best.pt"
# model_path="./human_seg.pt"
model_path=("./best_n.pt")


input_video_path="./choi_test.mp4"

output_video_path="./output_test_choi.avi"
view_path = "./offline_processed_3D_model_choi.npz"

loaded_data = np.load(view_path)
        

"/data/MIN/min/errorFrame/test1" 

'''
class tooth_track():
    def __init__(self, isUp, tooth,tooth_num, implant_left, implant_right, 
                    show_m=True, show_b=True,show_d=False,show_n=True,guding_2D=False,show_guiding=False
                    ):
    '''
#def run(self, model_path, video_path, output_path, exception_frames_dir_path):    

run_object=ToothTracker(isUp, tooth, tooth_num, implant_num, loaded_data)
run_object.set_option(show_m=False, show_b=False, show_d=False, show_n=False, show_p=False, guding_2D=False, guding_3D=True, alpha=0.5)

# 영상으로 수행
run_object.run(model_path, input_video_path, output_video_path)

# 웹캠으로 수행

webcam_id = 0  # 기본 웹캠 ID

# run_object.run(model_path, webcam_id, output_video_path)

