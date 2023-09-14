
# 1. 휴먼 인식수
## 2,3번과 함께


# edge에서 run result 옮기고

# TODO 20230914 SYH
python test_det_result_count_human.py mot --save_dir /media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/accre_output


# 2. 검출 정확도

# TODO 20230914 SYH
# nms 처리된 결과
python test_det_result_mAP.py mot --save_dir /media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/accre_output_mAP

# 3. 검출 속도
## 1번과 함께

# 4. MOTA 분석 코드

python eval_mot.py


