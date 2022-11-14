
# 1.
## 2,3번과 함께


# edge에서 run result 옮기고


python test_det_result_count_human.py mot --save_dir /media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/accre_output


# 2.

# nms 처리된 결과
python test_det_result_mAP.py mot --save_dir /media/syh/ssd2/SynologyDrive/DB/인증시험용_데이터/accre_output_mAP

# 3.
## 1번과 함께

# 4. MOTA 분석 코드

python eval_mot.py


