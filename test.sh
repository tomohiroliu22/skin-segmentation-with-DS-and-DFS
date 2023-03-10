# U-Net with Deep Supervision and Deep Feature Sharing
python main.py --dataroot ./datasets --modelpath experiment_DFS_DS_1.pkl --model DFS_w_DS --phase test --fold 0
python main.py --dataroot ./datasets --modelpath experiment_DFS_DS_2.pkl --model DFS_w_DS --phase test --fold 1
python main.py --dataroot ./datasets --modelpath experiment_DFS_DS_3.pkl --model DFS_w_DS --phase test --fold 2
python main.py --dataroot ./datasets --modelpath experiment_DFS_DS_4.pkl --model DFS_w_DS --phase test --fold 3
python main.py --dataroot ./datasets --modelpath experiment_DFS_DS_5.pkl --model DFS_w_DS --phase test --fold 4

# U-Net with Deep Feature Sharing
python main.py --dataroot ./datasets --modelpath experiment_DFS_1.pkl --model DFS --phase test --fold 0
python main.py --dataroot ./datasets --modelpath experiment_DFS_2.pkl --model DFS --phase test --fold 1
python main.py --dataroot ./datasets --modelpath experiment_DFS_3.pkl --model DFS --phase test --fold 2
python main.py --dataroot ./datasets --modelpath experiment_DFS_4.pkl --model DFS --phase test --fold 3
python main.py --dataroot ./datasets --modelpath experiment_DFS_5.pkl --model DFS --phase test --fold 4

# U-Net with Deep Supervision
python main.py --dataroot ./datasets --modelpath experiment_DS_1.pkl --model DS --phase test --fold 0
python main.py --dataroot ./datasets --modelpath experiment_DS_2.pkl --model DS --phase test --fold 1
python main.py --dataroot ./datasets --modelpath experiment_DS_3.pkl --model DS --phase test --fold 2
python main.py --dataroot ./datasets --modelpath experiment_DS_4.pkl --model DS --phase test --fold 3
python main.py --dataroot ./datasets --modelpath experiment_DS_5.pkl --model DS --phase test --fold 4

# U-Net
python main.py --dataroot ./datasets --modelpath experiment_NONE_1.pkl --model NONE --phase test --fold 0
python main.py --dataroot ./datasets --modelpath experiment_NONE_2.pkl --model NONE --phase test --fold 1
python main.py --dataroot ./datasets --modelpath experiment_NONE_3.pkl --model NONE --phase test --fold 2
python main.py --dataroot ./datasets --modelpath experiment_NONE_4.pkl --model NONE --phase test --fold 3
python main.py --dataroot ./datasets --modelpath experiment_NONE_5.pkl --model NONE --phase test --fold 4