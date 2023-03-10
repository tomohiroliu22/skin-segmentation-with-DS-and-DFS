# U-Net with Deep Supervision and Deep Feature Sharing
python main.py --dataroot ./datasets --name experiment_DFS_DS_1 --model DFS_w_DS --phase train --fold 0
python main.py --dataroot ./datasets --name experiment_DFS_DS_2 --model DFS_w_DS --phase train --fold 1
python main.py --dataroot ./datasets --name experiment_DFS_DS_3 --model DFS_w_DS --phase train --fold 2
python main.py --dataroot ./datasets --name experiment_DFS_DS_4 --model DFS_w_DS --phase train --fold 3
python main.py --dataroot ./datasets --name experiment_DFS_DS_5 --model DFS_w_DS --phase train --fold 4

# U-Net with Deep Feature Sharing
python main.py --dataroot ./datasets --name experiment_DFS_1 --model DFS --phase train --fold 0
python main.py --dataroot ./datasets --name experiment_DFS_2 --model DFS --phase train --fold 1
python main.py --dataroot ./datasets --name experiment_DFS_3 --model DFS --phase train --fold 2
python main.py --dataroot ./datasets --name experiment_DFS_4 --model DFS --phase train --fold 3
python main.py --dataroot ./datasets --name experiment_DFS_5 --model DFS --phase train --fold 4

# U-Net with Deep Supervision
python main.py --dataroot ./datasets --name experiment_DS_1 --model DS --phase train --fold 0
python main.py --dataroot ./datasets --name experiment_DS_2 --model DS --phase train --fold 1
python main.py --dataroot ./datasets --name experiment_DS_3 --model DS --phase train --fold 2
python main.py --dataroot ./datasets --name experiment_DS_4 --model DS --phase train --fold 3
python main.py --dataroot ./datasets --name experiment_DS_5 --model DS --phase train --fold 4

# U-Net
python main.py --dataroot ./datasets --name experiment_NONE_1 --model NONE --phase train --fold 0
python main.py --dataroot ./datasets --name experiment_NONE_2 --model NONE --phase train --fold 1
python main.py --dataroot ./datasets --name experiment_NONE_3 --model NONE --phase train --fold 2
python main.py --dataroot ./datasets --name experiment_NONE_4 --model NONE --phase train --fold 3
python main.py --dataroot ./datasets --name experiment_NONE_5 --model NONE --phase train --fold 4